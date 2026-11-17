"""PyTorch reference pieces of Kokoro-82M, used by tools/export.py and tools/verify.py."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


def bake_weight_norm(model: nn.Module):
    """Fold weight norm into plain weights: new-style parametrizations and the legacy
    hook form (weight_g/weight_v, used by the AdainResBlk1d pools). With the legacy
    form, `.weight` is stale until a forward call, so reading it directly is wrong."""
    for m in model.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)
    return model

def up2(x):
    """Nearest x2 upsample as expand+reshape (OpenVINO GPU mis-fuses Interpolate+1x1 conv)."""
    B, C, T = x.shape
    return x[..., None].expand(B, C, T, 2).reshape(B, C, 2 * T)

def dw_up2(pool, x):
    """Depthwise ConvTranspose1d(k=3, s=2, p=1, output_padding=1) as polyphase
    elementwise ops (OpenVINO GPU miscomputes GroupConvolutionBackpropData):
      y[2t] = w1 x[t] + b,   y[2t+1] = w0 x[t+1] + w2 x[t] + b
    """
    w = pool.weight[:, 0, :]  # [C,3]
    b = pool.bias[None, :, None]
    w0, w1, w2 = (w[:, k][None, :, None] for k in range(3))
    x_next = F.pad(x[..., 1:], (0, 1))
    even = w1 * x + b
    odd = w0 * x_next + w2 * x + b
    B, C, T = x.shape
    return torch.stack([even, odd], dim=-1).reshape(B, C, 2 * T)

def snake(x, a):
    return x + (1 / a) * (torch.sin(a * x) ** 2)

class BertPart(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.bert, self.enc = km.bert, km.bert_encoder

    def forward(self, ids, attn):
        h = self.bert(ids, attention_mask=attn)
        return self.enc(h).transpose(-1, -2)

class ProsodyPart(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.p = km.predictor

    def forward(self, d_en, s):
        de = self.p.text_encoder
        L = d_en.shape[-1]
        sx = s[:, None, :].expand(1, L, s.shape[-1])  # [1,L,128]
        x = torch.cat([d_en.transpose(1, 2), sx], dim=-1)  # [1,L,640]
        for blk in de.lstms:
            if isinstance(blk, nn.LSTM):
                x, _ = blk(x)  # [1,L,512]
            else:  # AdaLayerNorm on [1,L,512]
                h = blk.fc(s)  # [1,1024]
                gamma, beta = torch.chunk(h, 2, dim=1)
                xn = F.layer_norm(x, (blk.channels,), eps=blk.eps)
                x = (1 + gamma[:, None]) * xn + beta[:, None]
                x = torch.cat([x, sx], dim=-1)
        d = x  # [1,L,640]
        y, _ = self.p.lstm(d)
        dur = torch.sigmoid(self.p.duration_proj(y)).sum(-1)  # [1,L]
        return d, dur

class F0NPart(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.p = km.predictor

    def forward(self, en, s):
        x, _ = self.p.shared(en.transpose(-1, -2))
        x = x.transpose(-1, -2)
        f0, n = x, x
        for b in self.p.F0:
            f0 = b(f0, s)
        for b in self.p.N:
            n = b(n, s)
        return self.p.F0_proj(f0).squeeze(1), self.p.N_proj(n).squeeze(1)

class TextEncPart(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.t = km.text_encoder

    def forward(self, ids):
        t = self.t
        x = t.embedding(ids).transpose(1, 2)
        for c in t.cnn:
            x = c(x)
        x, _ = t.lstm(x.transpose(1, 2))
        return x.transpose(1, 2)

# ---------------------------------------------------------------- plain decoder (dynamic length, no masks)
def _adain_plain(adain, x, s):
    h = adain.fc(s)[:, :, None]
    gamma, beta = torch.chunk(h, 2, dim=1)
    return (1 + gamma) * adain.norm(x) + beta  # InstanceNorm -> OpenVINO MVN


def _rb_plain(rb, x, s):
    for c1, c2, n1, n2, a1, a2 in zip(rb.convs1, rb.convs2, rb.adain1, rb.adain2, rb.alpha1, rb.alpha2):
        xt = c1(snake(_adain_plain(n1, x, s), a1))
        xt = c2(snake(_adain_plain(n2, xt, s), a2))
        x = xt + x
    return x


def _blk_plain(b, x, s):
    r = b.actv(_adain_plain(b.norm1, x, s))
    up = b.upsample_type != "none"
    if up:
        r = dw_up2(b.pool, r)
    r = b.conv2(b.actv(_adain_plain(b.norm2, b.conv1(r), s)))
    sc = up2(x) if up else x
    if b.learned_sc:
        sc = b.conv1x1(sc)
    return (r + sc) * (1 / math.sqrt(2))


class DecoderPlain(nn.Module):
    """Original decoder maths at the exact length (no padding). har is an input."""

    def __init__(self, km):
        super().__init__()
        self.dec = km.decoder
        self.gen = km.decoder.generator

    def forward(self, asr, f0, n, s, har):
        d, g = self.dec, self.gen
        F0 = d.F0_conv(f0[:, None])
        N = d.N_conv(n[:, None])
        x = _blk_plain(d.encode, torch.cat([asr, F0, N], dim=1), s)
        asr_res = d.asr_res(asr)
        for blk in d.decode:
            x = _blk_plain(blk, torch.cat([x, asr_res, F0, N], dim=1), s)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            xs_ = _rb_plain(g.noise_res[i], g.noise_convs[i](har), s)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + xs_
            acc = None
            for j in range(g.num_kernels):
                y = _rb_plain(g.resblocks[i * g.num_kernels + j], x, s)
                acc = y if acc is None else acc + y
            x = acc / g.num_kernels
        x = g.conv_post(F.leaky_relu(x))
        k = g.post_n_fft // 2 + 1
        return torch.exp(x[:, :k]), torch.sin(x[:, k:])

"""Kokoro-82M split into deployable parts.

Accelerator parts (static, bucketed shapes; padding is masked so results match the
unpadded original):
  BertPart      ids[1,L], attn[1,L]                     -> d_en[1,512,L]
  DecoderPart   asr[1,512,F], f0[1,2F], n[1,2F], s[1,128],
                har[1,22,120F+1], masks per rate          -> mag[1,11,120F+1], phase[1,11,120F+1]
CPU parts (dynamic shapes, contain LSTMs):
  ProsodyPart   d_en[1,512,L], s[1,128]                   -> d[1,L,640], dur[1,L]
  F0NPart       en[1,640,F], s[1,128]                     -> f0[1,2F], n[1,2F]
  TextEncPart   ids[1,L]                                  -> t_en[1,512,L]
CPU numpy: tokenize, duration -> alignment, harmonic source + STFT, iSTFT.

Rates (F = aligned frames at 40 Hz): 80 Hz = 2F, 800 Hz = 20F, 4800 Hz = 120F+1, audio = 600F.
"""
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


# ---------------------------------------------------------------- masked helpers
EPS_TRICK = True  # see m_adain
def m_adain(adain, x, s, m, mw):
    """AdaIN1d with instance-norm statistics over valid frames only; output masked.
    m:  [1,1,T] 0/1 mask
    mw: [1,T,1] mask / n_valid, so x @ mw is the masked mean as a matrix-vector
        product (fast on GPU/NPU, accumulates in fp32, unlike long ReduceMeans).
    Norm affine + style scale fold into one per-channel scale/shift."""
    h = adain.fc(s)[:, :, None]
    gamma, beta = torch.chunk(h, 2, dim=1)
    mean = torch.matmul(x, mw)  # [1,C,1]
    xc = (x - mean) * m
    var = torch.matmul(xc * xc, mw)
    w = adain.norm.weight[None, :, None]
    b = adain.norm.bias[None, :, None]
    # eps derived from an input (not a Constant): OpenVINO's GPU plugin pattern-matches
    # x*(y+eps)^-0.5 as precision-sensitive and then runs the following full-length
    # elementwise chain in f32 reference kernels (~3x slower).
    eps = (s[:, :1, None] * 0 + adain.norm.eps) if EPS_TRICK else adain.norm.eps
    scale = torch.rsqrt(var + eps) * w * (1 + gamma)
    shift = (1 + gamma) * b + beta
    return xc * scale + shift * m


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


def m_resblock1(rb, x, s, m, n):
    """istftnet.AdaINResBlock1, masked."""
    for c1, c2, n1, n2, a1, a2 in zip(rb.convs1, rb.convs2, rb.adain1, rb.adain2, rb.alpha1, rb.alpha2):
        xt = snake(m_adain(n1, x, s, m, n), a1) * m
        xt = c1(xt)
        xt = snake(m_adain(n2, xt, s, m, n), a2) * m
        xt = c2(xt)
        x = xt + x
    return x


def m_adainresblk(b, x, s, m_in, n_in, m_out, n_out):
    """istftnet.AdainResBlk1d, masked. m_in/m_out differ when the block upsamples."""
    r = b.actv(m_adain(b.norm1, x, s, m_in, n_in))
    if b.upsample_type != "none":
        r = dw_up2(b.pool, r) * m_out
    r = b.conv1(r)
    r = b.actv(m_adain(b.norm2, r, s, m_out, n_out))
    r = b.conv2(r)
    sc = up2(x) if b.upsample_type != "none" else x
    if b.learned_sc:
        sc = b.conv1x1(sc)
    return (r + sc) * (1 / math.sqrt(2))


# ---------------------------------------------------------------- accelerator parts
class BertPart(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.bert, self.enc = km.bert, km.bert_encoder

    def forward(self, ids, attn):
        h = self.bert(ids, attention_mask=attn)
        return self.enc(h).transpose(-1, -2)


class DecoderPart(nn.Module):
    """istftnet.Decoder + Generator up to (mag, phase). Harmonic source comes in as `har`."""

    def __init__(self, km):
        super().__init__()
        self.dec = km.decoder
        self.gen = km.decoder.generator

    def forward(self, asr, f0, n, s, har, m40, m80, m800, m4800):
        d, g = self.dec, self.gen
        mw = lambda m: (m / m.sum(-1, keepdim=True)).transpose(1, 2)  # [1,T,1]
        c40, c80, c800, c4800 = mw(m40), mw(m80), mw(m800), mw(m4800)
        F0 = d.F0_conv(f0[:, None]) * m40
        N = d.N_conv(n[:, None]) * m40
        x = torch.cat([asr, F0, N], dim=1)
        x = m_adainresblk(d.encode, x, s, m40, c40, m40, c40)
        asr_res = d.asr_res(asr)
        for blk in d.decode:
            x = torch.cat([x, asr_res, F0, N], dim=1)
            up = blk.upsample_type != "none"
            x = m_adainresblk(blk, x, s, m40, c40, m80 if up else m40, c80 if up else c40)
        # generator
        masks = [(m800, c800), (m4800, c4800)]
        for i in range(g.num_upsamples):
            mi, ci = masks[i]
            x = F.leaky_relu(x, 0.1) * (m80 if i == 0 else m800)
            xs_ = m_resblock1(g.noise_res[i], g.noise_convs[i](har) * mi, s, mi, ci)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = (x + xs_) * mi
            acc = None
            for j in range(g.num_kernels):
                y = m_resblock1(g.resblocks[i * g.num_kernels + j], x, s, mi, ci)
                acc = y if acc is None else acc + y
            x = acc / g.num_kernels
        x = F.leaky_relu(x) * m4800
        x = g.conv_post(x)
        k = g.post_n_fft // 2 + 1
        return torch.exp(x[:, :k]), torch.sin(x[:, k:])


# ---------------------------------------------------------------- CPU parts (batch 1, unpadded)
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


# ---------------------------------------------------------------- 4D decoder (GPU-friendly)
# Same maths as DecoderPart, but on [1,C,1,T] tensors with 2D convs (the layout
# OpenVINO GPU kernels/fusions are built for), and masks only where needed: right
# before every conv with kernel > 1. Norm statistics ignore padding via `mw`.
def _c2(c, x):
    return F.conv2d(x, c.weight[:, :, None, :], c.bias, stride=(1, c.stride[0]), padding=(0, c.padding[0]),
                    dilation=(1, c.dilation[0]), groups=c.groups)


def _ct2(c, x):
    return F.conv_transpose2d(x, c.weight[:, :, None, :], c.bias, stride=(1, c.stride[0]), padding=(0, c.padding[0]),
                              output_padding=(0, c.output_padding[0]), groups=c.groups)


EPS_TRICK = True  # see m_adain
NORM_MODE = "reduce"  # "reduce": masked ReduceMean (stays in GPU blocked layout); "matmul": x @ mw


def _adain4(adain, x, s, mw):
    """mw is (m, inv_r) for NORM_MODE=reduce, else the [1,1,T,1] weight column."""
    h = adain.fc(s)[:, :, None, None]
    gamma, beta = torch.chunk(h, 2, dim=1)
    if NORM_MODE == "reduce":
        ms = mw  # mask pre-scaled by 1/valid_fraction: mean(x * ms) == masked mean
        mean = (x * ms).mean(-1, keepdim=True)
        xc = x - mean
        var = (xc * xc * ms).mean(-1, keepdim=True)
    else:
        mean = torch.matmul(x, mw)  # [1,C,1,1]
        xc = x - mean
        var = torch.matmul(xc * xc, mw)
    eps = s[:, :1, None, None] * 0 + adain.norm.eps
    w = adain.norm.weight[None, :, None, None]
    b = adain.norm.bias[None, :, None, None]
    return xc * (torch.rsqrt(var + eps) * w * (1 + gamma)) + ((1 + gamma) * b + beta)


def _rb4(rb, x, s, m, mw):
    for c1, c2, n1, n2, a1, a2 in zip(rb.convs1, rb.convs2, rb.adain1, rb.adain2, rb.alpha1, rb.alpha2):
        a1, a2 = a1[..., None], a2[..., None]
        xt = _c2(c1, snake(_adain4(n1, x, s, mw), a1) * m)
        xt = _c2(c2, snake(_adain4(n2, xt, s, mw), a2) * m)
        x = xt + x
    return x


def _up2_4(x):
    B, C, H, T = x.shape
    return x[..., None].expand(B, C, H, T, 2).reshape(B, C, H, 2 * T)


def _dwup2_4(pool, x):
    w = pool.weight[:, 0, :]
    b = pool.bias[None, :, None, None]
    w0, w1, w2 = (w[:, k][None, :, None, None] for k in range(3))
    x_next = F.pad(x[..., 1:], (0, 1))
    B, C, H, T = x.shape
    return torch.stack([w1 * x + b, w0 * x_next + w2 * x + b], dim=-1).reshape(B, C, H, 2 * T)


def _blk4(b, x, s, m_in, mw_in, m_out, mw_out):
    r = b.actv(_adain4(b.norm1, x, s, mw_in))
    up = b.upsample_type != "none"
    if up:
        r = _dwup2_4(b.pool, r * m_in)
    r = _c2(b.conv1, r * m_out)
    r = b.actv(_adain4(b.norm2, r, s, mw_out))
    r = _c2(b.conv2, r * m_out)
    sc = _up2_4(x) if up else x
    if b.learned_sc:
        sc = _c2(b.conv1x1, sc)
    return (r + sc) * (1 / math.sqrt(2))


class DecoderPart4D(nn.Module):
    def __init__(self, km):
        super().__init__()
        self.dec = km.decoder
        self.gen = km.decoder.generator

    def forward(self, asr, f0, n, s, har, m40, m80, m800, m4800):
        d, g = self.dec, self.gen
        u = lambda t: t[:, :, None, :]  # [1,C,T] -> [1,C,1,T]
        asr, har = u(asr), u(har)
        m40, m80, m800, m4800 = u(m40), u(m80), u(m800), u(m4800)
        if NORM_MODE == "reduce":
            mw = lambda m: m / m.mean(-1, keepdim=True)
        else:
            mw = lambda m: (m / m.sum(-1, keepdim=True)).transpose(2, 3)  # [1,1,T,1]
        w40, w80, w800, w4800 = mw(m40), mw(m80), mw(m800), mw(m4800)
        F0 = _c2(d.F0_conv, f0[:, None, None, :])
        N = _c2(d.N_conv, n[:, None, None, :])
        x = torch.cat([asr, F0, N], dim=1)
        x = _blk4(d.encode, x, s, m40, w40, m40, w40)
        asr_res = _c2(d.asr_res[0], asr)
        for blk in d.decode:
            x = torch.cat([x, asr_res, F0, N], dim=1)
            up = blk.upsample_type != "none"
            x = _blk4(blk, x, s, m40, w40, m80 if up else m40, w80 if up else w40)
        rates = [(m80, m800, w800), (m800, m4800, w4800)]
        for i in range(g.num_upsamples):
            m_prev, mi, wi = rates[i]
            x = F.leaky_relu(x, 0.1)
            xs_ = _rb4(g.noise_res[i], _c2(g.noise_convs[i], har), s, mi, wi)
            x = _ct2(g.ups[i], x * m_prev)
            if i == g.num_upsamples - 1:
                x = F.pad(x, (1, 0, 0, 0), mode="reflect")
            x = x + xs_
            acc = None
            for j in range(g.num_kernels):
                y = _rb4(g.resblocks[i * g.num_kernels + j], x, s, mi, wi)
                acc = y if acc is None else acc + y
            x = acc / g.num_kernels
        x = _c2(g.conv_post, F.leaky_relu(x) * m4800)[:, :, 0, :]
        k = g.post_n_fft // 2 + 1
        return torch.exp(x[:, :k]), torch.sin(x[:, k:])


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

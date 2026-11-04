"""Check the split pipeline (torch parts + numpy DSP, padded to buckets) against the
original Kokoro forward. Noise is disabled in both so outputs are comparable."""
import numpy as np
import torch
from kokoro import KModel, KPipeline

import dsp
from kparts import BertPart, DecoderPart, F0NPart, ProsodyPart, TextEncPart, bake_weight_norm

torch.set_grad_enabled(False)
km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
TEXT = "Okay, but here's the bad news: the build broke on the main branch, and I'm not sure why."
ph = next(pipe(TEXT, voice="af_aoede")).phonemes
ids = [0, *[km.vocab[p] for p in ph if p in km.vocab], 0]
L = len(ids)
pack = pipe.load_voice("af_aoede")
ref_s = pack[L - 3]  # [1,256]

# ---- reference, noise off
_rand, _randn = torch.rand, torch.randn_like
torch.rand = lambda *a, **k: torch.zeros(*a, **{x: y for x, y in k.items() if x != "generator"})
torch.randn_like = lambda t, **k: torch.zeros_like(t)
audio_ref, dur_ref = km.forward_with_tokens(torch.LongTensor([ids]), ref_s, 1.0)
torch.rand, torch.randn_like = _rand, _randn
audio_ref = audio_ref.numpy()

# ---- split pipeline
bert, dec = BertPart(km), DecoderPart(km)
pros, f0n, tenc = ProsodyPart(km), F0NPart(km), TextEncPart(km)
s_pros, s_dec = ref_s[:, 128:], ref_s[:, :128]
lin = km.decoder.generator.m_source.l_linear


def run(Lb, Fb_of):
    ids_p = torch.zeros(1, Lb, dtype=torch.long); ids_p[0, :L] = torch.tensor(ids)
    attn = torch.zeros(1, Lb, dtype=torch.long); attn[0, :L] = 1
    d_en = bert(ids_p, attn)[:, :, :L]
    d, dur = pros(d_en, s_pros)
    pd = np.clip(np.round(dur[0].numpy()), 1, None).astype(np.int64)
    Fr = int(pd.sum())
    aln = np.zeros((L, Fr), np.float32); aln[np.repeat(np.arange(L), pd), np.arange(Fr)] = 1
    aln = torch.from_numpy(aln)[None]
    en = d.transpose(-1, -2) @ aln
    f0, n = f0n(en, s_pros)
    asr = tenc(torch.LongTensor([ids])) @ aln
    har = dsp.harmonic_source(f0[0].numpy(), lin.weight[0].numpy(), float(lin.bias[0]), noise=False)
    Fb = Fb_of(Fr)

    def pad(x, n):  # pad last dim to n
        return torch.nn.functional.pad(torch.as_tensor(x), (0, n - x.shape[-1]))

    def mask(valid, n):
        m = torch.zeros(1, 1, n); m[..., :valid] = 1; return m

    mag, phase = dec(pad(asr, Fb), pad(f0, 2 * Fb), pad(n, 2 * Fb), s_dec, pad(torch.from_numpy(har)[None], 120 * Fb + 1),
                     mask(Fr, Fb), mask(2 * Fr, 2 * Fb), mask(20 * Fr, 20 * Fb), mask(120 * Fr + 1, 120 * Fb + 1))
    T = 120 * Fr + 1
    return dsp.istft(mag[0, :, :T].numpy().astype(np.float64), phase[0, :, :T].numpy().astype(np.float64)), pd


def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return 10 * np.log10((b ** 2).sum() / max(((a - b) ** 2).sum(), 1e-20))


def lsd(a, b):
    """Log-spectral distance in dB (phase-insensitive), 1024/256 STFT."""
    def spec(x):
        x = torch.as_tensor(x[: min(len(a), len(b))])
        return torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs().clamp_min(1e-5)
    A, B = 20 * torch.log10(spec(a)), 20 * torch.log10(spec(b))
    return float(((A - B) ** 2).mean(0).sqrt().mean())


print(f"tokens={L} frames={int(dur_ref.sum())} ref_len={len(audio_ref)}")
# baseline: the original model against itself with different random noise
runs = []
for seed in (1, 2):
    torch.manual_seed(seed)
    runs.append(km.forward_with_tokens(torch.LongTensor([ids]), ref_s, 1.0)[0].numpy())
print(f"baseline  original vs original (different noise): LSD={lsd(runs[0], runs[1]):.2f} dB")
for name, Lb, fb in [("exact", L, lambda f: f), ("bucketed", 128, lambda f: 1 << (f - 1).bit_length())]:
    a, pd = run(Lb, fb)
    print(f"{name:9s} len={len(a)} dur_match={np.array_equal(pd, dur_ref.numpy())} SNR={snr(a, audio_ref):.1f} dB LSD={lsd(a, audio_ref):.2f} dB")

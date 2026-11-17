"""End-to-end fidelity of kokorolake against the reference PyTorch Kokoro-82M.

    .venv-tools/bin/python tools/verify.py

Both run with the harmonic-source noise disabled, so differences come only from kokorolake's
numerics (fp16 weights/activations, kernel reductions). As a yardstick, the reference is also
compared with itself across two noise seeds: that is the variation Kokoro has by design.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kokoro import KModel  # noqa: E402
from kokorolake.engine import Engine  # noqa: E402

TEXTS = ["Wait, what? All of the tests passed?",
         "Okay, but here's the bad news: the build broke on the main branch, and I'm not sure why.",
         "The meeting is at 3:45 PM on March 3rd; it costs $12.50."]


def lsd(a, b):
    """log-spectral distance (dB) over speech frames: 1024-point frames within 40 dB of the
    loudest frame, spectra floored at -80 dB re. the peak (so near-silent fp16 noise floors
    don't dominate)."""
    n = min(len(a), len(b))
    S = lambda v: np.abs(np.stack([np.fft.rfft(v[i:i + 1024] * np.hanning(1024)) for i in range(0, n - 1024, 256)]))
    A, B = S(a[:n]), S(b[:n])
    peak = max(A.max(), B.max())
    A, B = 20 * np.log10(np.maximum(A, peak * 1e-4)), 20 * np.log10(np.maximum(B, peak * 1e-4))
    e = 10 * np.log10((10 ** (B / 10)).sum(1))
    speech = e > e.max() - 40
    return float(np.sqrt(((A - B) ** 2).mean(1))[speech].mean())


def snr(a, ref):
    n = min(len(a), len(ref))
    return float(10 * np.log10((ref[:n] ** 2).sum() / max(((a[:n] - ref[:n]) ** 2).sum(), 1e-20)))


def main():
    torch.set_grad_enabled(False)
    km = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    e = Engine()
    pack = torch.from_numpy(e.voice)

    def ref(ps, noise_seed=None):
        ids = torch.LongTensor([[0, *[km.vocab[p] for p in ps if p in km.vocab], 0]])
        s = pack[ids.shape[1] - 3]
        if noise_seed is None:
            rand, randn = torch.rand, torch.randn_like
            torch.rand = lambda *a, **k: torch.zeros(*a)
            torch.randn_like = lambda t, **k: torch.zeros_like(t)
            try:
                audio, dur = km.forward_with_tokens(ids, s, 1.0)
            finally:
                torch.rand, torch.randn_like = rand, randn
        else:
            torch.manual_seed(noise_seed)
            audio, dur = km.forward_with_tokens(ids, s, 1.0)
        return audio.numpy(), dur.numpy()

    def f0_ref(ps):
        """reference F0 contour (80 Hz), via the same path as forward_with_tokens"""
        ids = torch.LongTensor([[0, *[km.vocab[p] for p in ps if p in km.vocab], 0]])
        s = pack[ids.shape[1] - 3]
        L = ids.shape[1]
        m = torch.zeros(1, L, dtype=torch.bool)
        d_en = km.bert_encoder(km.bert(ids, attention_mask=(~m).int())).transpose(-1, -2)
        d = km.predictor.text_encoder(d_en, s[:, 128:], torch.tensor([L]), m)
        x, _ = km.predictor.lstm(d)
        pd = torch.round(torch.sigmoid(km.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long().squeeze()
        idx = torch.repeat_interleave(torch.arange(L), pd)
        aln = torch.zeros(L, len(idx)); aln[idx, torch.arange(len(idx))] = 1
        f0, _ = km.predictor.F0Ntrain(d.transpose(-1, -2) @ aln[None], s[:, 128:])
        return f0[0].numpy()

    def f0_lake(ps):
        f = e.front
        ids = np.array([0, *[e.vocab[p] for p in ps if p in e.vocab], 0], np.int64)
        L = len(ids)
        s = e.voice[L - 3][0, 128:].astype(np.float32)
        dur, gx = f.prosody(e.kdec.bert(ids, readback=False), L, s)
        idx = np.repeat(np.arange(L), np.clip(np.round(dur), 1, None).astype(np.int64))
        return e.kdec.f0n(f.shared(gx, idx), s)[0]

    rows = []
    for text in TEXTS:
        ps = e.phonemize(text)
        e.noise = False
        a_ref, d_ref = ref(ps)
        a, d = e.synth_phonemes(ps)
        # with noise on, as both run in practice: kokorolake vs reference, and reference vs itself
        e.noise = True
        lake_on = [e.synth_phonemes(ps)[0] for _ in range(3)]
        refs_on = [ref(ps, seed)[0] for seed in (1, 2, 3)]
        fr, fl = f0_ref(ps), f0_lake(ps)
        voiced = (fr > 50) & (fl > 50)
        cents = np.abs(1200 * np.log2(fl[voiced] / fr[voiced]))
        rows.append(dict(text=text, frames=f"{int(d.sum())}/{int(d_ref.sum())}", lsd_off=lsd(a, a_ref),
                         lsd_on=np.mean([lsd(x, refs_on[0]) for x in lake_on]),
                         lsd_self=np.mean([lsd(refs_on[i], refs_on[0]) for i in (1, 2)]),
                         cents=float(np.median(cents)), cents95=float(np.percentile(cents, 95))))
    print(f"{'text':38s} {'frames':>8s} {'pitch err (cents)':>18s} {'LSD noise off':>14s} {'LSD noise on':>13s} {'ref vs ref':>11s}")
    for r in rows:
        print(f"{r['text'][:36]:38s} {r['frames']:>8s} {r['cents']:8.1f} (p95 {r['cents95']:4.1f}) {r['lsd_off']:14.2f} "
              f"{r['lsd_on']:13.2f} {r['lsd_self']:11.2f}")


if __name__ == "__main__":
    main()

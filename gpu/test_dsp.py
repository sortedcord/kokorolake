import os; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time; sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, pyopencl as cl, dsp
from kdec import KDecoder
d = KDecoder()
a = np.load("models/assets.npz"); lw, lb = a["lin_w"], float(a["lin_b"][0])
def lsd(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    S = lambda v: np.abs(np.stack([np.fft.rfft(v[i:i+1024]*np.hanning(1024)) for i in range(0, n-1024, 256)])) + 1e-5
    A, B = 20*np.log10(S(x)), 20*np.log10(S(y)); return float(np.sqrt(((A-B)**2).mean(1)).mean())
for f in ["in_first.npz", "bench_inputs.npz"]:
    x = np.load(f); F = x["asr"].shape[-1]; f0 = x["f0"][0]
    # 1) source + STFT: GPU vs numpy (noise off)
    Hn = dsp.harmonic_source(f0, lw, lb, noise=False, layout="tc")
    d.run(x["asr"][0], f0, x["n"][0], x["s"][0], source=(lw, lb, 0, 0))
    Hg = np.empty_like(Hn); cl.enqueue_copy(d.q, Hg, d.buf("har", 120*F+1, 32))
    mag_err = np.abs(Hg[:, :11].astype(np.float32) - Hn[:, :11].astype(np.float32)).max()
    big = Hn[:, :11].astype(np.float32) > 0.05
    ph = np.angle(np.exp(1j*(Hg[:, 11:22].astype(np.float32) - Hn[:, 11:22].astype(np.float32))))[big]
    # 2) full decoder: GPU source+iSTFT audio vs host source + numpy iSTFT (both noise off)
    mag, phs = d.run(x["asr"][0], f0, x["n"][0], x["s"][0], har=Hn)
    ref = dsp.istft(mag.astype(np.float64), phs.astype(np.float64))
    aud = d.run(x["asr"][0], f0, x["n"][0], x["s"][0], source=(lw, lb, 0, 0), out="audio")
    # 3) iSTFT alone: same spectrum (host har) through GPU istft
    aud_h = d.run(x["asr"][0], f0, x["n"][0], x["s"][0], har=Hn, out="audio")
    snr = lambda g, r: 10*np.log10((r**2).sum()/max(((g-r)**2).sum(), 1e-20))
    print(f"F={F}: STFT mag maxerr {mag_err:.3f}, phase err p99 {np.percentile(np.abs(ph), 99):.3f} rad | "
          f"GPU iSTFT vs numpy SNR {snr(aud_h, ref):.1f} dB | full GPU DSP vs host DSP: SNR {snr(aud, ref):.1f} dB, LSD {lsd(aud, ref):.2f} dB, len {len(aud)}/{len(ref)}")
    # timing
    for mode in ("host", "gpu"):
        w = []
        for _ in range(9):
            t = time.perf_counter(); c0 = time.thread_time()
            if mode == "host":
                H = dsp.harmonic_source(f0, lw, lb, layout="tc"); m_, p_ = d.run(x["asr"][0], f0, x["n"][0], x["s"][0], har=H)
                y = dsp.istft(m_.astype(np.float64), p_.astype(np.float64))
            else:
                y = d.run(x["asr"][0], f0, x["n"][0], x["s"][0], source=(lw, lb, 1234, 1), out="audio")
            w.append(((time.perf_counter()-t)*1e3, (time.thread_time()-c0)*1e3))
        w = np.median(np.array(w), 0)
        print(f"   {mode} DSP: source+decoder+istft {w[0]:.1f} ms wall, {w[1]:.1f} ms main-thread CPU")

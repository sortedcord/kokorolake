import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, dsp
from kdec import KDecoder
t=time.perf_counter(); dec=KDecoder(); print(f"init {time.perf_counter()-t:.1f}s")
for f in sys.argv[1:] or ["in_tiny.npz","bench_inputs.npz","in_short.npz"]:
    x=np.load(f); F=x["asr"].shape[-1]
    args=(x["asr"][0], x["f0"][0], x["n"][0], x["s"][0], x["har"])
    t=time.perf_counter(); mag,ph=dec.run(*args); first=(time.perf_counter()-t)*1e3
    l=[]
    for _ in range(7):
        t=time.perf_counter(); mag,ph=dec.run(*args); l.append((time.perf_counter()-t)*1e3)
    ref=dsp.istft(x["mag"][0].astype(np.float64),x["phase"][0].astype(np.float64))
    got=dsp.istft(mag.astype(np.float64),ph.astype(np.float64))
    snr=10*np.log10((ref**2).sum()/((got-ref)**2).sum())
    lm=np.abs(np.log(mag)-np.log(x["mag"][0])).mean()
    print(f"F={F:4d} ({F*0.025:.2f}s): first {first:6.1f} ms  steady median {np.median(l):6.1f} ms  min {min(l):6.1f}  SNR {snr:5.1f} dB  mean|dlogmag| {lm:.4f}")

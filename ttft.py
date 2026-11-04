"""End-to-end time-to-first-audio and per-stage timings."""
import sys, time, numpy as np, soundfile as sf
from engine import Engine

dev = sys.argv[1] if len(sys.argv) > 1 else "GPU"
t = time.perf_counter()
e = Engine(bert_dev=dev, dec_dev=dev)
print(f"engine ready in {time.perf_counter()-t:.1f}s (compile/load {e.load_s:.1f}s)")
TEXTS = ["Done. The tests passed.",
         "Wait, what? All of the tests passed? I honestly didn't think that would work! Okay... but here's the bad news: "
         "the build broke on the main branch, and I'm not sure why. Do you want me to roll it back, or should I keep digging?"]
for text in TEXTS:
    for rep in range(3):
        t0 = time.perf_counter(); ttft = None; chunks = []; rows = []
        for audio, T in e.stream(text):
            if ttft is None: ttft = time.perf_counter() - t0
            chunks.append(audio); rows.append(T)
        total = time.perf_counter() - t0
    dur = sum(len(c) for c in chunks) / 24000
    print(f"\n{text[:50]!r}...  chunks={len(chunks)} audio={dur:.1f}s  TTFT={ttft*1e3:.0f} ms  total={total*1e3:.0f} ms (RTF {total/dur:.3f})")
    for T in rows[:3]:
        print("   " + "  ".join(f"{k}={v*1e3:.0f}ms" if isinstance(v, float) else f"{k}={v}" for k, v in T.items()))
    sf.write(f"scratch/out_{dev}_{len(text)}.wav", np.concatenate(chunks), 24000)

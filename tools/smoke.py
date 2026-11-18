"""Quick check: load the engine and time the first audio of a short passage."""
import sys, time; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kokorolake.engine import Engine
t = time.perf_counter(); e = Engine(); print(f"ready {time.perf_counter()-t:.1f}s")
t0 = time.perf_counter(); first = None; n = 0
for a, T in e.stream("Wait, what? All of the tests passed? Okay, the build is green."):
    first = first or time.perf_counter() - t0; n += len(a)
print(f"TTFA {first*1e3:.0f} ms, {n/24000:.1f}s audio")

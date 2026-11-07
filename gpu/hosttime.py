import sys, time, cProfile, pstats
sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, pyopencl as cl
from kdec import KDecoder
dec=KDecoder()
x=np.load("bench_inputs.npz"); args=(x["asr"][0], x["f0"][0], x["n"][0], x["s"][0], x["har"])
for _ in range(3): dec.run(*args)
ws=[];es=[]
for _ in range(5):
    t=time.perf_counter(); dec.run(*args); ws.append((time.perf_counter()-t)*1e3); es.append((dec.t_enq_done-t)*1e3)
print(f"wall {np.median(ws):.1f} ms, host enqueue {np.median(es):.1f} ms")
pr=cProfile.Profile(); pr.enable(); dec.run(*args); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(8)

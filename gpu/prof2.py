import sys, time, collections
sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, pyopencl as cl
from kdec import KDecoder
dec=KDecoder()
dec.q=cl.CommandQueue(dec.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
dec._sites={}; dec.reserve(128)
x=np.load(sys.argv[1] if len(sys.argv)>1 else "bench_inputs.npz"); args=(x["asr"][0], x["f0"][0], x["n"][0], x["s"][0], x["har"])
for _ in range(3): dec.run(*args)
dec.events=[]; t=time.perf_counter(); dec.run(*args); wall=(time.perf_counter()-t)*1e3
ev=dec.events; st=min(e.profile.start for _,e in ev); en=max(e.profile.end for _,e in ev)
tot=collections.Counter(); n=collections.Counter()
for k,e in ev: d=(e.profile.end-e.profile.start)*1e-6; tot[k]+=d; n[k]+=1
print(f"F={x['asr'].shape[-1]} wall {wall:.1f} ms, device span {(en-st)*1e-6:.1f} ms, kernel sum {sum(tot.values()):.1f} ms, launches {len(ev)}")
for k,v in tot.most_common(int(sys.argv[2]) if len(sys.argv)>2 else 10): print(f"  {k:34s} {v:7.2f} ms  n={n[k]}  avg {v/n[k]:.3f}")

import os; os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, time, collections; sys.path.insert(0,"."); sys.path.insert(0,"gpu")
import numpy as np, pyopencl as cl
from kdec import KDecoder
d=KDecoder(); d.q=cl.CommandQueue(d.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE); d._sites={}
x=np.load(sys.argv[1] if len(sys.argv)>1 else "bench_inputs.npz"); ids=x["ids"].astype(np.int64)
for _ in range(3): d.bert(ids)
d.events=[]; t=time.perf_counter(); d.bert(ids); wall=(time.perf_counter()-t)*1e3
ev=d.events; st=min(e.profile.start for _,e in ev); en=max(e.profile.end for _,e in ev)
tot=collections.Counter(); n=collections.Counter()
for k,e in ev: dd=(e.profile.end-e.profile.start)*1e-6; tot[k]+=dd; n[k]+=1
# host enqueue time
d.events=None
t=time.perf_counter(); d._site,d._ns=0,"bert"
print(f"L={len(ids)} wall {wall:.1f} ms, device span {(en-st)*1e-6:.1f} ms, kernel sum {sum(tot.values()):.1f} ms, launches {len(ev)}")
for k,v in tot.most_common(): print(f"  {k:34s} {v:6.2f} ms n={n[k]} avg {v/n[k]*1e3:.0f} us")

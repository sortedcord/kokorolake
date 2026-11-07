import sys, time, collections
sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, pyopencl as cl
import kdec
from kdec import KDecoder
dec=KDecoder()
dec.q=cl.CommandQueue(dec.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
evs=[]
# wrap kernels to capture events
class W:
    def __init__(s, k, name): s.k, s.name = k, name
    def __call__(s, *a, **kw):
        ev=s.k(*a, **kw); evs.append((s.name if s.name!="conv" else dec._last_conv, a[1], ev)); return ev
for mb in list(dec.conv_k): dec.conv_k[mb]=W(dec.conv_k[mb], f"conv")
for c in [x for x in vars(dec).values()]: pass
def rewrap(obj):
    for v in (obj if isinstance(obj,list) else [obj]):
        if isinstance(v, kdec.Conv): v.knl = dec.conv_k[v.MB]
        elif isinstance(v, kdec.ConvT): [rewrap(c) for _,c in v.phases]
        elif isinstance(v, dict): [rewrap(x) for x in v.values()]
        elif isinstance(v, list): [rewrap(x) for x in v]
for attr in ["encode","decode","ups","noise_convs","noise_res","resblocks","conv_post"]: rewrap(getattr(dec, attr))
dec.k_stats_part=W(dec.k_stats_part,"stats");dec.k_stats_fin=W(dec.k_stats_fin,"stats_fin"); dec.k_normact=W(dec.k_normact,"normact"); dec.k_dwup2=W(dec.k_dwup2,"dwup2"); dec.k_axpby=W(dec.k_axpby,"axpby"); dec.k_reflect=W(dec.k_reflect,"reflect")
x=np.load(sys.argv[1] if len(sys.argv)>1 else "bench_inputs.npz")
args=(x["asr"][0], x["f0"][0], x["n"][0], x["s"][0], x["har"])
for _ in range(3): dec.run(*args)
evs.clear(); dec.q.finish()
t=time.perf_counter(); dec.run(*args); wall=(time.perf_counter()-t)*1e3; enq=(dec.t_enq_done-t)*1e3
tot=collections.Counter(); n=collections.Counter(); first=None; last=None
for name,gs,ev in evs:
    d=(ev.profile.end-ev.profile.start)*1e-6; tot[name]+=d; n[name]+=1
    first=ev.profile.start if first is None else min(first,ev.profile.start); last=max(last or 0, ev.profile.end)
print(f"host enqueue {enq:.1f} ms, wall {wall:.1f} ms, device span {(last-first)*1e-6:.1f} ms, kernel sum {sum(tot.values()):.1f} ms, launches {len(evs)}")
for k,v in tot.most_common(14): print(f"  {k:34s} {v:7.2f} ms  n={n[k]}  avg {v/n[k]:.3f}")
# top conv launches
cv=sorted([((ev.profile.end-ev.profile.start)*1e-6, gs) for name,gs,ev in evs if name=="conv"], reverse=True)[:6]
print("  slowest convs (ms, global size):", [(round(a,2), b) for a,b in cv])

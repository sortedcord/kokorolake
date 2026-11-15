import os; os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys, time; sys.path.insert(0,"."); sys.path.insert(0,"gpu")
import numpy as np, pyopencl as cl
import kdec
ctx=cl.create_some_context(interactive=False); q=cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
rng=np.random.default_rng(0)
shapes=[("qkv",768,2304,1),("dense",768,768,1),("ffn",768,2048,1),("ffo",2048,768,1),("tenc",512,512,5)]
res={}
import itertools
for NT,KS,UN,SMB in [(2,8,4,1),(2,8,4,2),(2,8,2,4),(2,4,4,2),(2,4,2,4),(4,8,2,2),(4,4,2,2),(4,8,4,2),(8,8,2,2),(8,4,2,2),(2,8,4,4)]:
    try: prg=cl.Program(ctx,kdec.SRC).build(options=["-DMB=1",f"-DNT={NT}",f"-DKS={KS}",f"-DUN={UN}",f"-DSMB={SMB}","-cl-mad-enable","-cl-fast-relaxed-math"])
    except Exception as e: print("fail",NT,KS,str(e)[:200]); continue
    k=prg.skinny; row=[]
    for name,Cin,Cout,K in shapes:
      for L in (13,43):
        W=(rng.standard_normal((Cout,Cin,K))/np.sqrt(Cin*K)).astype(np.float16)
        X=(rng.standard_normal((L,Cin))*0.5).astype(np.float16)
        mf=cl.mem_flags
        Xb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=X); Wb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=kdec.pack_weights(W,Cout,Cin))
        Bb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.zeros(Cout,np.float32)); Yb=cl.Buffer(ctx,mf.READ_WRITE,L*Cout*2)
        args=(Xb,np.int32(L),np.int32(Cin),np.int32(Cin),Wb,Bb,Xb,np.int32(0),np.float32(0),Yb,np.int32(L),np.int32(Cout),np.int32(Cout),np.int32(K),np.int32(1),np.int32((K-1)//2),np.int32(0),np.float32(0),np.float32(1))
        gs=(Cout//(8*SMB)*8*KS,(L+8*NT-1)//(8*NT)); ls=(8*KS,1)
        for _ in range(3): k(q,gs,ls,*args)
        evs=[k(q,gs,ls,*args) for _ in range(20)]; q.finish()
        ts=[(ev.profile.end-ev.profile.start)*1e-3 for ev in evs[4:]]
        Y=np.empty((L,Cout),np.float16); cl.enqueue_copy(q,Y,Yb)
        ref=sum(np.pad(X.astype(np.float64),((1*(K-1)//2,(K-1)//2),(0,0)))[kk:kk+L] @ W[:,:,kk].T.astype(np.float64) for kk in range(K))
        err=np.abs(Y-ref).max()/np.abs(ref).max()
        row.append(f"{np.median(ts):5.0f}"+("" if err<1e-2 else "!"))
    print(f"NT={NT} KS={KS} UN={UN} SMB={SMB}: "+" ".join(row), flush=True)
print("cols (us):", [f"{n}/L{L}" for n,_,_,_ in shapes for L in (13,43)])

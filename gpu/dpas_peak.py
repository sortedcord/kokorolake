import numpy as np, pyopencl as cl
SRC=r"""
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void peak(__global float* out, int iters) {
    int8 a = (int8)(get_global_id(0)); int8 b = (int8)(get_local_id(0) + 1);
    float8 c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    for (int i = 0; i < iters; i++) {
        c0 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c0);
        c1 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c1);
        c2 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c2);
        c3 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c3);
        c4 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c4);
        c5 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c5);
        c6 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c6);
        c7 = intel_sub_group_f16_f16_matrix_mad_k16(a, b, c7);
    }
    float8 s = c0+c1+c2+c3+c4+c5+c6+c7;
    out[get_global_id(0)] = s.s0+s.s1+s.s2+s.s3+s.s4+s.s5+s.s6+s.s7;
}
__kernel void fma_peak(__global float* out, int iters) {
    half8 a=(half8)(get_global_id(0)*0.001h), b=(half8)(0.999h), c=0;
    half8 d=a, e=b;
    for (int i=0;i<iters;i++){ c=fma(a,b,c); d=fma(e,b,d); a=fma(c,b,a); e=fma(d,b,e);}
    out[get_global_id(0)]=(float)(c.s0+d.s1+a.s2+e.s3);
}
"""
ctx=cl.create_some_context(interactive=False); q=cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
prg=cl.Program(ctx,SRC).build()
out=cl.Buffer(ctx,cl.mem_flags.WRITE_ONLY,4*1024*1024)
for name, flop_per_iter, gs in [("dpas", 8*2*8*8*16, 112*8*8*8), ("fp16 fma", 4*8*2, 112*8*64*4)]:
    k=getattr(prg, "peak" if name=="dpas" else "fma_peak")
    iters=20000 if name=="dpas" else 20000
    for _ in range(2): k(q,(gs,),(64,),out,np.int32(100)).wait()
    ev=k(q,(gs,),(64,),out,np.int32(iters)); ev.wait()
    t=(ev.profile.end-ev.profile.start)*1e-9
    sg = gs/8 if name=="dpas" else gs
    print(f"{name}: {flop_per_iter*iters*sg/t/1e12:.1f} TFLOPS ({t*1e3:.1f} ms)")

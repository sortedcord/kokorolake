"""Prototype: 1D conv on Intel XMX (DPAS) via OpenCL, activations [T, C] fp16.

out[t, co] = bias[co] + sum_{k, ci} W[co, ci, k] * x[t*stride + k*dil - pad, ci]
"""
import time

import numpy as np
import pyopencl as cl

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// one subgroup (8 work-items) = 8 consecutive output times x (8*MB) output channels
// work-item n owns time t0+n; acc[mb][m] = out[t0+n, co0 + 8*mb + m]
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void conv_dpas(
    __global const half* X, int Tin, int Cin_pad,       // [Tin, Cin_pad]
    __global const uint* Wp,                            // packed A tiles [K][Cout/8][Cin_pad/16][8 rows][8 ints]
    __global const float* bias,                         // [Cout_pad]
    __global half* Y, int Tout, int Cout_pad,           // [Tout, Cout_pad]
    int K, int dil, int pad, int stride)
{
    const int lane = get_sub_group_local_id();
    const int sg = get_global_id(0) / 8;                 // subgroup index along time
    const int tbase = sg * (8 * NB) + lane;              // this lane's times: tbase + 8*nb
    const int co0 = get_global_id(1) * (8 * MB);
    const int nci = Cin_pad / 16;
    const int nco8 = Cout_pad / 8;
    float8 acc[NB][MB];
    for (int nb = 0; nb < NB; nb++) for (int mb = 0; mb < MB; mb++) acc[nb][mb] = (float8)(0.0f);

    for (int k = 0; k < K; k++) {
        __global const uint* wk = Wp + (size_t)k * nco8 * nci * 64 + (size_t)(co0 / 8) * nci * 64;
        __global const int* xrow[NB];
        bool ok[NB];
        for (int nb = 0; nb < NB; nb++) {
            const int t = tbase + 8 * nb;
            const int tin = t * stride + k * dil - pad;
            ok[nb] = (t < Tout) && (tin >= 0) && (tin < Tin);
            xrow[nb] = (__global const int*)(X + (size_t)(ok[nb] ? tin : 0) * Cin_pad);
        }
        for (int cb = 0; cb < nci; cb++) {
            int8 b[NB];
            for (int nb = 0; nb < NB; nb++) b[nb] = ok[nb] ? vload8(0, xrow[nb] + cb * 8) : (int8)(0);
            for (int mb = 0; mb < MB; mb++) {
                int8 a = as_int8(intel_sub_group_block_read8(wk + ((size_t)mb * nci + cb) * 64));
                for (int nb = 0; nb < NB; nb++)
                    acc[nb][mb] = intel_sub_group_f16_f16_matrix_mad_k16(a, b[nb], acc[nb][mb]);
            }
        }
    }
    for (int nb = 0; nb < NB; nb++) {
        const int t = tbase + 8 * nb;
        if (t < Tout) {
            for (int mb = 0; mb < MB; mb++) {
                const int c = co0 + 8 * mb;
                float8 bb = vload8(0, bias + c);
                vstore_half8(acc[nb][mb] + bb, 0, Y + (size_t)t * Cout_pad + c);
            }
        }
    }
}
"""


def pack_weights(W, Cout_pad, Cin_pad):
    """W [Cout, Cin, K] -> A tiles [K][Cout_pad/8][Cin_pad/16][row m][lane i] uint32,
    where uint (row m, lane i) packs halves (ci0+2i, ci0+2i+1) of output channel co0+m."""
    Cout, Cin, K = W.shape
    Wf = np.zeros((Cout_pad, Cin_pad, K), np.float16)
    Wf[:Cout, :Cin] = W
    t = Wf.transpose(2, 0, 1)  # [K, Cout_pad, Cin_pad]
    t = t.reshape(K, Cout_pad // 8, 8, Cin_pad // 16, 16)  # k, cb8, m, cb16, j
    t = t.transpose(0, 1, 3, 2, 4)  # k, cb8, cb16, m, j(16)
    t = np.ascontiguousarray(t).view(np.uint32)  # pairs of halves -> [.., m, 8]
    return t.reshape(-1)


def ref_conv(x, W, b, dil, pad, stride, Tout):
    """x [Tin, Cin] float, W [Cout, Cin, K]."""
    Tin = x.shape[0]
    Cout, Cin, K = W.shape
    out = np.tile(b[None], (Tout, 1)).astype(np.float64)
    for k in range(K):
        idx = np.arange(Tout) * stride + k * dil - pad
        ok = (idx >= 0) & (idx < Tin)
        xs = np.zeros((Tout, Cin))
        xs[ok] = x[idx[ok]]
        out += xs @ W[:, :, k].T.astype(np.float64)
    return out


def main():
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    import itertools, sys
    combos = [(4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (2, 4), (2, 8)]
    for MB, NB in combos:
        try:
            prg = cl.Program(ctx, SRC).build(options=[f"-DMB={MB}", f"-DNB={NB}", "-cl-mad-enable",
                                                      "-cl-fast-relaxed-math", "-cl-intel-256-GRF-per-thread"])
        except Exception as e:
            print("build fail", MB, NB, str(e)[:200]); continue
        knl = prg.conv_dpas
        for C, K, d, T in [(128, 11, 5, 13201), (128, 3, 1, 13201), (256, 7, 3, 2200), (1024, 3, 1, 110)]:
            rng = np.random.default_rng(0)
            x = (rng.standard_normal((T, C)) * 0.5).astype(np.float16)
            W = (rng.standard_normal((C, C, K)) * (1 / np.sqrt(C * K))).astype(np.float16)
            b = rng.standard_normal(C).astype(np.float32) * 0.1
            pad = (K - 1) // 2 * d
            mf = cl.mem_flags
            Xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
            Wb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pack_weights(W, C, C))
            Bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            Yb = cl.Buffer(ctx, mf.READ_WRITE, size=T * C * 2)
            nsg = (T + 8 * NB - 1) // (8 * NB)
            gsz = ((nsg + 7) // 8 * 64, C // (8 * MB))
            lsz = (64, 1)
            args = (Xb, np.int32(T), np.int32(C), Wb, Bb, Yb, np.int32(T), np.int32(C),
                    np.int32(K), np.int32(d), np.int32(pad), np.int32(1))
            for _ in range(3):
                knl(q, gsz, lsz, *args)
            q.finish()
            ts = []
            for _ in range(10):
                ev = knl(q, gsz, lsz, *args); ev.wait()
                ts.append((ev.profile.end - ev.profile.start) * 1e-6)
            y = np.empty((T, C), np.float16)
            cl.enqueue_copy(q, y, Yb)
            r = ref_conv(x.astype(np.float64), W.astype(np.float64), b, d, pad, 1, T)
            err = np.abs(y - r).max() / np.abs(r).max()
            flop = 2 * C * C * K * T
            print(f"MB={MB} NB={NB} C={C:4d} K={K:2d} d={d} T={T:5d}: {np.median(ts):6.3f} ms  {flop / np.median(ts) / 1e9:6.2f} TFLOPS  rel.err={err:.1e}")


if __name__ == "__main__":
    main()

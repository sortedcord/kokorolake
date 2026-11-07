"""Conv v2: time = DPAS M dimension (rows), output channels = N (lanes).

A (activations): row m = time t+m, 16 input channels; lane i holds channels (2i, 2i+1)
   -> one subgroup block read of 32 bytes per row (any dilation shift is just a row offset)
B (weights): lane n = output channel co0+n, 16 input channels packed in pairs
   -> one block_read8 of a pre-packed 256-byte tile
C: acc[m] on lane n = out[t+m, co0+n] -> stores are subgroup block writes (coalesced).
Each subgroup: TM time blocks (8 rows each) x CN channel blocks (8 channels each).
"""
import numpy as np
import pyopencl as cl

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void conv2(
    __global const half* X, int Tin, int ldx, int Cin_pad,
    __global const uint* Wp,                 // [K][Cin_pad/16][Cout_pad/8][8 pairs j][8 lanes n]
    __global const float* bias,
    __global half* Y, int Tout, int ldy, int Cout_pad,
    int K, int dil, int pad)
{
    const int lane = get_sub_group_local_id();
    const int sgt = get_global_id(0) / 8;                // subgroup index along time
    const int t0 = sgt * (8 * TM);
    const int cob0 = get_global_id(1) * CN;              // first 8-channel block
    const int nci = Cin_pad / 16, nco8 = Cout_pad / 8;
    float8 acc[TM][CN];
    for (int a = 0; a < TM; a++) for (int b = 0; b < CN; b++) acc[a][b] = (float8)(0.0f);

    for (int k = 0; k < K; k++) {
        const int base = t0 + k * dil - pad;
        for (int cb = 0; cb < nci; cb++) {
            int8 A[TM];
            for (int a = 0; a < TM; a++) {
                int r[8];
                for (int m = 0; m < 8; m++) {
                    const int t = base + a * 8 + m;
                    r[m] = (t >= 0 && t < Tin)
                        ? as_int(intel_sub_group_block_read((__global const uint*)(X + (size_t)t * ldx + cb * 16)))
                        : 0;
                }
                A[a] = (int8)(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
            __global const uint* wt = Wp + (((size_t)k * nci + cb) * nco8 + cob0) * 64;
            for (int b = 0; b < CN; b++) {
                int8 B = as_int8(intel_sub_group_block_read8(wt + b * 64));
                for (int a = 0; a < TM; a++)
                    acc[a][b] = intel_sub_group_f16_f16_matrix_mad_k16(A[a], B, acc[a][b]);
            }
        }
    }
    for (int b = 0; b < CN; b++) {
        const int c = (cob0 + b) * 8;
        const float bb = bias[c + lane];
        for (int a = 0; a < TM; a++) {
            float v[8];
            vstore8(acc[a][b] + bb, 0, v);
            for (int m = 0; m < 8; m++) {
                const int t = t0 + a * 8 + m;
                if (t < Tout)
                    intel_sub_group_block_write_us((__global ushort*)(Y + (size_t)t * ldy + c), as_ushort((half)v[m]));
            }
        }
    }
}
"""


def pack_w2(W, Cout_pad, Cin_pad):
    """W [Cout, Cin, K] -> [K][Cin_pad/16][Cout_pad/8][pair j][lane n] uint32 (halves ci0+2j, ci0+2j+1 of co0+n)."""
    Cout, Cin, K = W.shape
    Wf = np.zeros((Cout_pad, Cin_pad, K), np.float16)
    Wf[:Cout, :Cin] = W
    t = Wf.transpose(2, 1, 0)  # [K, Cin_pad, Cout_pad]
    t = t.reshape(K, Cin_pad // 16, 8, 2, Cout_pad // 8, 8)  # k, cb, j, pair, cob, n
    t = t.transpose(0, 1, 4, 2, 5, 3)  # k, cb, cob, j, n, pair
    return np.ascontiguousarray(t).view(np.uint32).reshape(-1)


def ref_conv(x, W, b, dil, pad):
    T, Cin = x.shape
    out = np.tile(b[None], (T, 1)).astype(np.float64)
    for k in range(W.shape[2]):
        idx = np.arange(T) + k * dil - pad
        ok = (idx >= 0) & (idx < T)
        xs = np.zeros((T, Cin)); xs[ok] = x[idx[ok]]
        out += xs @ W[:, :, k].T.astype(np.float64)
    return out


def main():
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    cases = [(128, 11, 5, 13201), (128, 3, 1, 13201), (256, 7, 3, 2200), (256, 3, 1, 2200), (1024, 3, 1, 110)]
    for TM, CN, grf in [(1, 4, 128), (2, 2, 128), (2, 4, 128), (4, 2, 128), (4, 4, 256), (2, 8, 256), (8, 2, 256)]:
        opts = [f"-DTM={TM}", f"-DCN={CN}", "-cl-mad-enable"] + (["-cl-intel-256-GRF-per-thread"] if grf == 256 else [])
        try:
            knl = cl.Program(ctx, SRC).build(options=opts).conv2
        except Exception as e:
            print("build fail", TM, CN, str(e)[:300]); continue
        res = []
        for C, K, d, T in cases:
            if C % (8 * CN):
                res.append("   -   "); continue
            rng = np.random.default_rng(0)
            x = (rng.standard_normal((T, C)) * 0.5).astype(np.float16)
            W = (rng.standard_normal((C, C, K)) / np.sqrt(C * K)).astype(np.float16)
            b = (rng.standard_normal(C) * 0.1).astype(np.float32)
            pad = (K - 1) // 2 * d
            mf = cl.mem_flags
            Xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
            Wb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pack_w2(W, C, C))
            Bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            Yb = cl.Buffer(ctx, mf.READ_WRITE, size=T * C * 2)
            nsg = (T + 8 * TM - 1) // (8 * TM)
            gsz = ((nsg + 7) // 8 * 64, C // (8 * CN))
            args = (Xb, np.int32(T), np.int32(C), np.int32(C), Wb, Bb, Yb, np.int32(T), np.int32(C), np.int32(C),
                    np.int32(K), np.int32(d), np.int32(pad))
            for _ in range(3):
                knl(q, gsz, (64, 1), *args)
            ts = []
            for _ in range(10):
                ev = knl(q, gsz, (64, 1), *args); ev.wait(); ts.append((ev.profile.end - ev.profile.start) * 1e-6)
            y = np.empty((T, C), np.float16); cl.enqueue_copy(q, y, Yb)
            r = ref_conv(x.astype(np.float64), W.astype(np.float64), b, d, pad)
            err = np.abs(y - r).max() / np.abs(r).max()
            tf = 2 * C * C * K * T / np.median(ts) / 1e9
            res.append(f"{np.median(ts):5.2f}ms {tf:4.1f}TF" + ("" if err < 1e-2 else f" ERR{err:.0e}"))
        print(f"TM={TM} CN={CN} grf={grf}: " + " | ".join(res), flush=True)
    print("cases:", cases)


if __name__ == "__main__":
    main()

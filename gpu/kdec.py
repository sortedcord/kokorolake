"""Kokoro decoder + iSTFTNet generator as hand-written OpenCL kernels (Intel XMX/DPAS).

Exact lengths (no padding, no masks, no per-shape compilation): every kernel takes the
sequence length at run time. Activations are [T, C] fp16 (time-major, channels padded
to 16). Convs run on the matrix engines (intel_sub_group_f16_f16_matrix_mad_k16);
instance-norm statistics are per-channel sums accumulated with atomics; norm-apply is
fused with the following activation (leaky ReLU or snake).

    mag, phase = KDecoder().run(asr[512,F], f0[2F], n[2F], s[128], har[22,120F+1])
"""
import math
from pathlib import Path

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// y[q] (row q*ostride+ooff) = ((sum_k W_k x[q*stride + k*dil - pad]) + bias + rscale*R[row/rdiv]) * oscale
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void conv_dpas(
    __global const half* X, int Tin, int ldx, int Cin_pad,
    __global const uint* Wp, __global const float* bias,
    __global const half* R, int ldr, int rdiv, float rscale,
    __global half* Y, int Tq, int ldy, int coff, int Cout_pad, int ostride, int ooff,
    int K, int dil, int pad, int stride, float oscale)
{
    const int q = get_global_id(0);
    const int co0 = get_global_id(1) * (8 * MB);
    const int nci = Cin_pad / 16;
    const int nco8 = Cout_pad / 8;
    float8 acc[MB];
    for (int mb = 0; mb < MB; mb++) acc[mb] = (float8)(0.0f);
    for (int k = 0; k < K; k++) {
        const int tin = q * stride + k * dil - pad;
        const bool ok = (q < Tq) && (tin >= 0) && (tin < Tin);
        __global const int* xrow = (__global const int*)(X + (size_t)(ok ? tin : 0) * ldx);
        __global const uint* wk = Wp + ((size_t)k * nco8 + co0 / 8) * nci * 64;
        for (int cb = 0; cb < nci; cb++) {
            int8 b = ok ? vload8(0, xrow + cb * 8) : (int8)(0);
            for (int mb = 0; mb < MB; mb++) {
                int8 a = as_int8(intel_sub_group_block_read8(wk + ((size_t)mb * nci + cb) * 64));
                acc[mb] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc[mb]);
            }
        }
    }
    if (q < Tq) {
        const int row = q * ostride + ooff;
        for (int mb = 0; mb < MB; mb++) {
            const int c = co0 + 8 * mb;
            float8 v = acc[mb] + vload8(0, bias + c);
            if (rscale != 0.0f) v += rscale * convert_float8(vload8(0, R + (size_t)(row / rdiv) * ldr + c));
            vstore_half8(v * oscale, 0, Y + (size_t)row * ldy + coff + c);
        }
    }
}

// stage 1: work-group (C/8 x RL) covers ROWS rows; lane (c8, r) strides rows by RL.
// partial sums -> P[wg][2][Cs]
__kernel void stats_part(__global const half* X, int T, int ld, __global float* P, int Cs, int rows)
{
    const int c8 = get_local_id(0), r = get_local_id(1), RL = get_local_size(1);
    const int c = c8 * 8;
    const int t0 = get_group_id(1) * rows;
    const int t1 = min(T, t0 + rows);
    float8 s = 0.0f, ss = 0.0f;
    for (int t = t0 + r; t < t1; t += RL) {
        float8 v = convert_float8(vload8(0, X + (size_t)t * ld + c));
        s += v; ss += v * v;
    }
    __local float8 ls[1024], lss[1024];   // up to C/8 * RL = 1024 lanes
    const int li = r * get_local_size(0) + c8;
    ls[li] = s; lss[li] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (r == 0) {
        for (int i = 1; i < RL; i++) { s += ls[i * get_local_size(0) + c8]; ss += lss[i * get_local_size(0) + c8]; }
        __global float* o = P + (size_t)get_group_id(1) * 2 * Cs;
        vstore8(s, 0, o + c); vstore8(ss, 0, o + Cs + c);
    }
}

// stage 2: S[2][Cs] = sum over nwg partials
__kernel void stats_fin(__global const float* P, int nwg, __global float* S, int Cs)
{
    const int i = get_global_id(0);       // 0 .. 2*Cs-1
    float a = 0.0f;
    for (int g = 0; g < nwg; g++) a += P[(size_t)g * 2 * Cs + i];
    S[i] = a;
}

// Y = act(norm(X * iscale)), norm: (x-mean)*rstd*G + B (G/B fold AdaIN style + affine)
// act: 0 none, 1 leaky(slope), 2 snake(alpha)
__kernel void normact(__global const half* X, int ldx, __global half* Y, int ldy,
    __global const float* S, int Cs, float invT, __global const float* GB, int goff, int use_norm,
    int act, float slope, __global const float* alpha, float iscale)
{
    const int c = get_global_id(0) * 8;
    __global const float* G = GB + goff;
    __global const float* B = GB + goff + Cs;
    const int t = get_global_id(1);
    float8 v = convert_float8(vload8(0, X + (size_t)t * ldx + c)) * iscale;
    if (use_norm) {
        float8 mean = vload8(0, S + c) * invT;
        float8 var = fmax(vload8(0, S + Cs + c) * invT - mean * mean, 0.0f);
        v = (v - mean) * (rsqrt(var + 1e-5f) * vload8(0, G + c)) + vload8(0, B + c);
    }
    if (act == 1) v = select(v * slope, v, v > 0.0f);
    else if (act == 2) {
        float8 a = vload8(0, alpha + c);
        float8 sn = native_sin(a * v);
        v = v + sn * sn / a;
    }
    vstore_half8(v, 0, Y + (size_t)t * ldy + c);
}

// depthwise ConvTranspose1d(k=3, s=2, p=1, output_padding=1): Y[2t]=w1 x[t]+b, Y[2t+1]=w0 x[t+1]+w2 x[t]+b
__kernel void dwup2(__global const half* X, int T, int ld, __global half* Y, __global const float* W, int Cs,
                    __global const float* Bv)
{
    const int c = get_global_id(0) * 8;
    const int t = get_global_id(1);
    float8 x = convert_float8(vload8(0, X + (size_t)t * ld + c));
    float8 xn = (t + 1 < T) ? convert_float8(vload8(0, X + (size_t)(t + 1) * ld + c)) : (float8)(0.0f);
    float8 b = vload8(0, Bv + c);
    vstore_half8(vload8(0, W + Cs + c) * x + b, 0, Y + (size_t)(2 * t) * ld + c);
    vstore_half8(vload8(0, W + c) * xn + vload8(0, W + 2 * Cs + c) * x + b, 0, Y + (size_t)(2 * t + 1) * ld + c);
}

// Y = a*Y + b*X
__kernel void axpby(__global half* Y, __global const half* X, int ld, float a, float b)
{
    const size_t i = (size_t)get_global_id(1) * ld + get_global_id(0) * 8;
    float8 y = a == 0.0f ? (float8)(0.0f) : convert_float8(vload8(0, Y + i));
    vstore_half8(a * y + b * convert_float8(vload8(0, X + i)), 0, Y + i);
}

// reflection pad (1,0) after ups[1] whose epilogue already added XS: row0 = (Y[2]-XS[2]) + XS[0]
__kernel void reflect_fix(__global half* Y, __global const half* XS, int ld)
{
    const int c = get_global_id(0) * 8;
    float8 v = convert_float8(vload8(0, Y + 2 * ld + c)) - convert_float8(vload8(0, XS + 2 * ld + c))
             + convert_float8(vload8(0, XS + c));
    vstore_half8(v, 0, Y + c);
}
"""

KS = (3, 7, 11)
MAXF = 256  # largest chunk (frames at 40 Hz) buffers are sized for
DILS = (1, 3, 5)


def rup(x, m):
    return (x + m - 1) // m * m


def pack_weights(W, Cout_pad, Cin_pad):
    """W [Cout, Cin, K] -> DPAS A tiles [K][Cout_pad/8][Cin_pad/16][row m][lane i] uint32."""
    Cout, Cin, K = W.shape
    Wf = np.zeros((Cout_pad, Cin_pad, K), np.float16)
    Wf[:Cout, :Cin] = W
    t = Wf.transpose(2, 0, 1).reshape(K, Cout_pad // 8, 8, Cin_pad // 16, 16).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(t).view(np.uint32).reshape(-1)


class Conv:
    def __init__(self, dec, W, b, dil=1, pad=None, stride=1):
        Cout, Cin, K = W.shape
        self.Cout, self.Cin, self.K, self.dil, self.stride = Cout, Cin, K, dil, stride
        self.pad = (K - 1) // 2 * dil if pad is None else pad
        self.Cin_pad, self.Cout_pad = rup(Cin, 16), rup(Cout, 8)
        n8 = self.Cout_pad // 8
        self.MB = next(m for m in (8, 4, 2, 1) if n8 % m == 0)
        self.W = dec.upload(pack_weights(W, self.Cout_pad, self.Cin_pad))
        bb = np.zeros(self.Cout_pad, np.float32)
        if b is not None:
            bb[:Cout] = b
        self.b = dec.upload(bb)
        self.knl = dec.conv_k[self.MB]
        self.name = f"conv{Cin}->{Cout} k{K} d{dil} s{stride}"

    def __call__(self, dec, X, Tin, ldx, Y, Tq, ldy, coff=0, R=None, ldr=0, rdiv=1, rscale=0.0,
                 ostride=1, ooff=0, oscale=1.0):
        assert ldx >= self.Cin_pad
        nsg = (Tq + 7) // 8
        g0 = (nsg + 7) // 8 * 64
        # tile height: largest MB that still gives enough subgroups to fill the GPU
        n8 = self.Cout_pad // 8
        MB = 1
        for m in (8, 4, 2):
            if n8 % m == 0 and nsg * (n8 // m) >= dec.min_sg:
                MB = m
                break
        dec._last_conv = self.name
        dec.launch(dec.conv_prg[MB], "conv_dpas", (g0, self.Cout_pad // (8 * MB)), (64, 1),
                 X, np.int32(Tin), np.int32(ldx), np.int32(self.Cin_pad), self.W, self.b,
                 R if R is not None else dec.dummy, np.int32(ldr), np.int32(rdiv), np.float32(rscale),
                 Y, np.int32(Tq), np.int32(ldy), np.int32(coff), np.int32(self.Cout_pad),
                 np.int32(ostride), np.int32(ooff), np.int32(self.K), np.int32(self.dil),
                 np.int32(self.pad), np.int32(self.stride), np.float32(oscale))


class ConvT:
    """ConvTranspose1d as `stride` polyphase convs writing interleaved output rows."""

    def __init__(self, dec, W, b, stride, padding):
        Cin, Cout, K = W.shape
        self.s, self.p, self.K = stride, padding, K
        self.phases = []
        for r in range(stride):
            k0 = (r + padding) % stride
            ks = list(range(k0, K, stride))
            J = len(ks)
            c_r = (r + padding - k0) // stride
            Wc = np.stack([W[:, :, ks[J - 1 - kk]].T for kk in range(J)], axis=-1)  # [Cout, Cin, J]
            self.phases.append((r, Conv(dec, Wc, b, dil=1, pad=J - 1 - c_r, stride=1)))

    def out_len(self, Lin):
        return (Lin - 1) * self.s - 2 * self.p + self.K

    def __call__(self, dec, X, Tin, ldx, Y, ldy, row_off=0, R=None, ldr=0):
        Lout = self.out_len(Tin)
        for r, conv in self.phases:
            Tq = (Lout - r + self.s - 1) // self.s
            conv(dec, X, Tin, ldx, Y, Tq, ldy, R=R, ldr=ldr, rscale=1.0 if R is not None else 0.0,
                 ostride=self.s, ooff=r + row_off)
        return Lout


class Norm:
    """AdaIN1d: per-call style terms come from fc(s); stats from the `stats` kernel."""

    def __init__(self, dec, sd, prefix, C, alpha=None):
        self.C, self.Cs = C, rup(C, 16)
        self.fcW, self.fcb = sd[prefix + ".fc.weight"], sd[prefix + ".fc.bias"]
        self.nw, self.nb = sd[prefix + ".norm.weight"], sd[prefix + ".norm.bias"]
        al = np.ones(self.Cs, np.float32)
        if alpha is not None:
            al[:C] = alpha.reshape(-1)
        self.alpha = dec.upload(al)
        self.goff = 0  # offset into the shared style buffer (set by KDecoder)


class KDecoder:
    def __init__(self, weights=HERE / "dec_weights.npz"):
        sd = {k: v for k, v in np.load(weights).items()}
        self.sd = sd
        plat = [p for p in cl.get_platforms() if any(d.type & cl.device_type.GPU for d in p.get_devices())][0]
        self.dev = [d for d in plat.get_devices() if d.type & cl.device_type.GPU][0]
        self.ctx = cl.Context([self.dev])
        self.q = cl.CommandQueue(self.ctx)
        self.conv_k = {}
        for mb in (1, 2, 4, 8):
            prg = cl.Program(self.ctx, SRC).build(options=[f"-DMB={mb}", "-cl-mad-enable", "-cl-fast-relaxed-math"])
            self.conv_k[mb] = prg.conv_dpas
        self.prg = prg
        self.k_stats_part, self.k_stats_fin = prg.stats_part, prg.stats_fin
        self.k_normact, self.k_dwup2 = prg.normact, prg.dwup2
        self.k_axpby, self.k_reflect = prg.axpby, prg.reflect_fix
        self.dummy = self.upload(np.zeros(16, np.float16))
        self._sites = {}
        self.min_sg = 1024
        self.sleep_wait = 0.0005
        self.events = None
        self._last_conv = ""
        self._site = 0
        self.conv_prg = {mb: self.conv_k[mb].program for mb in self.conv_k}
        self._build(sd)
        self.reserve(MAXF)

    # ------------------------------------------------------------------ launch with cached args
    # Each call site (by order within run()) owns a cl.Kernel whose arguments persist, so a
    # new utterance only re-sets the few length-dependent integers (pyopencl's __call__
    # costs ~110 us/launch setting all args; set_arg + enqueue is a few us).
    def launch(self, prg, kname, gsz, lsz, *args):
        i = self._site
        self._site += 1
        st = self._sites.get(i)
        if st is None or st[0] != (id(prg), kname):
            k = cl.Kernel(prg, kname)
            st = [(id(prg), kname), k, [None] * len(args)]
            self._sites[i] = st
        k, cache = st[1], st[2]
        for j, a in enumerate(args):
            c = cache[j]
            if c is a or (c is not None and type(c) is type(a) and not isinstance(a, cl.MemoryObjectHolder) and c == a):
                continue
            k.set_arg(j, a)
            cache[j] = a
        ev = cl.enqueue_nd_range_kernel(self.q, k, gsz, lsz)
        if self.events is not None:
            self.events.append((self._last_conv if kname == "conv_dpas" else kname, ev))

    def upload(self, a):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))

    # ------------------------------------------------------------------ model
    def _build(self, sd):
        C = lambda p, **kw: Conv(self, sd[p + ".weight"], sd.get(p + ".bias"), **kw)
        self.norms = []

        def N(prefix, Ch, alpha=None):
            n = Norm(self, sd, prefix, Ch, alpha)
            self.norms.append(n)
            return n

        def blk(p, cin, cout, up=False):
            return dict(n1=N(p + ".norm1", cin), n2=N(p + ".norm2", cout), c1=C(p + ".conv1"), c2=C(p + ".conv2"),
                        sc=Conv(self, sd[p + ".conv1x1.weight"], None), up=up,
                        pool=(sd[p + ".pool.weight"], sd[p + ".pool.bias"]) if up else None)

        self.encode = blk("encode", 514, 1024)
        self.decode = [blk(f"decode.{i}", 1090, 1024) for i in range(3)] + [blk("decode.3", 1090, 512, up=True)]
        pw, pb = self.decode[3]["pool"]
        Cs = rup(1090, 16)
        W3 = np.zeros((3, Cs), np.float32); W3[:, :1090] = pw[:, 0, :].T
        B3 = np.zeros(Cs, np.float32); B3[:1090] = pb
        self.pool_w, self.pool_b = self.upload(W3), self.upload(B3)

        def resblock(p, Ch, k):
            units = []
            for j, d in enumerate(DILS):
                units.append(dict(
                    n1=N(f"{p}.adain1.{j}", Ch, sd[f"{p}.alpha1.{j}"]),
                    n2=N(f"{p}.adain2.{j}", Ch, sd[f"{p}.alpha2.{j}"]),
                    c1=C(f"{p}.convs1.{j}", dil=d), c2=C(f"{p}.convs2.{j}", dil=1)))
            return units

        g = "generator"
        self.ups = [ConvT(self, sd[f"{g}.ups.0.weight"], sd[f"{g}.ups.0.bias"], 10, 5),
                    ConvT(self, sd[f"{g}.ups.1.weight"], sd[f"{g}.ups.1.bias"], 6, 3)]
        self.noise_convs = [Conv(self, sd[f"{g}.noise_convs.0.weight"], sd[f"{g}.noise_convs.0.bias"], pad=3, stride=6),
                            Conv(self, sd[f"{g}.noise_convs.1.weight"], sd[f"{g}.noise_convs.1.bias"])]
        self.noise_res = [resblock(f"{g}.noise_res.0", 256, 7), resblock(f"{g}.noise_res.1", 128, 11)]
        self.resblocks = [resblock(f"{g}.resblocks.{i}", 256 if i < 3 else 128, KS[i % 3]) for i in range(6)]
        self.conv_post = C(f"{g}.conv_post", pad=3)
        # all AdaIN fc layers stacked: one matmul per utterance -> [gamma; beta] per norm
        off = 0
        for nm in self.norms:
            nm.goff = off
            off += 2 * nm.Cs
        self.gb_len = off
        self.fcW_all = np.concatenate([nm.fcW for nm in self.norms]).astype(np.float32)
        self.fcb_all = np.concatenate([nm.fcb for nm in self.norms]).astype(np.float32)
        self.GBbuf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, off * 4)
        self.S = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 2 * 1104 * 4 * 4)  # stats scratch (reused)
        self.P = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4096 * 2 * 1104 * 4)  # stats partials
        self._bufs = {}

    # ------------------------------------------------------------------ helpers
    def buf(self, name, rows, ld):
        need = rows * ld * 2
        b = self._bufs.get(name)
        if b is None or b.size < need:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max(need, 64))
            self._bufs[name] = b
        return b

    def stats(self, X, T, ld, Cs):
        c8 = Cs // 8
        RL = max(1, min(64, 1024 // c8, 256 // c8 if c8 <= 256 else 1))
        rows = max(RL * 16, 256)
        nwg = (T + rows - 1) // rows
        self.launch(self.prg, "stats_part", (c8, nwg * RL), (c8, RL), X, np.int32(T), np.int32(ld), self.P,
                          np.int32(Cs), np.int32(rows))
        self.launch(self.prg, "stats_fin", (2 * Cs,), None, self.P, np.int32(nwg), self.S, np.int32(Cs))

    def normact(self, X, T, ldx, Y, ldy, Cs, norm=None, act=0, slope=0.0, iscale=1.0):
        self.launch(self.prg, "normact", (Cs // 8, T), None, X, np.int32(ldx), Y, np.int32(ldy), self.S, np.int32(Cs),
                       np.float32(1.0 / T), self.GBbuf, np.int32(norm.goff if norm else 0),
                       np.int32(norm is not None), np.int32(act), np.float32(slope),
                       norm.alpha if norm else self.dummy, np.float32(iscale))

    def axpby(self, Y, X, T, ld, a, b):
        self.launch(self.prg, "axpby", (ld // 8, T), None, Y, X, np.int32(ld), np.float32(a), np.float32(b))

    def adain_resblk(self, b, X, T, ldx, Y, ldy, coff, Cout):
        """AdainResBlk1d (non-upsampling): Y[:, coff:coff+Cout] = (conv2(act(n2(conv1(act(n1(X)))))) + conv1x1(X)) / sqrt2"""
        Cs_in = ldx
        t1 = self.buf("blk_t1", T, Cs_in)
        self.stats(X, T, ldx, Cs_in)
        self.normact(X, T, ldx, t1, Cs_in, Cs_in, b["n1"], act=1, slope=0.2)
        h = self.buf("blk_h", T, Cout)
        b["c1"](self, t1, T, Cs_in, h, T, Cout)
        self.stats(h, T, Cout, Cout)
        t2 = self.buf("blk_t2", T, Cout)
        self.normact(h, T, Cout, t2, Cout, Cout, b["n2"], act=1, slope=0.2)
        sc = self.buf("blk_sc", T, Cout)
        b["sc"](self, X, T, ldx, sc, T, Cout)
        b["c2"](self, t2, T, Cout, Y, T, ldy, coff=coff, R=sc, ldr=Cout, rscale=1.0, oscale=1 / math.sqrt(2))

    def ada_resblock1(self, units, X, T, C, out, shared_stats=False):
        """istftnet.AdaINResBlock1 on X [T, C] -> out [T, C] (out may not alias X)."""
        t = self.buf(f"rb_t{C}", T, C)
        h = self.buf(f"rb_h{C}", T, C)
        cur = X
        for j, u in enumerate(units):
            if j > 0 or not shared_stats:
                self.stats(cur, T, C, C)
            self.normact(cur, T, C, t, C, C, u["n1"], act=2)
            u["c1"](self, t, T, C, h, T, C)
            if j == 0:
                # keep stats of X intact for sibling branches: stats(h) goes to a second scratch
                pass
            self.stats(h, T, C, C)
            self.normact(h, T, C, t, C, C, u["n2"], act=2)
            # cur_new = conv2(t) + cur ; first unit writes into `out` (X stays intact)
            u["c2"](self, t, T, C, out, T, C, R=cur, ldr=C, rscale=1.0)
            cur = out

    # ------------------------------------------------------------------ forward
    def run(self, asr, f0, n, s, har):
        """asr [512,F], f0/n [2F], s [128] (decoder half of the style), har [22,120F+1]
        -> mag, phase [11, 120F+1] (float32, host)."""
        sd = self.sd
        self._site = 0
        F = asr.shape[1]
        assert F <= MAXF, f"{F} frames > MAXF={MAXF}"
        T80, T800, T4800 = 2 * F, 20 * F, 120 * F + 1
        h = self.fcW_all @ s.astype(np.float32) + self.fcb_all
        GB = np.zeros(self.gb_len, np.float32)
        o = 0
        for nm in self.norms:
            gamma, beta = h[o: o + nm.C], h[o + nm.C: o + 2 * nm.C]
            o += 2 * nm.C
            GB[nm.goff: nm.goff + nm.C] = (1 + gamma) * nm.nw
            GB[nm.goff + nm.Cs: nm.goff + nm.Cs + nm.C] = (1 + gamma) * nm.nb + beta
        cl.enqueue_copy(self.q, self.GBbuf, GB, is_blocking=False)
        # ---- 40 Hz inputs on host: F0/N strided convs, asr_res 1x1
        def c1d_s2(x, w, b):  # Conv1d(1,1,k3,s2,p1)
            xp = np.pad(x, (1, 1))
            return w[0, 0, 0] * xp[0:-2:2] + w[0, 0, 1] * xp[1:-1:2] + w[0, 0, 2] * xp[2::2] + b[0]
        F0 = c1d_s2(f0, sd["F0_conv.weight"], sd["F0_conv.bias"])[:F]
        Nn = c1d_s2(n, sd["N_conv.weight"], sd["N_conv.bias"])[:F]
        asr_res = sd["asr_res.0.weight"][:, :, 0] @ asr + sd["asr_res.0.bias"][:, None]  # [64,F]
        E = np.zeros((F, 528), np.float16)
        E[:, :512] = asr.T; E[:, 512] = F0; E[:, 513] = Nn
        D = np.zeros((F, 1104), np.float16)
        D[:, 1024:1088] = asr_res.T; D[:, 1088] = F0; D[:, 1089] = Nn
        Eb = self.buf("E", F, 528); cl.enqueue_copy(self.q, Eb, E, is_blocking=False)
        Db = [self.buf("D0", F, 1104), self.buf("D1", F, 1104)]
        cl.enqueue_copy(self.q, Db[0], D, is_blocking=False)
        cl.enqueue_copy(self.q, Db[1], Db[0], byte_count=D.nbytes)  # device-side duplicate
        if har.shape == (T4800, 32) and har.dtype == np.float16:
            H = har  # already in kernel layout (dsp.harmonic_source(..., layout="tc"))
        else:
            H = np.zeros((T4800, 32), np.float16); H[:, :22] = har.T
        Hb = self.buf("har", T4800, 32); cl.enqueue_copy(self.q, Hb, H, is_blocking=False)
        self._host_keep = (E, D, H)

        self.t_prep_done = __import__("time").perf_counter()
        # ---- encode + decode blocks (40 Hz)
        self.adain_resblk(self.encode, Eb, F, 528, Db[0], 1104, 0, 1024)
        cur = 0
        for b in self.decode[:3]:
            self.adain_resblk(b, Db[cur], F, 1104, Db[1 - cur], 1104, 0, 1024)
            cur = 1 - cur
        X = Db[cur]
        # ---- decode.3: upsampling block -> G0 [2F, 512]
        b = self.decode[3]
        t1 = self.buf("up_t1", F, 1104)
        self.stats(X, F, 1104, 1104)
        self.normact(X, F, 1104, t1, 1104, 1104, b["n1"], act=1, slope=0.2)
        t1u = self.buf("up_t1u", T80, 1104)
        self.launch(self.prg, "dwup2", (1104 // 8, F), None, t1, np.int32(F), np.int32(1104), t1u, self.pool_w,
                     np.int32(1104), self.pool_b)
        h = self.buf("up_h", T80, 512)
        b["c1"](self, t1u, T80, 1104, h, T80, 512)
        self.stats(h, T80, 512, 512)
        t2 = self.buf("up_t2", T80, 512)
        self.normact(h, T80, 512, t2, 512, 512, b["n2"], act=1, slope=0.2)
        sc = self.buf("up_sc", F, 512)
        b["sc"](self, X, F, 1104, sc, F, 512)  # 1x1 conv commutes with nearest upsample: do it at 40 Hz
        G0 = self.buf("G0", T80, 512)
        b["c2"](self, t2, T80, 512, G0, T80, 512, R=sc, ldr=512, rdiv=2, rscale=1.0, oscale=1 / math.sqrt(2))

        # ---- generator stage 0 (800 Hz, 256 ch)
        a = self.buf("g_act0", T80, 512)
        self.normact(G0, T80, 512, a, 512, 512, None, act=1, slope=0.1)
        nc = self.buf("nc0", T800, 256)
        self.noise_convs[0](self, Hb, T4800, 32, nc, T800, 256)
        xs = self.buf("xs0", T800, 256)
        self.stats(nc, T800, 256, 256)
        self.ada_resblock1(self.noise_res[0], nc, T800, 256, xs, shared_stats=True)
        x0 = self.buf("x0", T800, 256)
        L = self.ups[0](self, a, T80, 512, x0, 256, R=xs, ldr=256)
        assert L == T800, (L, T800)
        acc = self.avg_resblocks(self.resblocks[:3], x0, T800, 256, "0")

        # ---- generator stage 1 (4800 Hz, 128 ch)
        a1 = self.buf("g_act1", T800, 256)
        self.normact(acc, T800, 256, a1, 256, 256, None, act=1, slope=0.1, iscale=1 / 3)
        nc1 = self.buf("nc1", T4800, 128)
        self.noise_convs[1](self, Hb, T4800, 32, nc1, T4800, 128)
        xs1 = self.buf("xs1", T4800, 128)
        self.stats(nc1, T4800, 128, 128)
        self.ada_resblock1(self.noise_res[1], nc1, T4800, 128, xs1, shared_stats=True)
        x1 = self.buf("x1", T4800, 128)
        L = self.ups[1](self, a1, T800, 256, x1, 128, row_off=1, R=xs1, ldr=128)
        assert L + 1 == T4800, (L, T4800)
        self.launch(self.prg, "reflect_fix", (16,), None, x1, xs1, np.int32(128))
        acc1 = self.avg_resblocks(self.resblocks[3:], x1, T4800, 128, "1")

        # ---- conv_post
        a2 = self.buf("g_act2", T4800, 128)
        self.normact(acc1, T4800, 128, a2, 128, 128, None, act=1, slope=0.01, iscale=1 / 3)
        P = self.buf("post", T4800, 24)
        self.conv_post(self, a2, T4800, 128, P, T4800, 24)
        self.t_enq_done = __import__("time").perf_counter()
        out = np.empty((T4800, 24), np.float16)
        ev = cl.enqueue_copy(self.q, out, P, is_blocking=False)
        self.q.flush()
        if self.sleep_wait:  # don't spin a CPU core while the GPU works (shared package power)
            import time as _t
            while ev.command_execution_status != cl.command_execution_status.COMPLETE:
                _t.sleep(self.sleep_wait)
        ev.wait()
        self.t_gpu_done = __import__("time").perf_counter()
        o = out[:, :22].astype(np.float32).T
        return np.exp(o[:11]), np.sin(o[11:])

    def reserve(self, F):
        z = np.zeros
        self.run(z((512, F), np.float32), z(2 * F, np.float32), z(2 * F, np.float32), z(128, np.float32),
                 z((22, 120 * F + 1), np.float32))

    def avg_resblocks(self, rbs, x, T, C, tag):
        acc = self.buf(f"acc{tag}", T, C)
        br = self.buf(f"br{tag}", T, C)
        self.stats(x, T, C, C)
        for i, rb in enumerate(rbs):
            # stats(x) are recomputed per branch inside unit 0 only if needed; S is overwritten
            # by later stats calls, so recompute for branches 2/3.
            if i > 0:
                self.stats(x, T, C, C)
            self.ada_resblock1(rb, x, T, C, br, shared_stats=True)
            self.axpby(acc, br, T, C, 0.0 if i == 0 else 1.0, 1.0)
        return acc

"""Kokoro decoder + iSTFTNet generator as hand-written OpenCL kernels (Intel XMX/DPAS).

Exact lengths (no padding, no masks, no per-shape compilation): every kernel takes the
sequence length at run time. Activations are [T, C] fp16 (time-major, channels padded
to 16). Convs run on the matrix engines (intel_sub_group_f16_f16_matrix_mad_k16);
instance-norm statistics are per-channel sums accumulated with atomics; norm-apply is
fused with the following activation (leaky ReLU or snake).

    mag, phase = KDecoder().run(asr[512,F], f0[2F], n[2F], s[128], har[22,120F+1])
"""
import math
import os
from pathlib import Path

# Intel compute-runtime "direct submission" (ULLS): the GPU polls a ring buffer for new work
# instead of a kernel-driver submission per batch. ~15% faster decode here (OpenVINO's GPU
# plugin turns it on implicitly; plain OpenCL queues don't get it). Must precede driver init.
os.environ.setdefault("NEOReadDebugKeys", "1")
os.environ.setdefault("EnableDirectSubmission", "1")

import numpy as np
import pyopencl as cl

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights"

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
    int K, int dil, int pad, int stride, float oscale, __global float* Pst, int pbase)
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
    const bool valid = q < Tq;
    const int row = q * ostride + ooff;
    for (int mb = 0; mb < MB; mb++) {
        const int c = co0 + 8 * mb;
        float8 v = acc[mb] + vload8(0, bias + c);
        if (valid && rscale != 0.0f) v += rscale * convert_float8(vload8(0, R + (size_t)(row / rdiv) * ldr + c));
        v *= oscale;
        if (valid) vstore_half8(v, 0, Y + (size_t)row * ldy + coff + c);
#ifdef STATS
        // per-channel sum / sum of squares over this subgroup's 8 times -> one partial row
        float8 u = valid ? v : (float8)(0.0f);
        float8 u2 = u * u;
        float8 s, ss;
        s.s0 = sub_group_reduce_add(u.s0); s.s1 = sub_group_reduce_add(u.s1);
        s.s2 = sub_group_reduce_add(u.s2); s.s3 = sub_group_reduce_add(u.s3);
        s.s4 = sub_group_reduce_add(u.s4); s.s5 = sub_group_reduce_add(u.s5);
        s.s6 = sub_group_reduce_add(u.s6); s.s7 = sub_group_reduce_add(u.s7);
        ss.s0 = sub_group_reduce_add(u2.s0); ss.s1 = sub_group_reduce_add(u2.s1);
        ss.s2 = sub_group_reduce_add(u2.s2); ss.s3 = sub_group_reduce_add(u2.s3);
        ss.s4 = sub_group_reduce_add(u2.s4); ss.s5 = sub_group_reduce_add(u2.s5);
        ss.s6 = sub_group_reduce_add(u2.s6); ss.s7 = sub_group_reduce_add(u2.s7);
        if (get_sub_group_local_id() == 0) {
            __global float* pr = Pst + ((size_t)pbase + get_global_id(0) / 8) * 2 * Cout_pad;
            vstore8(s, 0, pr + c);
            vstore8(ss, 0, pr + Cout_pad + c);
        }
#endif
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

// stage 2: S[2][Cs] = sum over nwg partial rows (rows of 2*Cs floats)
// work-group of 64 lanes per 8 consecutive entries of the [2*Cs] row; lanes stride rows
__kernel void stats_fin(__global const float* P, int nwg, __global float* S, int Cs)
{
    const int i8 = get_group_id(0) * 8;   // entry 0 .. 2*Cs-1
    const int l = get_local_id(0), L = get_local_size(0);
    float8 a = 0.0f;
    for (int g = l; g < nwg; g += L) a += vload8(0, P + (size_t)g * 2 * Cs + i8);
    __local float8 sm[64];
    sm[l] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = L / 2; w > 0; w >>= 1) {
        if (l < w) sm[l] += sm[l + w];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) vstore8(sm[0], 0, S + i8);
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


// ---------------------------------------------------------------- DSP on the GPU
inline uint hash32(uint x) {  // lowbias32
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
inline float gauss(uint a, uint seed) {  // Box-Muller from two hashed uniforms
    float u1 = ((float)(hash32(a * 2u + seed) >> 8) + 0.5f) * (1.0f / 16777216.0f);
    float u2 = ((float)(hash32(a * 2u + 1u + seed * 7919u) >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return sqrt(-2.0f * log(u1)) * cos(6.283185307f * u2);
}

// SourceModuleHnNSF + SineGen (see dsp.harmonic_source): PH = cumsum(rad)*2pi at 80 Hz [T80][9]
__kernel void source(__global const float* PH, __global const float* F0, int T80, __global float* SRC, int L,
                     __global const float* linw, float linb, uint seed, int noise)
{
    const int i = get_global_id(0);
    if (i >= L) return;
    const float sp = fmax((i + 0.5f) * (1.0f / 300.0f) - 0.5f, 0.0f);
    const int i0 = min((int)sp, T80 - 1), i1 = min(i0 + 1, T80 - 1);
    const float w = sp - (float)i0;
    const float uv = F0[i / 300] > 10.0f ? 1.0f : 0.0f;
    const float namp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);
    float acc = 0.0f;
    for (int h = 0; h < 9; h++) {
        const float a = PH[i0 * 9 + h] * 300.0f, b = PH[i1 * 9 + h] * 300.0f;
        float v = sin(a * (1.0f - w) + b * w) * 0.1f * uv;
        if (noise) v += namp * gauss((uint)(i * 9 + h), seed);
        acc += v * linw[h];
    }
    SRC[i] = tanh(acc + linb);
}

// STFT n_fft=20 hop=5 hann, center (reflect) -> H[frame][0..10]=|X|, [11..21]=angle(X), rest 0
__kernel void stft(__global const float* SRC, int L, __global half* H, int nfr)
{
    const int j = get_global_id(0);
    if (j >= nfr) return;
    float xr[20];
    for (int t = 0; t < 20; t++) {
        int m = 5 * j + t - 10;
        m = m < 0 ? -m : (m >= L ? 2 * (L - 1) - m : m);
        const float win = 0.5f - 0.5f * cos(6.283185307f * t / 20.0f);
        xr[t] = SRC[m] * win;
    }
    __global half* o = H + (size_t)j * 32;
    for (int k = 0; k < 11; k++) {
        float re = 0.0f, im = 0.0f;
        for (int t = 0; t < 20; t++) {
            const float ang = 6.283185307f * (float)((k * t) % 20) / 20.0f;
            re += xr[t] * cos(ang);
            im -= xr[t] * sin(ang);
        }
        o[k] = (half)sqrt(re * re + im * im);
        o[11 + k] = (half)atan2(im, re);
    }
    for (int k = 22; k < 32; k++) o[k] = (half)0.0f;
}

// conv_post output P[frame][24] (x) -> mag=exp(x[0:11]), phase=sin(x[11:22]) -> iSTFT (center, hann) y[n]
__kernel void istft(__global const half* P, int nfr, __global float* Y, int N)
{
    const int n = get_global_id(0);
    if (n >= N) return;
    const int pn = n + 10;  // index in the padded signal
    const int j0 = max(0, (pn - 19 + 4) / 5), j1 = min(nfr - 1, pn / 5);
    float ola = 0.0f, env = 0.0f;
    for (int j = j0; j <= j1; j++) {
        const int m = pn - 5 * j;
        if (m < 0 || m >= 20) continue;
        __global const half* x = P + (size_t)j * 24;
        float v = exp((float)x[0]) * cos(sin((float)x[11]));                      // k = 0 (real part only)
        v += exp((float)x[10]) * cos(sin((float)x[21])) * ((m & 1) ? -1.0f : 1.0f); // k = 10
        for (int k = 1; k < 10; k++) {
            const float mag = exp((float)x[k]), ph = sin((float)x[11 + k]);
            v += 2.0f * mag * cos(ph + 6.283185307f * (float)((k * m) % 20) / 20.0f);
        }
        v *= 1.0f / 20.0f;
        const float win = 0.5f - 0.5f * cos(6.283185307f * m / 20.0f);
        ola += win * v;
        env += win * win;
    }
    Y[n] = env > 1e-11f ? ola / env : 0.0f;
}


// ---------------------------------------------------------------- ALBERT pieces
// attention for one (query i, head h): online softmax over all L keys (no mask: exact length)
__kernel void attn(__global const half* QKV, int L, __global half* O)
{
    const int i = get_global_id(0), h = get_global_id(1);
    if (i >= L) return;
    __global const half* qp = QKV + (size_t)i * 2304 + h * 64;
    float8 q[8], acc[8];
    for (int d = 0; d < 8; d++) { q[d] = convert_float8(vload8(d, qp)) * 0.125f; acc[d] = (float8)(0.0f); }
    float m = -INFINITY, ssum = 0.0f;
    for (int j = 0; j < L; j++) {
        __global const half* kp = QKV + (size_t)j * 2304 + 768 + h * 64;
        __global const half* vp = kp + 768;
        float8 s8 = (float8)(0.0f);
        for (int d = 0; d < 8; d++) s8 += q[d] * convert_float8(vload8(d, kp));
        const float sc = s8.s0 + s8.s1 + s8.s2 + s8.s3 + s8.s4 + s8.s5 + s8.s6 + s8.s7;
        const float mn = fmax(m, sc);
        const float a = exp(m - mn), e = exp(sc - mn);
        ssum = ssum * a + e;
        for (int d = 0; d < 8; d++) acc[d] = acc[d] * a + e * convert_float8(vload8(d, vp));
        m = mn;
    }
    __global half* op = O + (size_t)i * 768 + h * 64;
    for (int d = 0; d < 8; d++) vstore_half8(acc[d] / ssum, d, op);
}

// LayerNorm over C channels of each row (64-lane work-group per row), eps given
__kernel void layernorm(__global const half* X, __global half* Y, int C, __global const float* g,
                        __global const float* b, float eps, float leaky)
{
    const int t = get_group_id(0), l = get_local_id(0);
    __global const half* x = X + (size_t)t * C;
    float s = 0.0f, ss = 0.0f;
    for (int c = l; c < C; c += 64) { float v = x[c]; s += v; ss += v * v; }
    __local float ls[64], lss[64];
    ls[l] = s; lss[l] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = 32; w > 0; w >>= 1) { if (l < w) { ls[l] += ls[l + w]; lss[l] += lss[l + w]; } barrier(CLK_LOCAL_MEM_FENCE); }
    const float mean = ls[0] / C, var = fmax(lss[0] / C - mean * mean, 0.0f), r = rsqrt(var + eps);
    for (int c = l; c < C; c += 64) {
        float v = ((float)x[c] - mean) * r * g[c] + b[c];
        if (leaky != 0.0f && v < 0.0f) v *= leaky;
        Y[(size_t)t * C + c] = (half)v;
    }
}

// gelu_new (tanh approximation), in place
__kernel void gelu(__global half* X)
{
    const size_t i = get_global_id(0) * 8;
    float8 x = convert_float8(vload8(0, X + i));
    float8 y = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    vstore_half8(y, 0, X + i);
}


// ---------------------------------------------------------------- skinny GEMM / conv (few time rows)
// Work-group = 8 subgroups splitting the reduction (taps x 16-channel blocks); each subgroup keeps
// NT independent accumulators (one per 8-time block), so DPAS chains interleave. One work-group per
// 8 output channels x 8*NT times. Partial sums reduced through SLM; fused epilogue:
// y = act(acc + bias + rscale*R) * oscale.  act: 0 none, 1 leaky(slope), 3 gelu(tanh)
#ifndef NT
#define NT 2
#endif
#ifndef KS
#define KS 8
#endif
#ifndef UN
#define UN 4
#endif
#ifndef SMB
#define SMB 2
#endif
__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8 * KS, 1, 1)))
__kernel void skinny(
    __global const half* X, int Tin, int ldx, int Cin_pad,
    __global const uint* Wp, __global const float* bias,
    __global const half* R, int ldr, float rscale,
    __global half* Y, int Tq, int ldy, int Cout_pad,
    int K, int dil, int pad, int act, float slope, float oscale)
{
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int co0 = get_group_id(0) * 8 * SMB;
    const int t0 = get_group_id(1) * (8 * NT);
    const int nci = Cin_pad / 16, nco8 = Cout_pad / 8, total = K * nci;
    float8 acc[NT][SMB];
    for (int a = 0; a < NT; a++) for (int m = 0; m < SMB; m++) acc[a][m] = (float8)(0.0f);
    for (int base = sg; base < total; base += KS * UN) {
        int8 A[UN][SMB];
        for (int u = 0; u < UN; u++) {
            const int it = base + u * KS;
            const int k = it / nci, cb = it - k * nci;
            for (int m = 0; m < SMB; m++)
                A[u][m] = it < total ? as_int8(intel_sub_group_block_read8(
                              Wp + (((size_t)k * nco8 + co0 / 8 + m) * nci + cb) * 64)) : (int8)(0);
        }
        for (int u = 0; u < UN; u++) {
            const int it = base + u * KS;
            if (it >= total) break;
            const int k = it / nci, cb = it - k * nci;
            for (int a = 0; a < NT; a++) {
                const int t = t0 + a * 8 + lane;
                const int tin = t + k * dil - pad;
                const bool ok = (t < Tq) && (tin >= 0) && (tin < Tin);
                int8 B = ok ? vload8(0, (__global const int*)(X + (size_t)tin * ldx) + cb * 8) : (int8)(0);
                for (int m = 0; m < SMB; m++)
                    acc[a][m] = intel_sub_group_f16_f16_matrix_mad_k16(A[u][m], B, acc[a][m]);
            }
        }
    }
    __local float red[KS][NT][SMB][8][8];
    for (int a = 0; a < NT; a++) for (int m = 0; m < SMB; m++) vstore8(acc[a][m], 0, &red[sg][a][m][lane][0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int job = get_local_id(0); job < NT * SMB * 8; job += 8 * KS) {
        const int a = job / (SMB * 8), m = (job / 8) % SMB, tl = job % 8;
        const int c = co0 + 8 * m;
        float8 v = vload8(0, bias + c);
        for (int s = 0; s < KS; s++) v += vload8(0, &red[s][a][m][tl][0]);
        const int t = t0 + a * 8 + tl;
        if (t < Tq) {
            if (rscale != 0.0f) v += rscale * convert_float8(vload8(0, R + (size_t)t * ldr + c));
            if (act == 1) v = select(v * slope, v, v > 0.0f);
            else if (act == 3) v = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
            vstore_half8(v * oscale, 0, Y + (size_t)t * ldy + c);
        }
    }
}

// attention, one subgroup per (query i, head h); lanes hold 8 of the 64 dims
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn8(__global const half* QKV, int L, __global half* O)
{
    const int i = get_global_id(0) / 8, h = get_global_id(1), lane = get_sub_group_local_id();
    if (i >= L) return;
    const float8 q = convert_float8(vload8(0, QKV + (size_t)i * 2304 + h * 64 + lane * 8)) * 0.125f;
    float m = -INFINITY, ssum = 0.0f;
    float8 acc = (float8)(0.0f);
    for (int j = 0; j < L; j++) {
        __global const half* kp = QKV + (size_t)j * 2304 + 768 + h * 64 + lane * 8;
        const float8 kk = convert_float8(vload8(0, kp));
        const float8 vv = convert_float8(vload8(0, kp + 768));
        const float8 p = q * kk;
        const float sc = sub_group_reduce_add(p.s0 + p.s1 + p.s2 + p.s3 + p.s4 + p.s5 + p.s6 + p.s7);
        const float mn = fmax(m, sc), al = exp(m - mn), e = exp(sc - mn);
        ssum = ssum * al + e;
        acc = acc * al + e * vv;
        m = mn;
    }
    vstore_half8(acc / ssum, 0, O + (size_t)i * 768 + h * 64 + lane * 8);
}

// Y = a*Y + b*X
__kernel void axpby(__global half* Y, __global const half* X, int ld, float a, float b)
{
    const size_t i = (size_t)get_global_id(1) * ld + get_global_id(0) * 8;
    float8 y = a == 0.0f ? (float8)(0.0f) : convert_float8(vload8(0, Y + i));
    vstore_half8(a * y + b * convert_float8(vload8(0, X + i)), 0, Y + i);
}

// reflection pad (1,0) after ups[1] whose epilogue already added XS: row0 = (Y[2]-XS[2]) + XS[0]
__kernel void reflect_fix(__global half* Y, __global const half* XS, int ld, __global float* P, int prow)
{
    const int c = get_global_id(0) * 8;
    float8 v = convert_float8(vload8(0, Y + 2 * ld + c)) - convert_float8(vload8(0, XS + 2 * ld + c))
             + convert_float8(vload8(0, XS + c));
    vstore_half8(v, 0, Y + c);
    __global float* pr = P + (size_t)prow * 2 * ld;
    vstore8(v, 0, pr + c);
    vstore8(v * v, 0, pr + ld + c);
}
"""

KS = (3, 7, 11)

MAXF = 256  # largest chunk (frames at 40 Hz); buffers grow on demand up to this
DILS = (1, 3, 5)


class _Lazy:
    """dict-like view of an .npz that reads each tensor on access and keeps nothing."""

    def __init__(self, path):
        self.z = np.load(path)

    def __getitem__(self, k):
        return self.z[k]

    def get(self, k, default=None):
        return self.z[k] if k in self.z.files else default

    def __contains__(self, k):
        return k in self.z.files


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
                 ostride=1, ooff=0, oscale=1.0, pbase=None):
        """pbase: if set, also write per-channel partial stats rows starting at pbase
        (dec.P); returns the number of rows written."""
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
        prg = dec.conv_prg[MB] if pbase is None else dec.conv_prg_s[MB]
        dec.launch(prg, "conv_dpas", (g0, self.Cout_pad // (8 * MB)), (64, 1),
                 X, np.int32(Tin), np.int32(ldx), np.int32(self.Cin_pad), self.W, self.b,
                 R if R is not None else dec.dummy, np.int32(ldr), np.int32(rdiv), np.float32(rscale),
                 Y, np.int32(Tq), np.int32(ldy), np.int32(coff), np.int32(self.Cout_pad),
                 np.int32(ostride), np.int32(ooff), np.int32(self.K), np.int32(self.dil),
                 np.int32(self.pad), np.int32(self.stride), np.float32(oscale),
                 dec.P, np.int32(pbase or 0))
        return g0 // 8


def skinny(dec, conv, X, T, ldx, Y, ldy, R=None, ldr=0, rscale=0.0, act=0, slope=0.0, oscale=1.0):
    """conv on few time rows (T <= a few hundred): split-K work-groups (kernel `skinny`)."""
    assert conv.stride == 1 and ldx >= conv.Cin_pad and conv.Cout_pad % 16 == 0
    dec._last_conv = "skinny " + conv.name
    # defaults NT=2 (16 rows), KS=8 (split-K subgroups), UN=4, SMB=2 (16 channels): tuned on Arc 130T
    dec.launch(dec.prg, "skinny", (conv.Cout_pad // 16 * 64, (T + 15) // 16), (64, 1),
               X, np.int32(T), np.int32(ldx), np.int32(conv.Cin_pad), conv.W, conv.b,
               R if R is not None else dec.dummy, np.int32(ldr), np.float32(rscale),
               Y, np.int32(T), np.int32(ldy), np.int32(conv.Cout_pad),
               np.int32(conv.K), np.int32(conv.dil), np.int32(conv.pad), np.int32(act), np.float32(slope),
               np.float32(oscale))


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

    def __call__(self, dec, X, Tin, ldx, Y, ldy, row_off=0, R=None, ldr=0, stats=False):
        """Returns (Lout, partial stats rows written)."""
        Lout = self.out_len(Tin)
        rows = 0
        for r, conv in self.phases:
            Tq = (Lout - r + self.s - 1) // self.s
            rows += conv(dec, X, Tin, ldx, Y, Tq, ldy, R=R, ldr=ldr, rscale=1.0 if R is not None else 0.0,
                         ostride=self.s, ooff=r + row_off, pbase=rows if stats else None)
        return Lout, rows


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
    def __init__(self, weights=WEIGHTS / "decoder.npz"):
        sd = _Lazy(weights)  # tensors are read one at a time while packing (keeps the load-time peak low)
        self.sd = sd
        plat = [p for p in cl.get_platforms() if any(d.type & cl.device_type.GPU for d in p.get_devices())][0]
        self.dev = [d for d in plat.get_devices() if d.type & cl.device_type.GPU][0]
        self.ctx = cl.Context([self.dev])
        self.q = cl.CommandQueue(self.ctx)
        self.conv_k, self.conv_ks = {}, {}
        for mb in (1, 2, 4, 8):
            opts = [f"-DMB={mb}", "-cl-mad-enable", "-cl-fast-relaxed-math"]
            prg = cl.Program(self.ctx, SRC).build(options=opts)
            self.conv_k[mb] = prg.conv_dpas
            self.conv_ks[mb] = cl.Program(self.ctx, SRC).build(options=opts + ["-DSTATS"]).conv_dpas
        self.prg = prg
        # DSP kernels need exact sin/cos (the harmonic source evaluates sin at ~1e5 rad):
        # built without -cl-fast-relaxed-math
        self.prg_dsp = cl.Program(self.ctx, SRC).build(options=["-DMB=1"])
        self.k_stats_part, self.k_stats_fin = prg.stats_part, prg.stats_fin
        self.k_normact, self.k_dwup2 = prg.normact, prg.dwup2
        self.k_axpby, self.k_reflect = prg.axpby, prg.reflect_fix
        self.dummy = self.upload(np.zeros(16, np.float16))
        self._sites = {}
        self._z = {}
        self._ns = "dec"
        self.min_sg = 1024
        self.sleep_wait = 0.0005
        self.events = None
        self._last_conv = ""
        self._site = 0
        self.conv_prg = {mb: self.conv_k[mb].program for mb in self.conv_k}
        self.conv_prg_s = {mb: self.conv_ks[mb].program for mb in self.conv_ks}
        self._build(sd)
        self.load_f0n()
        self.load_bert()
        # keep only what run() needs on the host; the rest now lives on the GPU
        keep = ("F0_conv.weight", "F0_conv.bias", "N_conv.weight", "N_conv.bias", "asr_res.0.weight", "asr_res.0.bias")
        self.sd = {k: sd[k] for k in keep}
        for nm in self.norms:
            nm.fcW = nm.fcb = None
        del sd
        self.reserve(96)  # typical chunk; larger ones grow buffers once

    # ------------------------------------------------------------------ launch with cached args
    # Each call site (by order within run()) owns a cl.Kernel whose arguments persist, so a
    # new utterance only re-sets the few length-dependent integers (pyopencl's __call__
    # costs ~110 us/launch setting all args; set_arg + enqueue is a few us).
    def launch(self, prg, kname, gsz, lsz, *args):
        i = (self._ns, self._site)
        self._site += 1
        st = self._sites.get(i)
        if st is None or st[0] != (id(prg), kname):
            k = cl.Kernel(prg, kname)
            st = [(id(prg), kname), k, [None] * len(args)]
            self._sites[i] = st
        k, cache = st[1], st[2]
        for j, a in enumerate(args):
            c = cache[j]
            if c is a or (c is not None and type(c) is type(a) and not isinstance(a, cl.MemoryObjectHolder)
                          and bool(np.all(c == a))):
                continue
            k.set_arg(j, a)
            cache[j] = a
        ev = cl.enqueue_nd_range_kernel(self.q, k, gsz, lsz)
        if self.events is not None:
            self.events.append((self._last_conv if kname in ("conv_dpas", "skinny") else kname, ev))

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
        self.Sx = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 2 * 1104 * 4 * 4)  # stats of a stage input (kept)
        # stats partial rows: one row per 8 output times; largest = 120*MAXF+1 rows / 8 x 2 x 128 ch x 4 B
        rows = ((120 * MAXF + 1 + 7) // 8 + 7) // 8 * 8 + 1  # conv epilogue rows at 4800 Hz (+ reflect row)
        self.P = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max(rows * 2 * 128, 256 * 2 * 1104) * 4)
        self._bufs = {}

    # ------------------------------------------------------------------ helpers
    def zbuf(self, name, shape, dtype):
        """Zero-copy buffer: page-aligned host memory wrapped with CL_MEM_USE_HOST_PTR. The iGPU
        reads/writes it in place (shared RAM), so host<->GPU transfers are plain numpy writes/reads
        (~5-20 us) instead of enqueue_copy (~200 us each through pyopencl + the copy engine).
        Write before enqueueing the kernels that read it; read after waiting for the writers."""
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        z = self._z.get(name)
        if z is None or z[2] < nbytes:
            cap = (max(nbytes, 4096) + 4095) // 4096 * 4096
            raw = np.empty(cap + 4096, np.uint8)
            off = (-raw.ctypes.data) % 4096
            mem = raw[off:off + cap]
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=mem)
            z = (b, mem, cap, raw)
            self._z[name] = z
        return z[0], z[1][:nbytes].view(dtype).reshape(shape)

    def wait(self, timeout=15.0):
        """Wait for all enqueued work; sleep-polls on a marker so no CPU core spins.
        Raises on a device error (negative event status after a GPU hang/reset) or a stall."""
        ev = cl.enqueue_marker(self.q)
        self.q.flush()
        if self.sleep_wait:
            import time as _t
            t0 = _t.monotonic()
            # status: 3 queued, 2 submitted, 1 running, 0 complete, < 0 error
            while ev.command_execution_status > cl.command_execution_status.COMPLETE:
                if _t.monotonic() - t0 > timeout:
                    raise TimeoutError(f"GPU did not finish within {timeout:.0f} s")
                _t.sleep(self.sleep_wait)
        ev.wait()  # raises pyopencl.RuntimeError if the event ended in an error

    # Generator-stage activations share 5 physical slots (liveness worked out in run()):
    # stage 0 (800 Hz x 256 ch) and stage 1 (4800 Hz x 128 ch) each have <= 5 live buffers.
    ALIAS = {"nc0": "S1", "xs0": "S2", "x0": "S3", "rb_t256": "S4", "rb_h256": "S5", "br0": "S1", "acc0": "S2",
             "g_act1": "S2",  # normact(acc0 -> g_act1) runs in place
             "nc1": "S1", "xs1": "S3", "rb_t128": "S4", "rb_h128": "S5", "x1": "S1", "br1": "S2", "acc1": "S3",
             "g_act2": "S3"}  # normact(acc1 -> g_act2) in place

    def buf(self, name, rows, ld):
        name = self.ALIAS.get(name, name)
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
        assert nwg * 2 * Cs * 4 <= self.P.size, (T, Cs)
        self.launch(self.prg, "stats_part", (c8, nwg * RL), (c8, RL), X, np.int32(T), np.int32(ld), self.P,
                          np.int32(Cs), np.int32(rows))
        self.launch(self.prg, "stats_fin", (2 * Cs // 8 * 64,), (64,), self.P, np.int32(nwg), self.S, np.int32(Cs))

    def fin(self, nrows, Cs, S):
        """Sum `nrows` partial stats rows of dec.P into S [2][Cs]."""
        assert nrows * 2 * Cs * 4 <= self.P.size
        self.launch(self.prg, "stats_fin", (2 * Cs // 8 * 64,), (64,), self.P, np.int32(nrows), S, np.int32(Cs))

    def normact(self, X, T, ldx, Y, ldy, Cs, norm=None, act=0, slope=0.0, iscale=1.0, S=None, GB=None):
        self.launch(self.prg, "normact", (Cs // 8, T), None, X, np.int32(ldx), Y, np.int32(ldy), S or self.S, np.int32(Cs),
                       np.float32(1.0 / T), GB or self.GBbuf, np.int32(norm.goff if norm else 0),
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

    def ada_resblock1(self, units, X, T, C, out, Sx, last_stats=False):
        """istftnet.AdaINResBlock1 on X [T, C] -> out [T, C]. Sx holds stats(X) already.
        Conv epilogues produce the stats each following norm needs."""
        t = self.buf(f"rb_t{C}", T, C)
        h = self.buf(f"rb_h{C}", T, C)
        cur, Scur = X, Sx
        for j, u in enumerate(units):
            self.normact(cur, T, C, t, C, C, u["n1"], act=2, S=Scur)
            rows = u["c1"](self, t, T, C, h, T, C, pbase=0)
            self.fin(rows, C, self.S)
            self.normact(h, T, C, t, C, C, u["n2"], act=2, S=self.S)
            need = j < len(units) - 1 or last_stats
            rows = u["c2"](self, t, T, C, out, T, C, R=cur, ldr=C, rscale=1.0, pbase=0 if need else None)
            if need:
                self.fin(rows, C, self.S)
            cur, Scur = out, self.S

    # ------------------------------------------------------------------ forward
    def run(self, asr, f0, n, s, har=None, source=None, out="spec"):
        """asr [512,F], f0/n [2F], s [128] (decoder half of the style).
        har [22,120F+1] (or [120F+1,32] fp16) from the host, or source=(lin_w, lin_b, seed, noise)
        to build the harmonic source + STFT on the GPU.
        out="spec": mag, phase [11, 120F+1] (host float32); out="audio": waveform [600F] via GPU iSTFT."""
        sd = self.sd
        self._site, self._ns = 0, "dec"
        F = asr.shape[1]
        assert F <= MAXF, f"{F} frames > MAXF={MAXF}"
        T80, T800, T4800 = 2 * F, 20 * F, 120 * F + 1
        h = self.fcW_all @ s.astype(np.float32) + self.fcb_all
        self.GBbuf, GB = self.zbuf("GB", (self.gb_len,), np.float32)
        GB[:] = 0
        o = 0
        for nm in self.norms:
            gamma, beta = h[o: o + nm.C], h[o + nm.C: o + 2 * nm.C]
            o += 2 * nm.C
            GB[nm.goff: nm.goff + nm.C] = (1 + gamma) * nm.nw
            GB[nm.goff + nm.Cs: nm.goff + nm.Cs + nm.C] = (1 + gamma) * nm.nb + beta
        # ---- 40 Hz inputs on host: F0/N strided convs, asr_res 1x1
        def c1d_s2(x, w, b):  # Conv1d(1,1,k3,s2,p1)
            xp = np.pad(x, (1, 1))
            return w[0, 0, 0] * xp[0:-2:2] + w[0, 0, 1] * xp[1:-1:2] + w[0, 0, 2] * xp[2::2] + b[0]
        F0 = c1d_s2(f0, sd["F0_conv.weight"], sd["F0_conv.bias"])[:F]
        Nn = c1d_s2(n, sd["N_conv.weight"], sd["N_conv.bias"])[:F]
        asr_res = sd["asr_res.0.weight"][:, :, 0] @ asr + sd["asr_res.0.bias"][:, None]  # [64,F]
        Eb, E = self.zbuf("E", (F, 528), np.float16)
        E[:] = 0
        E[:, :512] = asr.T; E[:, 512] = F0; E[:, 513] = Nn
        Db = []
        for nm_ in ("D0", "D1"):
            b_, D = self.zbuf(nm_, (F, 1104), np.float16)
            D[:, 1090:] = 0
            D[:, 1024:1088] = asr_res.T; D[:, 1088] = F0; D[:, 1089] = Nn
            Db.append(b_)
        if har is None:
            # harmonic source + STFT on the GPU; host only builds the 80 Hz phase table
            lin_w, lin_b, seed, noise = source
            f0f = f0.astype(np.float32)
            h9 = np.arange(1, 10, dtype=np.float32)
            rad = (f0f[:, None] * h9[None] / np.float32(24000)) % np.float32(1.0)
            PH = (np.cumsum(rad, axis=0, dtype=np.float32) * np.float32(2 * np.pi)).astype(np.float32)
            PHb, PHz = self.zbuf("src_ph", PH.shape, np.float32); PHz[:] = PH
            F0b, F0z = self.zbuf("src_f0", f0f.shape, np.float32); F0z[:] = f0f
            Hb = self.buf("har", T4800, 32)
            if not hasattr(self, "_linw"):
                self._linw = self.upload(lin_w.astype(np.float32))
            L = 600 * F
            SRCb = self.buf("src", L * 2, 1)
            self.launch(self.prg_dsp, "source", ((L + 63) // 64 * 64,), (64,), PHb, F0b, np.int32(T80), SRCb,
                        np.int32(L), self._linw, np.float32(lin_b), np.uint32(seed), np.int32(noise))
            self.launch(self.prg_dsp, "stft", ((T4800 + 63) // 64 * 64,), (64,), SRCb, np.int32(L), Hb, np.int32(T4800))
            H = (PH, f0f)
        else:
            Hb, H = self.zbuf("har_in", (T4800, 32), np.float16)
            if har.shape == (T4800, 32) and har.dtype == np.float16:
                H[:] = har  # already in kernel layout (dsp.harmonic_source(..., layout="tc"))
            else:
                H[:] = 0; H[:, :22] = har.T

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
        rows = self.noise_convs[0](self, Hb, T4800, 32, nc, T800, 256, pbase=0)
        self.fin(rows, 256, self.Sx)
        xs = self.buf("xs0", T800, 256)
        self.ada_resblock1(self.noise_res[0], nc, T800, 256, xs, self.Sx)
        x0 = self.buf("x0", T800, 256)
        L, rows = self.ups[0](self, a, T80, 512, x0, 256, R=xs, ldr=256, stats=True)
        assert L == T800, (L, T800)
        self.fin(rows, 256, self.Sx)
        acc = self.avg_resblocks(self.resblocks[:3], x0, T800, 256, "0")

        # ---- generator stage 1 (4800 Hz, 128 ch)
        a1 = self.buf("g_act1", T800, 256)
        self.normact(acc, T800, 256, a1, 256, 256, None, act=1, slope=0.1, iscale=1 / 3)
        nc1 = self.buf("nc1", T4800, 128)
        rows = self.noise_convs[1](self, Hb, T4800, 32, nc1, T4800, 128, pbase=0)
        self.fin(rows, 128, self.Sx)
        xs1 = self.buf("xs1", T4800, 128)
        self.ada_resblock1(self.noise_res[1], nc1, T4800, 128, xs1, self.Sx)
        x1 = self.buf("x1", T4800, 128)
        L, rows = self.ups[1](self, a1, T800, 256, x1, 128, row_off=1, R=xs1, ldr=128, stats=True)
        assert L + 1 == T4800, (L, T4800)
        # row 0 (reflection pad) is written after the epilogue stats: add it as one more partial row
        self.launch(self.prg, "reflect_fix", (16,), None, x1, xs1, np.int32(128), self.P, np.int32(rows))
        self.fin(rows + 1, 128, self.Sx)
        acc1 = self.avg_resblocks(self.resblocks[3:], x1, T4800, 128, "1")

        # ---- conv_post
        a2 = self.buf("g_act2", T4800, 128)
        self.normact(acc1, T4800, 128, a2, 128, 128, None, act=1, slope=0.01, iscale=1 / 3)
        P = self.buf("post", T4800, 24)
        self.conv_post(self, a2, T4800, 128, P, T4800, 24)
        if out == "audio":
            N = 600 * F
            Yb, Yz = self.zbuf("audio", (N,), np.float32)
            self.launch(self.prg_dsp, "istft", ((N + 63) // 64 * 64,), (64,), P, np.int32(T4800), Yb, np.int32(N))
        self.t_enq_done = __import__("time").perf_counter()
        self.wait()
        if out == "audio":
            res = Yz.copy()
        else:
            res = np.empty((T4800, 24), np.float16)
            cl.enqueue_copy(self.q, res, P)
        self.t_gpu_done = __import__("time").perf_counter()
        if out == "audio":
            return res
        o = res[:, :22].astype(np.float32).T
        return np.exp(o[:11]), np.sin(o[11:])

    # ------------------------------------------------------------------ F0/N predictor stacks
    def load_f0n(self, path=WEIGHTS / "f0n.npz"):
        sd = _Lazy(path)
        norms = []

        def blk(p, cin, cout, up):
            n1, n2 = Norm(self, sd, p + ".norm1", cin), Norm(self, sd, p + ".norm2", cout)
            norms.extend([n1, n2])
            b = dict(n1=n1, n2=n2, up=up, c1=Conv(self, sd[p + ".conv1.weight"], sd[p + ".conv1.bias"]),
                     c2=Conv(self, sd[p + ".conv2.weight"], sd[p + ".conv2.bias"]),
                     sc=Conv(self, sd[p + ".conv1x1.weight"], None) if p + ".conv1x1.weight" in sd else None)
            if up:
                W3 = np.ascontiguousarray(sd[p + ".pool.weight"][:, 0, :].T.astype(np.float32))  # [3, cin]
                b["pw"], b["pb"] = self.upload(W3), self.upload(sd[p + ".pool.bias"].astype(np.float32))
            return b

        self.f0n_stacks = {}
        for name in ("F0", "N"):
            self.f0n_stacks[name] = ([blk(f"{name}.0", 512, 512, False), blk(f"{name}.1", 512, 256, True),
                                      blk(f"{name}.2", 256, 256, False)],
                                     Conv(self, sd[f"{name}_proj.weight"], sd[f"{name}_proj.bias"]))
        off = 0
        for nm in norms:
            nm.goff = off
            off += 2 * nm.Cs
        self.f0n_norms, self.f0n_gb_len = norms, off
        self.f0n_fcW = np.concatenate([nm.fcW for nm in norms]).astype(np.float32)
        self.f0n_fcb = np.concatenate([nm.fcb for nm in norms]).astype(np.float32)
        for nm in norms:
            nm.fcW = nm.fcb = None
        self.f0n_GB = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, off * 4)

    def resblk(self, b, X, T, ldx, Y, ldy, Cout, GB, tag):
        """AdainResBlk1d (leaky 0.2), optional x2 upsample, identity or 1x1 shortcut."""
        up = b["up"]
        T2 = 2 * T if up else T
        t1 = self.buf(f"{tag}t1", T, ldx)
        self.stats(X, T, ldx, ldx)
        self.normact(X, T, ldx, t1, ldx, ldx, b["n1"], act=1, slope=0.2, GB=GB)
        src = t1
        if up:
            src = self.buf(f"{tag}t1u", T2, ldx)
            self.launch(self.prg, "dwup2", (ldx // 8, T), None, t1, np.int32(T), np.int32(ldx), src, b["pw"],
                        np.int32(ldx), b["pb"])
        h = self.buf(f"{tag}h", T2, Cout)
        b["c1"](self, src, T2, ldx, h, T2, Cout)
        self.stats(h, T2, Cout, Cout)
        t2 = self.buf(f"{tag}t2", T2, Cout)
        self.normact(h, T2, Cout, t2, Cout, Cout, b["n2"], act=1, slope=0.2, GB=GB)
        if b["sc"] is not None:
            sc = self.buf(f"{tag}sc", T, Cout)
            b["sc"](self, X, T, ldx, sc, T, Cout)
            R, ldr = sc, Cout
        else:
            R, ldr = X, ldx
        b["c2"](self, t2, T2, Cout, Y, T2, ldy, R=R, ldr=ldr, rdiv=2 if up else 1, rscale=1.0, oscale=1 / math.sqrt(2))

    def f0n(self, x, s):
        """x: shared-LSTM output [F, 512] (time-major), s: prosody style [128] -> f0, n [2F] (host)."""
        self._site, self._ns = 0, "f0n"
        F = x.shape[0]
        h = self.f0n_fcW @ s.astype(np.float32) + self.f0n_fcb
        GB = np.zeros(self.f0n_gb_len, np.float32)
        o = 0
        for nm in self.f0n_norms:
            gamma, beta = h[o: o + nm.C], h[o + nm.C: o + 2 * nm.C]
            o += 2 * nm.C
            GB[nm.goff: nm.goff + nm.C] = (1 + gamma) * nm.nw
            GB[nm.goff + nm.Cs: nm.goff + nm.Cs + nm.C] = (1 + gamma) * nm.nb + beta
        self.f0n_GB, GBz = self.zbuf("f0n_GB", GB.shape, np.float32); GBz[:] = GB
        Xb, Xz = self.zbuf("f0n_x", (F, 512), np.float16); Xz[:] = x
        xh = None
        outs = []
        for name in ("F0", "N"):
            blocks, proj = self.f0n_stacks[name]
            a = self.buf(f"f0n_{name}a", F, 512)
            self.resblk(blocks[0], Xb, F, 512, a, 512, 512, self.f0n_GB, "f0n_b0")
            b2 = self.buf(f"f0n_{name}b", 2 * F, 256)
            self.resblk(blocks[1], a, F, 512, b2, 256, 256, self.f0n_GB, "f0n_b1")
            c = self.buf(f"f0n_{name}c", 2 * F, 256)
            self.resblk(blocks[2], b2, 2 * F, 256, c, 256, 256, self.f0n_GB, "f0n_b2")
            pr, prz = self.zbuf(f"f0n_{name}p", (2 * F, 8), np.float16)
            proj(self, c, 2 * F, 256, pr, 2 * F, 8)
            outs.append(prz)
        self.wait()
        res = outs
        self._host_keep_f0n = (GB, xh)
        return res[0][:, 0].astype(np.float32), res[1][:, 0].astype(np.float32)

    # ------------------------------------------------------------------ PL-BERT (ALBERT)
    def load_bert(self, path=WEIGHTS / "bert.npz"):
        sd = _Lazy(path)
        p = "encoder.albert_layer_groups.0.albert_layers.0."
        lin = lambda W, b: Conv(self, W[:, :, None], b)
        self.b_emb = (sd["embeddings.word_embeddings.weight"], sd["embeddings.position_embeddings.weight"],
                      sd["embeddings.token_type_embeddings.weight"][0], sd["embeddings.LayerNorm.weight"],
                      sd["embeddings.LayerNorm.bias"])
        self.b_in = lin(sd["encoder.embedding_hidden_mapping_in.weight"], sd["encoder.embedding_hidden_mapping_in.bias"])
        self.b_qkv = lin(np.concatenate([sd[p + f"attention.{n}.weight"] for n in ("query", "key", "value")]),
                         np.concatenate([sd[p + f"attention.{n}.bias"] for n in ("query", "key", "value")]))
        self.b_dense = lin(sd[p + "attention.dense.weight"], sd[p + "attention.dense.bias"])
        self.b_ffn = lin(sd[p + "ffn.weight"], sd[p + "ffn.bias"])
        self.b_ffo = lin(sd[p + "ffn_output.weight"], sd[p + "ffn_output.bias"])
        self.b_out = lin(sd["bert_encoder.weight"], sd["bert_encoder.bias"])
        up = lambda a: self.upload(a.astype(np.float32))
        self.b_ln1 = (up(sd[p + "attention.LayerNorm.weight"]), up(sd[p + "attention.LayerNorm.bias"]))
        self.b_ln2 = (up(sd[p + "full_layer_layer_norm.weight"]), up(sd[p + "full_layer_layer_norm.bias"]))

    def bert(self, ids, readback=True):
        """ids [L] -> d_en [512, L] (host float32), or the GPU buffer [L, 512] fp16 if not readback.
        12 shared ALBERT layers, exact length."""
        self._site, self._ns = 0, "bert"
        L = len(ids)
        wte, wpe, wtt, lg, lb = self.b_emb
        e = wte[ids] + wpe[:L] + wtt
        mu, var = e.mean(-1, keepdims=True), e.var(-1, keepdims=True)
        Eb, ez = self.zbuf("b_e", (L, 128), np.float16)
        ez[:] = (e - mu) / np.sqrt(var + 1e-12) * lg + lb
        e = None
        H = self.buf("b_h", L, 768)
        skinny(self, self.b_in, Eb, L, 128, H, 768)
        QKV, A, H1, Fb = self.buf("b_qkv", L, 2304), self.buf("b_a", L, 768), self.buf("b_h1", L, 768), self.buf("b_f", L, 2048)
        for _ in range(12):
            skinny(self, self.b_qkv, H, L, 768, QKV, 2304)
            self.launch(self.prg_dsp, "attn8", (L * 8, 12), (8, 1), QKV, np.int32(L), A)
            skinny(self, self.b_dense, A, L, 768, H1, 768, R=H, ldr=768, rscale=1.0)
            self.launch(self.prg_dsp, "layernorm", (L * 64,), (64,), H1, H1, np.int32(768), *self.b_ln1, np.float32(1e-12), np.float32(0))
            skinny(self, self.b_ffn, H1, L, 768, Fb, 2048, act=3)
            skinny(self, self.b_ffo, Fb, L, 2048, H, 768, R=H1, ldr=768, rscale=1.0)
            self.launch(self.prg_dsp, "layernorm", (L * 64,), (64,), H, H, np.int32(768), *self.b_ln2, np.float32(1e-12), np.float32(0))
        Ob, Oz = self.zbuf("b_o", (L, 512), np.float16)
        skinny(self, self.b_out, H, L, 768, Ob, 512)
        if not readback:
            return Ob
        self.wait()
        out = Oz
        self._keep_bert = e
        return out.astype(np.float32).T

    def reserve(self, F):
        z = np.zeros
        self.run(z((512, F), np.float32), z(2 * F, np.float32), z(2 * F, np.float32), z(128, np.float32),
                 z((22, 120 * F + 1), np.float32))

    def avg_resblocks(self, rbs, x, T, C, tag):
        """Sum of the resblocks on x (callers scale by 1/3); self.Sx = stats(x)."""
        acc = self.buf(f"acc{tag}", T, C)
        br = self.buf(f"br{tag}", T, C)
        for i, rb in enumerate(rbs):
            self.ada_resblock1(rb, x, T, C, br, self.Sx)
            self.axpby(acc, br, T, C, 0.0 if i == 0 else 1.0, 1.0)
        return acc

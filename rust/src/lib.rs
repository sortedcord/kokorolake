//! CPU kernels for kokorolake (called from Python via ctypes).
//!
//! lstm_bidir: bidirectional single-layer LSTM recurrence (PyTorch gate order i, f, g, o).
//! The input projection (x @ W_ih^T + b_ih + b_hh) is done by the caller as one GEMM;
//! this only runs the sequential h @ W_hh^T part, one thread per direction, AVX2/FMA.
use std::arch::x86_64::*;

/// exp(x) for 8 lanes: 2^(x*log2e), range-reduced, degree-6 polynomial (rel. err ~2e-7).
#[target_feature(enable = "avx2,fma")]
unsafe fn exp8(x: __m256) -> __m256 {
    let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0)), _mm256_set1_ps(-88.0));
    let t = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm256_sub_ps(t, n); // [-0.5, 0.5]
    let mut p = _mm256_set1_ps(1.5353362e-4);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3398874e-3));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181290e-3));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504110e-2));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022652e-1));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(p, _mm256_castsi256_ps(e))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid8(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn tanh8(x: __m256) -> __m256 {
    // tanh(x) = 2*sigmoid(2x) - 1
    let two = _mm256_set1_ps(2.0);
    _mm256_sub_ps(_mm256_mul_ps(two, sigmoid8(_mm256_mul_ps(two, x))), _mm256_set1_ps(1.0))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// gates[r] += dot(W[r, :], h) for r in 0..rows; W row-major fp16 [rows][hdim], hdim % 8 == 0, rows % 4 == 0.
/// fp16 weights halve the bytes streamed per step (the recurrence is L2-bandwidth bound).
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matvec_add(w: *const u16, h: *const f32, gates: *mut f32, rows: usize, hdim: usize) {
    let ld = |p: *const u16| _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i));
    let mut r = 0;
    while r < rows {
        let (w0, w1, w2, w3) = (w.add(r * hdim), w.add((r + 1) * hdim), w.add((r + 2) * hdim), w.add((r + 3) * hdim));
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut k = 0;
        while k < hdim {
            let hv = _mm256_loadu_ps(h.add(k));
            a0 = _mm256_fmadd_ps(ld(w0.add(k)), hv, a0);
            a1 = _mm256_fmadd_ps(ld(w1.add(k)), hv, a1);
            a2 = _mm256_fmadd_ps(ld(w2.add(k)), hv, a2);
            a3 = _mm256_fmadd_ps(ld(w3.add(k)), hv, a3);
            k += 8;
        }
        *gates.add(r) += hsum(a0);
        *gates.add(r + 1) += hsum(a1);
        *gates.add(r + 2) += hsum(a2);
        *gates.add(r + 3) += hsum(a3);
        r += 4;
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn run_dir(gx: *const f32, whh: *const u16, t: usize, h: usize, out: *mut f32, dir: usize) {
    let g4 = 4 * h;
    let mut hs = vec![0f32; h];
    let mut cs = vec![0f32; h];
    let mut gates = vec![0f32; g4];
    for step in 0..t {
        let tt = if dir == 0 { step } else { t - 1 - step };
        // gx layout [T][2][4H]
        let src = gx.add((tt * 2 + dir) * g4);
        std::ptr::copy_nonoverlapping(src, gates.as_mut_ptr(), g4);
        if step > 0 {
            matvec_add(whh, hs.as_ptr(), gates.as_mut_ptr(), g4, h);
        }
        let o = out.add(tt * 2 * h + dir * h);
        let gp = gates.as_ptr();
        let mut j = 0;
        while j < h {
            let i = sigmoid8(_mm256_loadu_ps(gp.add(j)));
            let f = sigmoid8(_mm256_loadu_ps(gp.add(h + j)));
            let g = tanh8(_mm256_loadu_ps(gp.add(2 * h + j)));
            let og = sigmoid8(_mm256_loadu_ps(gp.add(3 * h + j)));
            let c = _mm256_fmadd_ps(f, _mm256_loadu_ps(cs.as_ptr().add(j)), _mm256_mul_ps(i, g));
            _mm256_storeu_ps(cs.as_mut_ptr().add(j), c);
            let hv = _mm256_mul_ps(og, tanh8(c));
            _mm256_storeu_ps(hs.as_mut_ptr().add(j), hv);
            _mm256_storeu_ps(o.add(j), hv);
            j += 8;
        }
    }
}

/// gx: [T][2][4H] input gates (both directions, biases included)
/// whh: [2][4H][H] recurrent weights (forward, reverse), fp16
/// out: [T][2H] (forward half then reverse half), like PyTorch's bidirectional output
#[no_mangle]
pub unsafe extern "C" fn lstm_bidir(gx: *const f32, whh: *const u16, t: usize, h: usize, out: *mut f32) -> i32 {
    if h % 8 != 0 || !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") || !is_x86_feature_detected!("f16c") {
        return -1;
    }
    let (gx_u, whh_u, out_u) = (gx as usize, whh as usize, out as usize);
    let g4h = 4 * h * h;
    std::thread::scope(|s| {
        let bwd = s.spawn(move || run_dir(gx_u as *const f32, (whh_u as *const u16).add(g4h), t, h, out_u as *mut f32, 1));
        run_dir(gx_u as *const f32, whh_u as *const u16, t, h, out_u as *mut f32, 0);
        bwd.join().unwrap();
    });
    0
}

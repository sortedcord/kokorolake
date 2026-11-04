"""CPU signal parts of Kokoro's iSTFTNet vocoder, in numpy (no torch at runtime).

harmonic_source(): port of SourceModuleHnNSF + SineGen + TorchSTFT.transform
istft():           port of TorchSTFT.inverse (torch.istft, center=True)
"""
import numpy as np

SR = 24000
UP = 300  # samples per 80 Hz frame
N_FFT, HOP = 20, 5
HARMONICS = 9
SINE_AMP, NOISE_STD, VOICED_TH = 0.1, 0.003, 10.0
WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic hann


def _interp_linear(x: np.ndarray, out_len: int) -> np.ndarray:
    """torch F.interpolate(mode='linear', align_corners=False) along axis 0."""
    n = x.shape[0]
    scale = n / out_len
    src = np.maximum((np.arange(out_len) + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0)[:, None].astype(x.dtype)
    return x[i0] * (1 - w) + x[i1] * w


def harmonic_source(f0_80: np.ndarray, lin_w: np.ndarray, lin_b: float, rng=None, noise=True) -> np.ndarray:
    """f0_80: [2F] predicted F0 at 80 Hz. Returns har [22, 120F+1] float32."""
    # float32 throughout, like the original: its phase reaches ~1e5 rad, so fp32
    # rounding is part of the trained behaviour and fp64 would drift from it.
    f0_80 = f0_80.astype(np.float32)
    T = f0_80.shape[0]
    L = T * UP
    h = np.arange(1, HARMONICS + 1, dtype=np.float32)
    # rad per sample is constant within each 300-sample block; the 1/300 linear
    # downsample samples block centres, so it reduces to the per-frame value
    # (SineGen's random initial phase lands on sample 0 and is dropped by it).
    rad_down = (f0_80[:, None] * h[None] / np.float32(SR)) % np.float32(1.0)  # [T,9]
    phase = np.cumsum(rad_down, axis=0, dtype=np.float32) * np.float32(2 * np.pi)
    phase_up = _interp_linear(phase * np.float32(UP), L).astype(np.float32)  # [L,9]
    sines = np.sin(phase_up) * np.float32(SINE_AMP)
    f0s = np.repeat(f0_80, UP)  # nearest upsample
    uv = (f0s > VOICED_TH).astype(np.float32)[:, None]
    if noise:
        rng = rng or np.random.default_rng()
        amp = uv * NOISE_STD + (1 - uv) * SINE_AMP / 3
        nz = amp * rng.standard_normal((L, HARMONICS))
    else:
        nz = 0.0
    sw = sines * uv + nz
    src = np.tanh(sw @ lin_w.astype(np.float32) + np.float32(lin_b)).astype(np.float32)  # [L]
    return stft_mag_phase(src)


def stft_mag_phase(x: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[idx] * WINDOW, axis=1).T  # [11, frames]
    return np.concatenate([np.abs(spec), np.angle(spec)], axis=0).astype(np.float32)


def istft(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """mag/phase [11, T] -> waveform [(T-1)*HOP]."""
    T = mag.shape[1]
    frames = np.fft.irfft(mag * np.exp(1j * phase), n=N_FFT, axis=0).T * WINDOW  # [T,20]
    n = N_FFT + HOP * (T - 1)
    out = np.zeros(n, np.float64)
    env = np.zeros(n, np.float64)
    w2 = WINDOW.astype(np.float64) ** 2
    for k in range(N_FFT // HOP):  # 4 overlapping phases, vectorised per phase
        sl = frames[:, k * HOP:(k + 1) * HOP].reshape(-1)
        out[k * HOP:k * HOP + T * HOP] += sl
        env[k * HOP:k * HOP + T * HOP] += np.tile(w2[k * HOP:(k + 1) * HOP], T)
    pad = N_FFT // 2
    out, env = out[pad:n - pad], env[pad:n - pad]
    return (out / np.where(env > 1e-11, env, 1.0)).astype(np.float32)

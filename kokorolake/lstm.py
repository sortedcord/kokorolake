"""Bidirectional LSTM recurrence (rust/, AVX2 + F16C), loaded via ctypes.

The input projection of every LSTM runs as a GEMM on the iGPU (front.py); only the
sequential h @ W_hh part runs here, one thread per direction."""
import ctypes
from pathlib import Path

_SO = Path(__file__).resolve().parent.parent / "rust" / "target" / "release" / "libkokorolake_lstm.so"
if not _SO.exists():
    raise ImportError(f"{_SO} missing: build it with `cargo build --release --manifest-path rust/Cargo.toml`")
_lib = ctypes.CDLL(str(_SO))
_lib.lstm_bidir.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
_lib.lstm_bidir.restype = ctypes.c_int

# lstm_bidir(gx: f32 [T][2][4H], whh: f16 [2][4H][H], T, H, out: f32 [T][2H]) -> 0 on success
lstm_bidir = _lib.lstm_bidir

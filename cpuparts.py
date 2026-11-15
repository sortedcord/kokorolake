"""Kokoro's LSTM parts without OpenVINO/torch: numpy GEMMs + Rust recurrence (rs/, AVX2)."""
import ctypes
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_lib = ctypes.CDLL(str(HERE / "rs" / "target" / "release" / "libkokoro_rt.so"))
_lib.lstm_bidir.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
_lib.lstm_bidir.restype = ctypes.c_int


def _ln(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


class CPUParts:
    def __init__(self, path=HERE / "models" / "cpu_parts.npz"):
        self.w = {k: (v.astype(np.float16) if k.endswith(".whh") else v) for k, v in np.load(path).items()}

    def lstm(self, name, x):
        """x [T, In] float32 -> [T, 2H] (bidirectional, PyTorch semantics)."""
        w = self.w
        T = x.shape[0]
        whh = w[f"{name}.whh"]
        H = whh.shape[2]
        gx = np.ascontiguousarray(x @ w[f"{name}.wih"].T + w[f"{name}.b"], np.float32)  # [T, 8H] = [T][2][4H]
        out = np.empty((T, 2 * H), np.float32)
        rc = _lib.lstm_bidir(gx.ctypes.data, whh.ctypes.data, T, H, out.ctypes.data)
        assert rc == 0
        return out

    def prosody(self, d_en, s):
        """d_en [512, L], s [128] -> d [L, 640], dur [L]  (kparts.ProsodyPart)"""
        w = self.w
        L = d_en.shape[1]
        sx = np.broadcast_to(s, (L, 128))
        x = np.concatenate([d_en.T, sx], -1).astype(np.float32)
        for i in range(3):
            x = self.lstm(f"dur.lstm{i}", x)  # [L, 512]
            h = w[f"dur.aln{i}.w"] @ s + w[f"dur.aln{i}.b"]
            gamma, beta = h[:512], h[512:]
            x = (1 + gamma) * _ln(x) + beta
            x = np.concatenate([x, sx], -1).astype(np.float32)
        y = self.lstm("pred.lstm", x)
        z = y @ w["pred.proj.w"].T + w["pred.proj.b"]
        dur = (1 / (1 + np.exp(-z))).sum(-1)
        return x, dur

    def shared(self, en_tm):
        """en_tm [F, 640] (time-major: d[idx]) -> [F, 512] (predictor.shared LSTM)"""
        return self.lstm("pred.shared", np.ascontiguousarray(en_tm, np.float32))

    def textenc(self, ids):
        """ids [L] -> t_en [512, L] (kparts.TextEncPart)"""
        w = self.w
        x = w["tenc.emb"][ids]  # [L, 512]
        L = len(ids)
        for i in range(3):
            W = w[f"tenc.conv{i}.w"]  # [512, 512, 5]
            xp = np.pad(x, ((2, 2), (0, 0)))
            cols = np.stack([xp[k:k + L] for k in range(5)], axis=2).reshape(L, -1)  # [L, 512*5] (ci, k)
            x = cols @ W.reshape(512, -1).T + w[f"tenc.conv{i}.b"]
            x = _ln(x) * w[f"tenc.ln{i}.g"] + w[f"tenc.ln{i}.b"]
            x = np.where(x > 0, x, 0.2 * x).astype(np.float32)
        return self.lstm("tenc.lstm", x).T

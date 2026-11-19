"""Kokoro's text encoder + prosody predictor, split across iGPU and CPU:
input projections / convs as skinny XMX GEMMs on the iGPU (kdec), LSTM recurrences in Rust
(cpuparts), the rest in numpy. Replaces cpuparts.CPUParts' numpy GEMMs (and their ~45 MB of
fp32 weights) with fp16 weights on the iGPU."""
import threading

import numpy as np
import pyopencl as cl

from .gpu import Conv, skinny
from .lstm import lstm_bidir


def _ln(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


class Front:
    def __init__(self, kdec, path):
        self.k = k = kdec
        w = np.load(path)  # lazy: each tensor is read once, converted, and dropped
        self.whh = {n[:-4]: np.ascontiguousarray(w[n].astype(np.float16)) for n in w.files if n.endswith(".whh")}

        def gpu_lstm_in(names, cin):
            """stacked W_ih (main-input columns) of one or more LSTMs as one GEMM; style columns kept on host."""
            W = np.concatenate([w[f"{n}.wih"][:, :cin] for n in names])
            Ws = np.concatenate([w[f"{n}.wih"][:, cin:] for n in names]) if w[f"{names[0]}.wih"].shape[1] > cin else None
            b = np.concatenate([w[f"{n}.b"] for n in names])
            conv = Conv(k, W[:, :, None], b)
            return conv, Ws, b

        self.dur = [gpu_lstm_in([f"dur.lstm{i}"], 512) for i in range(3)]
        self.aln = [(w[f"dur.aln{i}.w"], w[f"dur.aln{i}.b"]) for i in range(3)]
        self.tail = gpu_lstm_in(["pred.lstm", "pred.shared"], 512)  # pred.lstm and shared share input x3
        self.proj = (w["pred.proj.w"], w["pred.proj.b"])
        self.emb = w["tenc.emb"].astype(np.float16)
        self.tconv = [Conv(k, w[f"tenc.conv{i}.w"], w[f"tenc.conv{i}.b"]) for i in range(3)]
        up = lambda a: k.upload(a.astype(np.float32))
        self.tln = [(up(w[f"tenc.ln{i}.g"]), up(w[f"tenc.ln{i}.b"])) for i in range(3)]
        self.tlstm = gpu_lstm_in(["tenc.lstm"], 512)
        del w

    # -------------------------------------------------------------- helpers
    def _bias(self, lstm, s):
        conv, Ws, b = lstm
        conv.b, bb = self.k.zbuf(f"bias{id(conv)}", (conv.Cout_pad,), np.float32)  # zero-copy, rewritten per call
        bb[:] = 0
        bb[: len(b)] = b if Ws is None else b + Ws @ s

    @staticmethod
    def _rec(name_whh, gx, T):
        H = name_whh.shape[2]
        out = np.empty((T, 2 * H), np.float32)
        assert lstm_bidir(np.ascontiguousarray(gx, np.float32).ctypes.data, name_whh.ctypes.data, T, H,
                          out.ctypes.data) == 0
        return out

    def _gemm_read(self, lstm, X, L, ldx, tag, s=None):
        """GEMM into a zero-copy buffer; returns (host view, marker event)."""
        k = self.k
        conv = lstm[0]
        if s is not None:
            self._bias(lstm, s)
        G, Gz = k.zbuf(tag, (L, conv.Cout_pad), np.float16)
        skinny(k, conv, X, L, ldx, G, conv.Cout_pad)
        ev = cl.enqueue_marker(k.q)
        k.q.flush()
        return Gz, ev, None

    # -------------------------------------------------------------- text encoder (async)
    def textenc_start(self, ids):
        """Enqueue the text encoder's GPU part (embedding, 3 convs + LN + leaky, LSTM input GEMM)
        ahead of BERT; returns a thread that finishes the LSTM recurrence on the CPU."""
        k = self.k
        k._site, k._ns = 0, "tenc"
        L = len(ids)
        a, x = k.zbuf("t_a", (L, 512), np.float16)
        x[:] = self.emb[ids]
        b = k.buf("t_b", L, 512)
        for i in range(3):
            skinny(k, self.tconv[i], a, L, 512, b, 512)
            k.launch(k.prg_dsp, "layernorm", (L * 64,), (64,), b, a, np.int32(512), *self.tln[i],
                     np.float32(1e-5), np.float32(0.2))
        gx, ev, _ = self._gemm_read(self.tlstm, a, L, 512, "t_g")
        res = {}

        def finish():
            try:
                ev.wait()
                res["t_en"] = self._rec(self.whh["tenc.lstm"], gx[:, :2048].astype(np.float32), L).T
            except BaseException as ex:  # surfaced by the caller after join()
                res["error"] = ex
        th = threading.Thread(target=finish)
        th.start()
        return th, res

    # -------------------------------------------------------------- prosody
    def prosody(self, d_en_buf, L, s):
        """d_en_buf: BERT output on the GPU [L, 512] fp16; s: prosody style [128].
        Returns dur [L] and the shared-LSTM input gates at token rate gx_sh [L, 2048]."""
        k = self.k
        k._site, k._ns = 0, "pros"
        X, ld = d_en_buf, 512
        h = None
        for i in range(3):
            gx, ev, keep = self._gemm_read(self.dur[i], X, L, ld, f"p_g{i}", s)
            ev.wait()
            h = self._rec(self.whh[f"dur.lstm{i}"], gx[:, :2048].astype(np.float32), L)
            aw, ab = self.aln[i]
            hh = aw @ s + ab
            X, xz = k.zbuf(f"p_x{i}", (L, 512), np.float16)
            xz[:] = (1 + hh[:512]) * _ln(h) + hh[512:]
        gx, ev, keep = self._gemm_read(self.tail, X, L, 512, "p_gt", s)
        ev.wait()
        gx = gx.astype(np.float32)
        y = self._rec(self.whh["pred.lstm"], gx[:, :2048], L)
        pw, pb = self.proj
        dur = (1 / (1 + np.exp(-(y @ pw.T + pb)))).sum(-1)
        return dur, gx[:, 2048:4096]

    def shared(self, gx_sh_L, idx):
        """shared LSTM over frames: its input gates are the token-rate gates gathered by alignment."""
        return self._rec(self.whh["pred.shared"], np.ascontiguousarray(gx_sh_L[idx]), len(idx))

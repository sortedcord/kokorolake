"""Kokoro-82M runtime for Intel Core Ultra (Arrow Lake) iGPU + CPU. No OpenVINO, no torch.

    text -> misaki G2P (CPU)                                  -> ids
         -> text encoder: numpy convs + Rust LSTM (CPU thread, overlaps BERT)
         -> PL-BERT: custom OpenCL kernels (iGPU, XMX)        -> d_en
         -> prosody LSTMs: numpy GEMM + Rust recurrence (CPU) -> d, durations
         -> alignment (numpy gather)
         -> shared LSTM (CPU) -> F0/N blocks (iGPU)           -> f0, n
         -> harmonic source, STFT, decoder, iGPU iSTFT         -> 24 kHz audio

Text is split into sentence/clause chunks; the first chunk is kept short so audio can
start as early as possible (stream()).
"""
import os

# numpy's OpenBLAS spin-waits worker threads on all cores after each small matmul; that
# burns package power the (power-capped) iGPU needs and ~8x CPU time. Must precede numpy.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ctypes
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .front import Front  # noqa: E402
from .gpu import MAXF, WEIGHTS, KDecoder  # noqa: E402

SR = 24000
HOP = 600  # audio samples per aligned frame
# max trailing pause kept after a chunk, by its final punctuation (seconds)
PAUSE = {",": 0.18, ";": 0.3, ":": 0.3, "—": 0.25, ".": 0.38, "!": 0.38, "?": 0.38}


def _trim_heap():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class Engine:
    def __init__(self, weights=WEIGHTS):
        weights = Path(weights)
        t = time.perf_counter()
        self.kdec = KDecoder()
        self.front = Front(self.kdec, weights / "front.npz")
        a = np.load(weights / "assets.npz")
        self.voice = a["voice"]  # [510,1,256]
        self.lin_w, self.lin_b = a["lin_w"], float(a["lin_b"][0])
        self.vocab = json.loads((weights / "vocab.json").read_text())
        self.rng = np.random.default_rng()
        self.noise = True  # Kokoro's harmonic-source noise (off only for bit-comparable verification)
        from .g2p import make_g2p
        self.g2p = make_g2p(british=False)  # misaki, compact lexicon + slim spaCy (same phonemes)
        self.load_s = time.perf_counter() - t
        self.warmup()
        _trim_heap()  # return load-time temporaries (npz arrays, packing buffers) to the OS

    def warmup(self):
        self.synth_phonemes("hˈɛloʊ.")
        self.synth_phonemes("hˈɛloʊ ðˈɛɹ, ðɪs ɪz ə lˈɑŋɡɚ wˈɔɹm ˈʌp sˈɛntəns.")

    # ------------------------------------------------------------ text
    def phonemize(self, text: str) -> str:
        ps, _ = self.g2p(text)
        return ps

    @staticmethod
    def split(text: str, first_max_chars=40):
        """Sentences, then clauses for long ones. The first chunk is cut at the first
        clause boundary when the opening sentence is long (lower time to first audio)."""
        sents = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+|\n+", text) if s.strip()]
        out = []
        for i, s in enumerate(sents):
            if i == 0 and len(s) > first_max_chars:
                parts = re.split(r"(?<=[,—–])\s+", s, maxsplit=1)
                out.extend(p for p in parts if p)
            else:
                out.append(s)
        return out

    # ------------------------------------------------------------ synthesis
    def synth_phonemes(self, ps: str, speed=1.0, timings=None):
        T = {} if timings is None else timings
        tk = lambda k, t0: T.__setitem__(k, T.get(k, 0) + time.perf_counter() - t0)
        ids = np.array([0, *[self.vocab[p] for p in ps if p in self.vocab], 0], np.int64)
        L = len(ids)
        ref_s = self.voice[min(max(L - 3, 0), len(self.voice) - 1)]  # [1,256]
        s_dec, s_pros = ref_s[0, :128].astype(np.float32), ref_s[0, 128:].astype(np.float32)

        t = time.perf_counter()
        # text encoder: convs + gates on the iGPU ahead of BERT; its LSTM recurrence runs on a
        # CPU thread (the Rust call releases the GIL) while the iGPU runs BERT
        th, res = self.front.textenc_start(ids)
        d_en = self.kdec.bert(ids, readback=False)  # GPU buffer [L, 512]
        tk("bert", t)

        t = time.perf_counter()
        dur, gx_sh = self.front.prosody(d_en, L, s_pros)
        pd = np.clip(np.round(dur / speed), 1, None).astype(np.int64)
        Fr = int(pd.sum())
        if Fr > MAXF:  # longer than the decoder's buffers: split at a pause
            th.join()
            a, b = self._halve(ps)
            ya, pa = self.synth_phonemes(a, speed, T)
            yb, pb = self.synth_phonemes(b, speed, T)
            return np.concatenate([ya, yb]), np.concatenate([pa, pb])
        idx = np.repeat(np.arange(L), pd)
        x = self.front.shared(gx_sh, idx)  # [F, 512]
        tk("cpu_lstm", t)
        t = time.perf_counter()
        f0, n = self.kdec.f0n(x, s_pros)
        tk("f0n", t)
        th.join()
        asr = np.ascontiguousarray(res["t_en"][:, idx])

        # harmonic source, STFT, decoder, iSTFT all on the iGPU; host gets the waveform
        t = time.perf_counter()
        seed = int(self.rng.integers(1, 2**31))
        audio = self.kdec.run(asr, f0, n, s_dec, source=(self.lin_w, self.lin_b, seed, int(self.noise)), out="audio")
        tk("decoder", t)
        T["frames"] = Fr
        return audio, pd

    @staticmethod
    def _halve(ps: str):
        """Split a phoneme string near its middle, preferring punctuation, then spaces."""
        mid = len(ps) // 2
        for chars in (",;:—.!?", " "):
            cands = [i for i, c in enumerate(ps) if c in chars and 0 < i < len(ps) - 1]
            if cands:
                i = min(cands, key=lambda i: abs(i - mid))
                return ps[: i + 1].strip(), ps[i + 1:].strip()
        return ps[:mid], ps[mid:]

    def stream(self, text: str, speed=1.0):
        """Yield (audio_chunk, timings) per text chunk. Leading silence is trimmed
        (it only delays the first audible sound); trailing silence is capped by the
        chunk's final punctuation so chunk joins sound like natural pauses."""
        chunks = self.split(text)
        for i, chunk in enumerate(chunks):
            T = {}
            t = time.perf_counter()
            ps = self.phonemize(chunk)
            T["g2p"] = time.perf_counter() - t
            if not ps.strip():
                continue
            audio, pd = self.synth_phonemes(ps, speed, T)
            lead = max(0, int(pd[0]) * HOP - int(0.02 * SR))
            tail_sil = int(pd[-1]) * HOP
            last = i == len(chunks) - 1
            keep = int(PAUSE.get(chunk.rstrip()[-1:], 0.12) * SR) if not last else int(0.1 * SR)
            end = len(audio) - max(0, tail_sil - keep)
            yield audio[lead:end], T

    def synth(self, text: str, speed=1.0):
        return np.concatenate([a for a, _ in self.stream(text, speed)])

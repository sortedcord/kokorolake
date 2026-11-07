"""Kokoro-82M inference engine on OpenVINO with per-stage device placement.

    text -> misaki G2P (CPU) -> ids
         -> BERT (GPU, token buckets)              -> d_en
         -> prosody LSTMs (CPU, dynamic)           -> d, durations
         -> alignment (numpy)
         -> F0/N predictor (CPU, dynamic)          -> f0, n
         -> text encoder (CPU, dynamic)            -> t_en -> asr
         -> harmonic source + STFT (numpy)         -> har
         -> decoder/vocoder (GPU, frame buckets)   -> mag, phase
         -> iSTFT (numpy)                          -> 24 kHz audio

No torch at runtime. Text is split into sentence/clause chunks; the first chunk is
kept short so audio can start as early as possible (stream()).
"""
import os

# numpy's OpenBLAS spin-waits worker threads on all cores after each small matmul; that
# burns package power the (power-capped) iGPU needs and ~8x CPU time. Must precede numpy.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

import dsp

HERE = Path(__file__).resolve().parent
SR = 24000
BERT_BUCKETS = (16, 32, 64, 128, 256, 512)
DEC_BUCKETS = (32, 48, 64, 128, 256)
LEAD_CAP = None  # capping the leading boundary silence before decoding changes the sound (global norms); keep exact
HOP = 600  # audio samples per aligned frame
# max trailing pause kept after a chunk, by its final punctuation (seconds)
PAUSE = {",": 0.18, ";": 0.3, ":": 0.3, "—": 0.25, ".": 0.38, "!": 0.38, "?": 0.38}


def _bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return None


class Engine:
    def __init__(self, models=HERE / "models", bert_dev="GPU", dec_dev="custom", cpu_dev="CPU",
                 dec_buckets=DEC_BUCKETS, bert_buckets=BERT_BUCKETS):
        """dec_dev: "custom" = hand-written OpenCL/XMX decoder (gpu/kdec.py, exact lengths),
        or an OpenVINO device ("GPU", "CPU") for the bucketed masked IR decoders."""
        models = Path(models)
        self.core = ov.Core()
        self.core.set_property({"CACHE_DIR": str(models / "cache")})
        hint = {"PERFORMANCE_HINT": "LATENCY"}
        comp = lambda name, dev: self.core.compile_model(self.core.read_model(models / f"{name}.xml"), dev, hint)
        t = time.perf_counter()
        self.bert = {L: comp(f"bert_L{L}", bert_dev).create_infer_request() for L in bert_buckets}
        self.custom = dec_dev == "custom"
        if self.custom:
            sys.path.insert(0, str(HERE / "gpu"))
            from kdec import MAXF, KDecoder
            self.kdec = KDecoder()
            self.dec = {}
            dec_buckets = (MAXF,)
        else:
            self.dec = {F: comp(f"decoder_F{F}", dec_dev).create_infer_request() for F in dec_buckets}
        self.pros = comp("prosody", cpu_dev).create_infer_request()
        self.f0n = comp("f0n", cpu_dev).create_infer_request()
        self.tenc = comp("textenc", cpu_dev).create_infer_request()
        self.load_s = time.perf_counter() - t
        a = np.load(models / "assets.npz")
        self.voice = a["voice"]  # [510,1,256]
        self.lin_w, self.lin_b = a["lin_w"], float(a["lin_b"][0])
        self.vocab = json.loads((models / "vocab.json").read_text())
        self.bert_buckets, self.dec_buckets = tuple(bert_buckets), tuple(dec_buckets)
        self.rng = np.random.default_rng()
        self.lead_cap = LEAD_CAP
        from misaki import en, espeak
        self.g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
        self.warmup()

    def warmup(self):
        for L, r in self.bert.items():
            r.infer([np.zeros((1, L), np.int64), np.ones((1, L), np.int64)])
        for F, r in self.dec.items():  # (OpenVINO decoders only)
            m = lambda n: np.ones((1, 1, n), np.float32)
            r.infer([np.zeros((1, 512, F), np.float32), np.zeros((1, 2 * F), np.float32), np.zeros((1, 2 * F), np.float32),
                     np.zeros((1, 128), np.float32), np.zeros((1, 22, 120 * F + 1), np.float32),
                     m(F), m(2 * F), m(20 * F), m(120 * F + 1)])
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
        ids = [0, *[self.vocab[p] for p in ps if p in self.vocab], 0]
        n_ps = len(ids) - 2
        L = len(ids)
        ref_s = self.voice[min(max(n_ps - 1, 0), len(self.voice) - 1)]  # [1,256]
        s_dec, s_pros = ref_s[:, :128].astype(np.float32), ref_s[:, 128:].astype(np.float32)

        t = time.perf_counter()
        Lb = _bucket(L, self.bert_buckets)
        idp = np.zeros((1, Lb), np.int64); idp[0, :L] = ids
        att = np.zeros((1, Lb), np.int64); att[0, :L] = 1
        d_en = self.bert[Lb].infer([idp, att])[0][:, :, :L]
        tk("bert", t)

        t = time.perf_counter()
        r = self.pros.infer([np.ascontiguousarray(d_en), s_pros])
        d, dur = r[0], r[1]
        pd = np.clip(np.round(dur[0] / speed), 1, None).astype(np.int64)
        if self.lead_cap is not None:
            pd[0] = min(pd[0], self.lead_cap)
        Fr = int(pd.sum())
        idx = np.repeat(np.arange(L), pd)
        en = np.ascontiguousarray(d[0].T[:, idx][None])  # d^T @ aln == gather
        r = self.f0n.infer([en, s_pros])
        f0, n = r[0], r[1]
        t_en = self.tenc.infer([np.array([ids], np.int64)])[0]
        asr = np.ascontiguousarray(t_en[:, :, idx])
        tk("cpu_lstm", t)

        t = time.perf_counter()
        har = dsp.harmonic_source(f0[0], self.lin_w, self.lin_b, rng=self.rng, layout="tc" if self.custom else "ct")
        tk("source", t)

        if self.custom:
            if Fr > self.dec_buckets[0]:  # longer than the decoder's buffers: split at a pause
                a, b = self._halve(ps)
                ya, pa = self.synth_phonemes(a, speed, T)
                yb, pb = self.synth_phonemes(b, speed, T)
                return np.concatenate([ya, yb]), np.concatenate([pa, pb])
            t = time.perf_counter()
            mag, ph = self.kdec.run(asr[0], f0[0], n[0], s_dec[0], har)
            tk("decoder", t)
            t = time.perf_counter()
            audio = dsp.istft(mag.astype(np.float64), ph.astype(np.float64))
            tk("istft", t)
            T["frames"], T["bucket"] = Fr, "exact"
            return audio, pd

        t = time.perf_counter()
        Fb = _bucket(Fr, self.dec_buckets)
        if Fb is None:  # too long for the largest bucket: split at a pause and recurse
            a, b = self._halve(ps)
            T.pop("frames", None)
            ya, pa = self.synth_phonemes(a, speed, T)
            yb, pb = self.synth_phonemes(b, speed, T)
            return np.concatenate([ya, yb]), np.concatenate([pa, pb])
        pad = lambda a, nn: np.pad(a.astype(np.float32), [(0, 0)] * (a.ndim - 1) + [(0, nn - a.shape[-1])])
        msk = lambda v, nn: np.pad(np.ones((1, 1, v), np.float32), [(0, 0), (0, 0), (0, nn - v)])
        out = self.dec[Fb].infer([pad(asr, Fb), pad(f0, 2 * Fb), pad(n, 2 * Fb), s_dec, pad(har[None], 120 * Fb + 1),
                                  msk(Fr, Fb), msk(2 * Fr, 2 * Fb), msk(20 * Fr, 20 * Fb), msk(120 * Fr + 1, 120 * Fb + 1)])
        Tv = 120 * Fr + 1
        mag, ph = out[0][0, :, :Tv], out[1][0, :, :Tv]
        tk("decoder", t)

        t = time.perf_counter()
        audio = dsp.istft(mag.astype(np.float64), ph.astype(np.float64))
        tk("istft", t)
        T["frames"], T["bucket"] = Fr, Fb
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

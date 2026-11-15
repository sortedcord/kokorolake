"""Lean misaki G2P: same phonemes, much less memory.

- Lexicon: the gold/silver pronunciation dicts (365k entries as Python dicts, ~59 MB) are
  replaced by CompactDict: one UTF-8 blob + numpy offsets, binary search (~12 MB), cached
  on disk so startup skips the JSON parse.
- spaCy: misaki only runs tok2vec + tagger; the parser, NER, lemmatizer and attribute ruler
  are excluded instead of merely disabled (not loaded at all).
"""
import bisect
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "models"


class _Keys:
    """Sequence view over sorted keys for bisect."""

    def __init__(self, blob, off):
        self.blob, self.off = blob, off

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return self.blob[self.off[i]:self.off[i + 1]].decode()


class CompactDict(Mapping):
    def __init__(self, kblob, koff, vblob, voff, kind):
        self._keys = _Keys(kblob, koff)
        self._vblob, self._voff, self._kind = vblob, voff, kind
        self._get = lru_cache(maxsize=4096)(self._lookup)

    @classmethod
    def build(cls, d):
        keys = sorted(d)
        enc = [k.encode() for k in keys]
        vals = [d[k] for k in keys]
        venc = [(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).encode() for v in vals]
        off = lambda xs: np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)
        return cls(b"".join(enc), off(enc), b"".join(venc), off(venc),
                   np.array([0 if isinstance(v, str) else 1 for v in vals], np.uint8))

    def arrays(self):
        k = self._keys
        return dict(kblob=np.frombuffer(k.blob, np.uint8), koff=k.off, vblob=np.frombuffer(self._vblob, np.uint8),
                    voff=self._voff, kind=self._kind)

    @classmethod
    def from_arrays(cls, a, p):
        return cls(a[p + "kblob"].tobytes(), a[p + "koff"], a[p + "vblob"].tobytes(), a[p + "voff"], a[p + "kind"])

    def _lookup(self, key):
        i = bisect.bisect_left(self._keys, key)
        if i == len(self._keys) or self._keys[i] != key:
            return KeyError
        v = self._vblob[self._voff[i]:self._voff[i + 1]].decode()
        return json.loads(v) if self._kind[i] else v

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        v = self._get(key)
        if v is KeyError:
            raise KeyError(key)
        return v

    def __contains__(self, key):
        return isinstance(key, str) and self._get(key) is not KeyError

    def get(self, key, default=None):
        if not isinstance(key, str):
            return default
        v = self._get(key)
        return default if v is KeyError else v

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return (self._keys[i] for i in range(len(self._keys)))


def _compact_lexicons(british):
    from misaki import en
    path = CACHE / f"lexicon_{'gb' if british else 'us'}.npz"
    if path.exists():
        a = np.load(path)
        return CompactDict.from_arrays(a, "g_"), CompactDict.from_arrays(a, "s_")
    lex = en.Lexicon(british)  # slow path, once: parse JSON + grow_dictionary
    g, s = CompactDict.build(lex.golds), CompactDict.build(lex.silvers)
    del lex
    np.savez(path, **{"g_" + k: v for k, v in g.arrays().items()}, **{"s_" + k: v for k, v in s.arrays().items()})
    return g, s


def make_g2p(british=False):
    import spacy
    from misaki import en, espeak

    golds, silvers = _compact_lexicons(british)

    def lexicon_init(self, british_):
        self.british = british_
        self.cap_stresses = (0.5, 2)
        self.golds, self.silvers = golds, silvers

    orig_load = spacy.load

    def slim_load(name, enable=None, **kw):
        keep = set(enable or ())
        nlp_all = ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "transformer")
        return orig_load(name, exclude=[c for c in nlp_all if c not in keep], **kw)

    orig_init = en.Lexicon.__init__
    en.Lexicon.__init__ = lexicon_init
    en.spacy.load = slim_load
    try:
        return en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british))
    finally:
        en.Lexicon.__init__ = orig_init
        en.spacy.load = orig_load

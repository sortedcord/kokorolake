"""Dump Kokoro decoder weights (weight norm folded) to gpu/dec_weights.npz, keyed by
module path. Runtime (gpu/kdec.py) needs only numpy + pyopencl."""
import sys
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kparts import bake_weight_norm  # noqa: E402

km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
sd = {k: v.detach().float().numpy() for k, v in km.decoder.state_dict().items()}
out = Path(__file__).with_name("dec_weights.npz")
np.savez(out, **sd)
print(f"{len(sd)} tensors, {sum(v.size for v in sd.values())/1e6:.1f}M params -> {out}")

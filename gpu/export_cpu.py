"""Weights for the CPU parts (LSTMs, AdaLayerNorms, duration proj, text-encoder convs) -> models/cpu_parts.npz"""
import sys
from pathlib import Path
import numpy as np, torch
from kokoro import KModel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kparts import bake_weight_norm
km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
out = {}
def lstm(name, m):
    out[f"{name}.wih"] = np.concatenate([m.weight_ih_l0.detach().numpy(), m.weight_ih_l0_reverse.detach().numpy()])  # [8H, In]
    out[f"{name}.b"] = np.concatenate([(m.bias_ih_l0 + m.bias_hh_l0).detach().numpy(),
                                      (m.bias_ih_l0_reverse + m.bias_hh_l0_reverse).detach().numpy()])  # [8H]
    out[f"{name}.whh"] = np.stack([m.weight_hh_l0.detach().numpy(), m.weight_hh_l0_reverse.detach().numpy()])  # [2,4H,H]
p = km.predictor
for i, blk in enumerate(p.text_encoder.lstms):
    if isinstance(blk, torch.nn.LSTM):
        lstm(f"dur.lstm{i//2}", blk)
    else:
        out[f"dur.aln{i//2}.w"] = blk.fc.weight.detach().numpy(); out[f"dur.aln{i//2}.b"] = blk.fc.bias.detach().numpy()
lstm("pred.lstm", p.lstm)
out["pred.proj.w"] = p.duration_proj.linear_layer.weight.detach().numpy(); out["pred.proj.b"] = p.duration_proj.linear_layer.bias.detach().numpy()
lstm("pred.shared", p.shared)
t = km.text_encoder
out["tenc.emb"] = t.embedding.weight.detach().numpy()
for i, c in enumerate(t.cnn):
    out[f"tenc.conv{i}.w"] = c[0].weight.detach().numpy(); out[f"tenc.conv{i}.b"] = c[0].bias.detach().numpy()
    out[f"tenc.ln{i}.g"] = c[1].gamma.detach().numpy(); out[f"tenc.ln{i}.b"] = c[1].beta.detach().numpy()
lstm("tenc.lstm", t.lstm)
out = {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}
np.savez(Path(__file__).resolve().parents[1] / "models" / "cpu_parts.npz", **out)
print(len(out), "tensors,", sum(v.nbytes for v in out.values()) / 2**20, "MB")

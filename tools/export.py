"""Export Kokoro-82M weights for kokorolake (needs torch + kokoro: tools/requirements.txt).

    .venv-tools/bin/python tools/export.py [--voice af_aoede]

Writes weights/: decoder.npz, f0n.npz, bert.npz, front.npz, assets.npz, vocab.json.
Weight norm is folded into plain weights. The runtime packs and converts to fp16 on load.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from kokoro import KModel, KPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kparts import bake_weight_norm  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "weights"


def f32(t):
    return np.ascontiguousarray(t.detach().float().numpy(), np.float32)


def lstm(out, name, m):
    out[f"{name}.wih"] = np.concatenate([f32(m.weight_ih_l0), f32(m.weight_ih_l0_reverse)])  # [8H, In]
    out[f"{name}.b"] = np.concatenate([f32(m.bias_ih_l0 + m.bias_hh_l0), f32(m.bias_ih_l0_reverse + m.bias_hh_l0_reverse)])
    out[f"{name}.whh"] = np.stack([f32(m.weight_hh_l0), f32(m.weight_hh_l0_reverse)])  # [2, 4H, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_aoede")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
    p = km.predictor

    # decoder + iSTFTNet generator (iGPU)
    np.savez(OUT / "decoder.npz", **{k: f32(v) for k, v in km.decoder.state_dict().items()})
    # F0/N predictor blocks (iGPU)
    np.savez(OUT / "f0n.npz", **{k: f32(v) for k, v in p.state_dict().items() if k.startswith(("F0.", "N.", "F0_proj", "N_proj"))})
    # PL-BERT (ALBERT) + projection (iGPU)
    bert = {k: f32(v) for k, v in km.bert.state_dict().items() if not k.startswith("pooler")}
    bert["bert_encoder.weight"], bert["bert_encoder.bias"] = f32(km.bert_encoder.weight), f32(km.bert_encoder.bias)
    np.savez(OUT / "bert.npz", **bert)
    # text encoder + prosody LSTMs (gates on the iGPU, recurrence on the CPU)
    front = {}
    for i, blk in enumerate(p.text_encoder.lstms):
        if isinstance(blk, torch.nn.LSTM):
            lstm(front, f"dur.lstm{i // 2}", blk)
        else:
            front[f"dur.aln{i // 2}.w"], front[f"dur.aln{i // 2}.b"] = f32(blk.fc.weight), f32(blk.fc.bias)
    lstm(front, "pred.lstm", p.lstm)
    front["pred.proj.w"], front["pred.proj.b"] = f32(p.duration_proj.linear_layer.weight), f32(p.duration_proj.linear_layer.bias)
    lstm(front, "pred.shared", p.shared)
    t = km.text_encoder
    front["tenc.emb"] = f32(t.embedding.weight)
    for i, c in enumerate(t.cnn):
        front[f"tenc.conv{i}.w"], front[f"tenc.conv{i}.b"] = f32(c[0].weight), f32(c[0].bias)
        front[f"tenc.ln{i}.g"], front[f"tenc.ln{i}.b"] = f32(c[1].gamma), f32(c[1].beta)
    lstm(front, "tenc.lstm", t.lstm)
    np.savez(OUT / "front.npz", **front)
    # voice pack, harmonic-source merge layer, phoneme vocabulary
    pack = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False).load_voice(args.voice).numpy()
    lin = km.decoder.generator.m_source.l_linear
    np.savez(OUT / "assets.npz", voice=pack, lin_w=f32(lin.weight)[0], lin_b=f32(lin.bias))
    (OUT / "vocab.json").write_text(json.dumps(km.vocab, ensure_ascii=False))
    print(f"wrote {', '.join(sorted(x.name for x in OUT.iterdir()))} (voice {args.voice})")


if __name__ == "__main__":
    main()

"""Export Kokoro parts to OpenVINO IR (fp16 weights) + runtime assets.

  models/bert_L{n}.xml      static token buckets (accelerator)
  models/decoder_F{n}.xml   static frame buckets (accelerator)
  models/prosody.xml, f0n.xml, textenc.xml   dynamic length (CPU)
  models/assets.npz         vocab, voice pack, harmonic-merge linear weights
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from kokoro import KModel, KPipeline

from kparts import BertPart, DecoderPart, DecoderPart4D, DecoderPlain, F0NPart, ProsodyPart, TextEncPart, bake_weight_norm

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="af_aoede")
ap.add_argument("--bert", default="32,64,128,256,512")
ap.add_argument("--decoder", default="32,64,128,256,512")
ap.add_argument("--norm", default="matmul")
ap.add_argument("--dim", default="3d")
ap.add_argument("--eps-trick", type=int, default=1)
ap.add_argument("--suffix", default="")
ap.add_argument("--only", default="", help="comma list: bert,decoder,cpu,assets")
args = ap.parse_args()
only = set(filter(None, args.only.split(",")))
want = lambda k: not only or k in only

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
import kparts
kparts.NORM_MODE = args.norm
kparts.EPS_TRICK = bool(args.eps_trick)
km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())


def save(model, inputs, name, shapes=None):
    m = ov.convert_model(model, example_input=inputs, input=shapes)
    ov.save_model(m, OUT / f"{name}.xml", compress_to_fp16=True)
    print("saved", name, flush=True)


if want("bert"):
    bp = BertPart(km)
    for L in map(int, args.bert.split(",")):
        ids = torch.zeros(1, L, dtype=torch.long)
        attn = torch.ones(1, L, dtype=torch.long)
        save(bp, (ids, attn), f"bert_L{L}", [[1, L], [1, L]])

if want("decoder"):
    dp = DecoderPart4D(km) if args.dim == "4d" else DecoderPart(km)
    for Fb in map(int, args.decoder.split(",")):
        ex = (torch.randn(1, 512, Fb), torch.rand(1, 2 * Fb) * 200, torch.randn(1, 2 * Fb), torch.randn(1, 128),
              torch.randn(1, 22, 120 * Fb + 1), torch.ones(1, 1, Fb), torch.ones(1, 1, 2 * Fb),
              torch.ones(1, 1, 20 * Fb), torch.ones(1, 1, 120 * Fb + 1))
        save(dp, ex, f"decoder_F{Fb}{args.suffix}", [list(t.shape) for t in ex])

if want("dynmask"):
    Fb = 64
    ex = (torch.randn(1, 512, Fb), torch.rand(1, 2 * Fb) * 200, torch.randn(1, 2 * Fb), torch.randn(1, 128),
          torch.randn(1, 22, 120 * Fb + 1), torch.ones(1, 1, Fb), torch.ones(1, 1, 2 * Fb),
          torch.ones(1, 1, 20 * Fb), torch.ones(1, 1, 120 * Fb + 1))
    dp = DecoderPart4D(km) if args.dim == "4d" else DecoderPart(km)
    save(dp, ex, "decoder_dynmask", [[1, 512, -1], [1, -1], [1, -1], [1, 128], [1, 22, -1], [1, 1, -1], [1, 1, -1], [1, 1, -1], [1, 1, -1]])

if want("dyndec"):
    F_ = 64
    ex = (torch.randn(1, 512, F_), torch.rand(1, 2 * F_) * 200, torch.randn(1, 2 * F_), torch.randn(1, 128),
          torch.randn(1, 22, 120 * F_ + 1))
    save(DecoderPlain(km), ex, "decoder_dyn", [[1, 512, -1], [1, -1], [1, -1], [1, 128], [1, 22, -1]])

if want("cpu"):
    L = 40
    save(ProsodyPart(km), (torch.randn(1, 512, L), torch.randn(1, 128)), "prosody", [[1, 512, -1], [1, 128]])
    save(F0NPart(km), (torch.randn(1, 640, 100), torch.randn(1, 128)), "f0n", [[1, 640, -1], [1, 128]])
    save(TextEncPart(km), (torch.zeros(1, L, dtype=torch.long),), "textenc", [[1, -1]])

if want("assets"):
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    pack = pipe.load_voice(args.voice).numpy()  # [510,1,256], indexed by phoneme count - 1
    lin = km.decoder.generator.m_source.l_linear
    np.savez(OUT / "assets.npz", voice=pack, lin_w=lin.weight[0].numpy(), lin_b=lin.bias.numpy())
    (OUT / "vocab.json").write_text(json.dumps(km.vocab, ensure_ascii=False))
    print("saved assets", flush=True)

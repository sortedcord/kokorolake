"""PL-BERT (ALBERT) + bert_encoder weights -> gpu/bert_weights.npz."""
from pathlib import Path
import numpy as np, torch
from kokoro import KModel
km = KModel(repo_id="hexgrad/Kokoro-82M").eval()
sd = {k: v.detach().float().numpy() for k, v in km.bert.state_dict().items() if not k.startswith("pooler")}
sd["bert_encoder.weight"] = km.bert_encoder.weight.detach().float().numpy()
sd["bert_encoder.bias"] = km.bert_encoder.bias.detach().float().numpy()
np.savez(Path(__file__).with_name("bert_weights.npz"), **sd)
print(len(sd), "tensors")

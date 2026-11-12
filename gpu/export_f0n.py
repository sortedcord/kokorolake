"""F0/N predictor split: shared LSTM -> OpenVINO CPU model (f0n_lstm, outputs [1,F,512] time-major);
F0/N AdainResBlk1d stacks + projections -> gpu/f0n_weights.npz for the custom GPU kernels."""
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from kokoro import KModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kparts import bake_weight_norm  # noqa: E402

torch.set_grad_enabled(False)
km = bake_weight_norm(KModel(repo_id="hexgrad/Kokoro-82M").eval())
p = km.predictor


class SharedLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = p.shared

    def forward(self, en):  # en [1,640,F] -> [1,F,512]
        x, _ = self.lstm(en.transpose(-1, -2))
        return x


m = ov.convert_model(SharedLSTM(), example_input=(torch.randn(1, 640, 50),), input=[[1, 640, -1]])
ov.save_model(m, Path(__file__).resolve().parents[1] / "models" / "f0n_lstm.xml", compress_to_fp16=True)
sd = {k: v.detach().float().numpy() for k, v in p.state_dict().items()
      if k.startswith(("F0.", "N.", "F0_proj", "N_proj"))}
np.savez(Path(__file__).with_name("f0n_weights.npz"), **sd)
print("saved f0n_lstm.xml and", len(sd), "tensors:", sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in sd}))

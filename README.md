# kokoro_npu: Kokoro-82M split across GPU / CPU (and tested on the NPU)

Custom inference for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on an
Intel Core Ultra 5 225H (Arrow Lake: Arc 130T/140T iGPU, NPU 3720, 14-core CPU) with
OpenVINO. Goal: lowest time-to-first-audio, then energy, keeping Kokoro's sound.

## Placement (measured)

| Stage | Device | Why |
|---|---|---|
| G2P (misaki + espeak fallback) | CPU | rules/lookups, ~1 ms |
| PL-BERT (static token buckets 16-512) | GPU | 5 ms / 28 mJ vs NPU 11 ms / 53 mJ vs CPU 41 ms / 306 mJ |
| Prosody / F0-N / text-encoder LSTMs | CPU (dynamic) | NPU has no LSTM kernel; ~2% of compute |
| Alignment, harmonic source + STFT, iSTFT | CPU (numpy) | tiny, dynamic |
| Decoder + iSTFTNet vocoder (frame buckets 32-256) | GPU | ~85% of compute; NPU was 6x slower and inaccurate in fp16 |

The NPU ended up unused for TTS; it runs Whisper STT (see ../stt_daemon).

## How the decoder stays exact with padded (static) shapes

Every AdaIN instance-norm in Kokoro averages over the whole utterance, so padding would
change the sound. The exported decoder (`kparts.DecoderPart4D`) takes per-rate masks
(40/80/800/4800 Hz), computes norm statistics over valid frames only, and zeroes the
padding right before every conv with kernel > 1. Padded output matches the unpadded
original to ~4e-5. Other GPU-specific rewrites (all numerically equivalent):

- depthwise ConvTranspose (k3,s2) -> polyphase elementwise ops (GPU miscomputes it)
- nearest x2 upsample -> expand+reshape (GPU mis-fuses Interpolate + 1x1 conv)
- [1,C,T] -> [1,C,1,T] with 2D convs; norm stats as masked ReduceMean with the mask
  pre-scaled by 1/valid_fraction (avoids a GPU fusion bug and fp16 overflow)
- eps derived from an input so the GPU plugin doesn't push the norm chain to fp32

Also: the AdainResBlk1d pools use legacy weight_norm hooks; `bake_weight_norm` folds both
kinds (reading `.weight` of a legacy-hooked module gives stale random init weights).

## Files

- `kparts.py`   model split + export-friendly rewrites
- `dsp.py`      numpy harmonic source / STFT / iSTFT (ports of Kokoro's torch code)
- `export_ov.py` export parts to OpenVINO IR (`--dim 4d --norm reduce --eps-trick 1`)
- `engine.py`   runtime (OpenVINO + numpy + misaki, no torch)
- `kokorod.py`  streaming daemon, same socket protocol as ttsd
- `verify_parts.py`, `bench_devices.py`, `bench_energy.py`, `ttft.py`

## Environments

- `.venv`    (Python 3.12): torch + kokoro, for export/verification
- `.venv-rt` (Python 3.12): runtime only (openvino, numpy, misaki, spacy, espeak)

Rebuild models: `.venv/bin/python export_ov.py --dim 4d --norm reduce --eps-trick 1 --bert 16,32,64,128,256,512 --decoder 32,48,64,128,256`

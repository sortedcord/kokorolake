# kokoro_npu: Kokoro-82M split across GPU / CPU (and tested on the NPU)

Custom inference for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on an
Intel Core Ultra 5 225H (Arrow Lake: Arc 130T/140T iGPU, NPU 3720, 14-core CPU) with
OpenVINO. Goal: lowest time-to-first-audio, then energy, keeping Kokoro's sound.

## Current pipeline (v2)

| Stage | Where | First-chunk time |
|---|---|---|
| G2P (misaki + espeak fallback) | CPU | ~1 ms |
| PL-BERT (OpenVINO, static token buckets) | GPU | ~7 ms |
| text encoder LSTM | CPU, concurrently with BERT | hidden |
| prosody + F0/N LSTMs (OpenVINO, 4 threads) | CPU | ~12 ms |
| harmonic source + STFT | CPU (numpy) | ~6 ms |
| decoder + iSTFTNet generator | **GPU, hand-written OpenCL kernels on the XMX units** (`gpu/kdec.py`) | ~55-65 ms |
| iSTFT | CPU | ~2 ms |

Time to first audio ~85-105 ms, ~330 mJ per second of audio (Piper amy-medium: ~70 ms,
~490 mJ), real-time factor ~0.05, startup ~3 s.

### Custom decoder (`gpu/kdec.py`)
- Exact lengths: every kernel takes the sequence length at run time, so there is no
  padding, no masking, and no per-shape compilation (OpenVINO's GPU plugin needed 0.4-2 s
  per new shape, so the IR path used padded buckets with masked norms).
- Convs on the matrix engines: `intel_sub_group_f16_f16_matrix_mad_k16` (DPAS, SIMD8),
  activations time-major `[T, C]` fp16, weights pre-packed into DPAS A tiles.
  Transposed convs run as `stride` polyphase convs with an output row stride.
- Instance-norm statistics come from the conv epilogue (subgroup reductions -> partial
  rows -> one finalize kernel); norm-apply is fused with leaky-ReLU / snake.
- Per-call-site cl.Kernel objects keep their arguments, so a new utterance only re-sets
  length-dependent ints (pyopencl's __call__ costs ~110 us/launch; this ~5 us).
- 48-49 dB SNR against the fp32 PyTorch decoder.

### Things that mattered more than expected
- **numpy/OpenBLAS thread spinning**: OpenBLAS keeps ~8 worker threads spinning after
  each small matmul. On this power-capped laptop that stole the iGPU's power budget:
  decoder 122 -> 72 ms and ~950 -> 9 ms of CPU time per call with OPENBLAS_NUM_THREADS=1.
- OpenVINO CPU auto-threading on tiny LSTM graphs (f0n 26 ms / 181 ms CPU vs 9 ms / 35 ms
  with 4 threads).
- The iGPU is held at ~1 GHz (of 2.2) by PL4 on the USB-C charger (`throttle_reason_pl4`),
  and an animated wallpaper (mpvpaper) time-shares it.
- Updating the Quickshell island at request start competed with synthesis; it now
  updates when the first audio is ready.

## Platform tuning (Core Ultra 5 225H)

Measured on the full 14 s passage, energy = package RAPL minus idle:

| Change | TTFT | Energy / audio-s | CPU time |
|---|---|---|---|
| v2 baseline | 92 ms | 355 mJ | 755 ms |
| harmonic source + STFT + iSTFT on the iGPU (`source`/`stft`/`istft` kernels, exact fp32 sin) | 87 ms | 306 mJ | 670 ms |
| F0/N predictor's six AdainResBlk1d blocks on the iGPU (shared LSTM stays on CPU) | 85 ms | 281 mJ | 507 ms |
| OpenVINO CPU threads 4 -> 2 | 82 ms | 258 mJ | 397 ms |

Tried and dropped (no gain or worse):
- pinning to E-cores (4-11): same latency, energy within noise; LP-E cores (12-13):
  OpenVINO's CPU plugin refuses them
- GuC SLPC knobs (ignore efficient freq, min freq 1600 MHz): no effect, the iGPU is held
  at ~1 GHz by PL4 on the USB-C charger
- cl_khr_throttle_hints / cl_khr_priority_hints queues: within noise
- PL-BERT on the custom kernels (`KDecoder.bert`, correct to 3e-3): 21-29 ms vs 6.5-8.4 ms
  for OpenVINO's GPU plugin (tiny-M matmuls are latency-bound in this conv kernel)
- PL-BERT on the NPU: slower (10.7 ms) and +416 MB vs +225 MB on the GPU

## Placement study (v1, OpenVINO only)

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

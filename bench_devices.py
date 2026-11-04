"""Per-part, per-device benchmark: compile time, latency, energy, accuracy.

Energy = (package energy during a run loop) - (idle package power * time), per call.
Needs readable /sys/class/powercap/intel-rapl:0/energy_uj.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov

import dsp

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="bert,decoder")
ap.add_argument("--devices", default="CPU,GPU,NPU")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--F", type=int, default=128)
ap.add_argument("--hint", default="LATENCY")
ap.add_argument("--inputs", default="bench_inputs.npz")
ap.add_argument("--suffix", default="")
args = ap.parse_args()

M = Path(__file__).parent / "models"
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
core = ov.Core()
core.set_property({"CACHE_DIR": str(M / "cache")})
ap2 = None
x = np.load(args.inputs)


def energy_uj():
    return int(open(RAPL).read())


def idle_power(sec=2.0):
    e0, t0 = energy_uj(), time.perf_counter()
    time.sleep(sec)
    return (energy_uj() - e0) / 1e6 / (time.perf_counter() - t0)  # W


def pad(a, n):
    out = np.zeros(a.shape[:-1] + (n,), np.float32)
    out[..., : a.shape[-1]] = a
    return out


def mask(valid, n):
    m = np.zeros((1, 1, n), np.float32)
    m[..., :valid] = 1
    return m


def inputs(part):
    if part == "bert":
        L = len(x["ids"])
        ids = np.zeros((1, args.L), np.int64); ids[0, :L] = x["ids"]
        attn = np.zeros((1, args.L), np.int64); attn[0, :L] = 1
        return [ids, attn]
    Fr, Fb = x["asr"].shape[-1], args.F
    return [pad(x["asr"], Fb), pad(x["f0"], 2 * Fb), pad(x["n"], 2 * Fb), x["s"].astype(np.float32),
            pad(x["har"][None], 120 * Fb + 1), mask(Fr, Fb), mask(2 * Fr, 2 * Fb), mask(20 * Fr, 20 * Fb),
            mask(120 * Fr + 1, 120 * Fb + 1)]


def accuracy(part, outs):
    if part == "bert":
        L = len(x["ids"])
        ref, got = x["d_en"], outs[0][:, :, :L]
        return f"maxrel={np.abs(got - ref).max() / np.abs(ref).max():.1e}"
    T = 120 * x["asr"].shape[-1] + 1
    ref = dsp.istft(x["mag"][0].astype(np.float64), x["phase"][0].astype(np.float64))
    got = dsp.istft(outs[0][0, :, :T].astype(np.float64), outs[1][0, :, :T].astype(np.float64))
    snr = 10 * np.log10((ref ** 2).sum() / max(((got - ref) ** 2).sum(), 1e-20))
    return f"wave SNR vs torch fp32={snr:.1f} dB"


P_idle = idle_power()
print(f"idle package power {P_idle:.2f} W", flush=True)
audio_s = x["asr"].shape[-1] * 600 / 24000
for part in args.parts.split(","):
    name = f"bert_L{args.L}" if part == "bert" else f"decoder_F{args.F}{args.suffix}"
    model = core.read_model(M / f"{name}.xml")
    inp = inputs(part)
    for dev in args.devices.split(","):
        cfg = {"PERFORMANCE_HINT": args.hint}
        try:
            t = time.perf_counter()
            cm = core.compile_model(model, dev, cfg)
            comp = time.perf_counter() - t
            req = cm.create_infer_request()
            for _ in range(3):
                req.infer(inp)
            lat = []
            for _ in range(10):
                t = time.perf_counter(); req.infer(inp); lat.append(time.perf_counter() - t)
            # energy over ~3 s of back-to-back calls
            n, e0, t0 = 0, energy_uj(), time.perf_counter()
            while time.perf_counter() - t0 < 3.0:
                req.infer(inp); n += 1
            dt = time.perf_counter() - t0
            e_call = ((energy_uj() - e0) / 1e6 - P_idle * dt) / n
            outs = [req.get_output_tensor(i).data.copy() for i in range(len(cm.outputs))]
            print(f"{name:13s} {dev}: compile={comp:6.1f}s  latency med={np.median(lat)*1e3:7.1f} ms  min={min(lat)*1e3:7.1f} ms"
                  f"  energy/call={e_call*1e3:7.1f} mJ  avgP={((energy_uj()-e0)/1e6)/dt if False else e_call*n/dt + P_idle:5.1f} W  {accuracy(part, outs)}",
                  flush=True)
            del req, cm
        except Exception as e:
            print(f"{name:13s} {dev}: FAIL {str(e).splitlines()[0][:300]}", flush=True)
print(f"(audio in decoder bucket: {audio_s:.2f}s valid of {args.F*600/24000:.2f}s)")

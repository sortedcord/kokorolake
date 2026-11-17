"""Benchmark one Kokoro-82M implementation (run once per engine, each in its own process/venv).

    python bench/compare.py <engine> > result.json

Same text, same voice (af_aoede), same sentence/clause chunking for every engine; chunks are
synthesized one after another, so time-to-first-audio is the first chunk's latency.
Energy = package RAPL energy during synthesis minus idle power x time (needs readable
/sys/class/powercap/intel-rapl:0/energy_uj). Memory = peak RSS + GPU allocations (DRM fdinfo).
"""
import glob
import json
import os
import re
import statistics
import sys
import time

TEXT = ("Wait, what? All of the tests passed? I honestly didn't think that would work! Okay... but here's the bad news: "
        "the build broke on the main branch, and I'm not sure why. Do you want me to roll it back, or should I keep digging?")
VOICE = "af_aoede"
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
HOME = os.path.expanduser("~")


def split(text, first_max_chars=40):
    sents = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+|\n+", text) if s.strip()]
    out = []
    for i, s in enumerate(sents):
        if i == 0 and len(s) > first_max_chars:
            out.extend(p for p in re.split(r"(?<=[,—–])\s+", s, maxsplit=1) if p)
        else:
            out.append(s)
    return out


def energy():
    return int(open(RAPL).read()) / 1e6


def gpu_mb():
    seen, tot = set(), 0
    for f in glob.glob("/proc/self/fdinfo/*"):
        try:
            t = open(f).read()
        except OSError:
            continue
        if "drm-client-id" not in t:
            continue
        cid = [l for l in t.splitlines() if l.startswith("drm-client-id")][0]
        if cid in seen:
            continue
        seen.add(cid)
        tot += sum(int(l.split()[1]) for l in t.splitlines() if l.startswith("drm-total-system0"))
    return tot / 1024


def peak_rss_mb():
    return int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1]) / 1024


def make(engine):
    """Returns (synth(chunk) -> number of audio samples at 24 kHz, label)."""
    if engine == "kokoro-pytorch":
        from kokoro import KPipeline
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        def synth(c):
            return sum(len(r.audio) for r in pipe(c, voice=VOICE))
        return synth, "kokoro (PyTorch, CPU)"
    if engine.startswith("kokoro-onnx"):
        from kokoro_onnx import Kokoro
        d = os.environ.get("KOKORO_ONNX_DIR", f"{HOME}/kokoro-onnx")  # kokoro-v1.0[.int8].onnx + voices-v1.0.bin
        k = Kokoro(f"{d}/kokoro-v1.0{'.int8' if engine.endswith('int8') else ''}.onnx", f"{d}/voices-v1.0.bin")
        def synth(c):
            a, sr = k.create(c, voice=VOICE, lang="en-us")
            return len(a)
        return synth, f"kokoro-onnx {'int8' if engine.endswith('int8') else 'fp32'} (CPU)"
    if engine.startswith("openvino"):
        import numpy as np
        import openvino as ov
        import openvino_genai as g
        dev = engine.split("-")[1].upper()
        p = os.environ.get("KOKORO_OV_DIR", f"{HOME}/Kokoro-82M-int8-ov")  # huggingface OpenVINO/Kokoro-82M-int8-ov
        pipe = g.Text2SpeechPipeline(p, dev)
        sh = list(pipe.get_speaker_embedding_shape())
        emb = ov.Tensor(np.fromfile(f"{p}/voices/{VOICE}.bin", np.float32)[: int(np.prod(sh))].reshape(sh))
        def synth(c):
            r = pipe.generate(c, emb, language="en-us")
            return np.array(r.speeches[0].data).size
        return synth, f"OpenVINO GenAI int8 ({dev})"
    if engine == "kokorolake":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kokorolake.engine import Engine
        e = Engine()
        return (lambda c: sum(len(a) for a, _ in e.stream(c))), "kokorolake (iGPU + CPU)"
    raise SystemExit(f"unknown engine {engine}")


def main():
    engine = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    t = time.perf_counter()
    synth, label = make(engine)
    chunks = split(TEXT)
    tw = time.perf_counter()
    for c in chunks:  # first pass: includes any per-shape compilation a new sentence would pay
        synth(c)
    first_pass = time.perf_counter() - tw
    for _ in range(int(os.environ.get("WARM", "0"))):  # extra warm-up passes (steady state)
        for c in chunks:
            synth(c)
    load = time.perf_counter() - t
    time.sleep(float(os.environ.get("SETTLE", "1")))
    e0, t0 = energy(), time.perf_counter()
    time.sleep(4)
    idle_w = (energy() - e0) / (time.perf_counter() - t0)
    rows = []
    for _ in range(reps):
        time.sleep(1)
        e0, c0, t0 = energy(), time.process_time(), time.perf_counter()
        first, n = None, 0
        for c in chunks:
            n += synth(c)
            if first is None:
                first = time.perf_counter() - t0
        dt = time.perf_counter() - t0
        rows.append(dict(ttfa=first, total=dt, audio=n / 24000, cpu=time.process_time() - c0,
                         energy=(energy() - e0) - idle_w * dt))
    med = lambda k: statistics.median(r[k] for r in rows)
    audio = rows[0]["audio"]
    print(json.dumps(dict(engine=engine, label=label, load_s=load, first_pass_s=first_pass, idle_w=idle_w, audio_s=audio,
                          ttfa_ms=med("ttfa") * 1e3, total_ms=med("total") * 1e3, rtf=med("total") / audio,
                          cpu_s_per_audio_s=med("cpu") / audio, mj_per_audio_s=med("energy") / audio * 1e3,
                          peak_rss_mb=peak_rss_mb(), gpu_mb=gpu_mb(), reps=reps)))


if __name__ == "__main__":
    main()

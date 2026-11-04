"""Energy + latency per engine, same text. Run with each engine's own python:
  bench_energy.py piper|kokoro-onnx|custom
Energy = package (RAPL) energy during synthesis minus idle power x time."""
import sys, time, os, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
TEXT = ("Wait, what? All of the tests passed? I honestly didn't think that would work! Okay... but here's the bad news: "
        "the build broke on the main branch, and I'm not sure why. Do you want me to roll it back, or should I keep digging?")
eng = sys.argv[1]
E = lambda: int(open(RAPL).read())

if eng == "piper":
    from piper import PiperVoice
    v = PiperVoice.load(os.path.expanduser("~/Projects/piper_daemon/voices/en_US-amy-medium.onnx"))
    def stream():
        for c in v.synthesize(TEXT):
            yield len(c.audio_int16_bytes) // 2 / v.config.sample_rate
elif eng == "kokoro-onnx":
    from kokoro_onnx import Kokoro
    d = os.path.expanduser("~/Projects/tts_bench/kokoro")
    k = Kokoro(f"{d}/kokoro-v1.0.int8.onnx", f"{d}/voices-v1.0.bin")
    def stream():
        a, sr = k.create(TEXT, voice="af_aoede", lang="en-us")
        yield len(a) / sr
else:
    from engine import Engine
    e = Engine()
    def stream():
        for a, _ in e.stream(TEXT):
            yield len(a) / 24000

for _ in stream():  # warm-up
    pass
time.sleep(1)
e0, t0 = E(), time.perf_counter(); time.sleep(3); P_idle = (E() - e0) / 1e6 / (time.perf_counter() - t0)
rows = []
for rep in range(5):
    e0, t0 = E(), time.perf_counter(); first = None; dur = 0
    for d in stream():
        if first is None: first = time.perf_counter() - t0
        dur += d
    dt = time.perf_counter() - t0
    rows.append(((E() - e0) / 1e6 - P_idle * dt, dt, first, dur))
    time.sleep(0.5)
J = statistics.median(r[0] for r in rows); dt = statistics.median(r[1] for r in rows)
ttft = statistics.median(r[2] for r in rows); dur = rows[0][3]
print(f"{eng:12s} audio={dur:5.1f}s  TTFT={ttft*1e3:5.0f} ms  total={dt*1e3:5.0f} ms  RTF={dt/dur:.3f}  "
      f"energy={J:5.2f} J ({J/dur*1000:4.0f} mJ per audio-second)  idle={P_idle:.1f} W")

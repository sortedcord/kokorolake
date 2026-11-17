"""Charts for the README from bench/results/*.json  ->  docs/*.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = {p.stem: json.loads(p.read_text()) for p in (ROOT / "bench" / "results").glob("*.json")}
ORDER = ["kokorolake", "kokoro-onnx-int8", "openvino-cpu", "kokoro-onnx-fp32", "openvino-gpu", "kokoro-pytorch"]
NAMES = {"kokorolake": "kokorolake", "kokoro-onnx-int8": "kokoro-onnx int8 (CPU)",
         "openvino-cpu": "OpenVINO GenAI int8 (CPU)", "kokoro-onnx-fp32": "kokoro-onnx fp32 (CPU)",
         "openvino-gpu": "OpenVINO GenAI int8 (iGPU)", "kokoro-pytorch": "kokoro, PyTorch (CPU)"}
ACCENT, OTHER, INK, MUTED = "#2f6fdb", "#b9c0cc", "#1f2430", "#6b7280"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#d0d4db",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})


def bar(ax, key, fmt, title, sub, log=False, invert=False):
    rows = [(NAMES[e], R[e][key] if not invert else 1 / R[e][key]) for e in ORDER if e in R]
    rows.sort(key=lambda r: -r[1] if invert else r[1])
    labels, vals = zip(*rows)
    colors = [ACCENT if l == "kokorolake" else OTHER for l in labels]
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    ax.set_yticks(list(y), labels)
    if log:
        ax.set_xscale("log")
    xmax = max(vals) * (2.8 if log else 1.22)
    ax.set_xlim(right=xmax)
    for yi, v, l in zip(y, vals, labels):
        ax.text(v * (1.08 if log else 1) + (0 if log else xmax * 0.012), yi, fmt(v), va="center",
                color=INK, fontweight="bold" if l == "kokorolake" else "normal", fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=18)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eceef2", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / name, dpi=150, facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 3.3))
bar(ax, "ttfa_ms", lambda v: f"{v:,.0f} ms", "Time to first audio", "Lower is better. Log scale.", log=True)
save(fig, "ttfa.png")

fig, ax = plt.subplots(figsize=(8, 3.3))
bar(ax, "rtf", lambda v: f"{v:.1f}x", "Synthesis speed", "Seconds of speech produced per second of compute. Higher is better.", invert=True)
save(fig, "speed.png")

fig, ax = plt.subplots(figsize=(8, 3.3))
bar(ax, "mj_per_audio_s", lambda v: f"{v / 1000:.2f} J", "Energy per second of speech",
    "Package energy above idle (RAPL). Lower is better.")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
save(fig, "energy.png")

# memory: stacked RAM + GPU
fig, ax = plt.subplots(figsize=(8, 3.3))
rows = sorted([(NAMES[e], R[e]["peak_rss_mb"], R[e]["gpu_mb"]) for e in ORDER if e in R], key=lambda r: r[1] + r[2])
y = list(range(len(rows)))[::-1]
for yi, (l, ram, gpu) in zip(y, rows):
    c = ACCENT if l == "kokorolake" else OTHER
    ax.barh(yi, ram, color=c, height=0.62)
    ax.barh(yi, gpu, left=ram, color=c, height=0.62, alpha=0.55, hatch="////", edgecolor="white", linewidth=0)
    ax.text(ram + gpu + 40, yi, f"{ram + gpu:,.0f} MB", va="center", color=INK, fontsize=9.5,
            fontweight="bold" if l == "kokorolake" else "normal")
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlim(right=max(r[1] + r[2] for r in rows) * 1.2)
ax.set_title("Peak memory", loc="left", fontsize=12, fontweight="bold", color=INK, pad=18)
ax.text(0, 1.02, "RAM (solid) plus GPU allocations (hatched, also in system RAM). Lower is better.",
        transform=ax.transAxes, color=MUTED, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eceef2", linewidth=0.8)
ax.set_axisbelow(True)
save(fig, "memory.png")
print("wrote", sorted(p.name for p in (ROOT / "docs").glob("*.png")))

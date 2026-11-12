import os; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time; sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, openvino as ov
from kdec import KDecoder
d = KDecoder()
c = ov.Core()
full = c.compile_model(c.read_model("models/f0n.xml"), "CPU", {"INFERENCE_NUM_THREADS": 4}).create_infer_request()
lstm = c.compile_model(c.read_model("models/f0n_lstm.xml"), "CPU", {"INFERENCE_NUM_THREADS": 4}).create_infer_request()
rng = np.random.default_rng(0)
for F in (54, 110):
    en = (rng.standard_normal((1, 640, F)) * 0.5).astype(np.float32)
    s = (rng.standard_normal((1, 128)) * 0.3).astype(np.float32)
    r = full.infer([en, s]); f0_ref, n_ref = r[0][0], r[1][0]
    x = lstm.infer([en])[0][0]
    f0, n = d.f0n(x, s[0])
    rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
    w = {"cpu": [], "gpu": []}
    for _ in range(9):
        t = time.perf_counter(); full.infer([en, s]); w["cpu"].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); x = lstm.infer([en])[0][0]; d.f0n(x, s[0]); w["gpu"].append((time.perf_counter() - t) * 1e3)
    print(f"F={F}: f0 rel err {rel(f0, f0_ref):.1e}, n rel err {rel(n, n_ref):.1e} | CPU f0n {np.median(w['cpu']):.1f} ms, LSTM(CPU)+blocks(GPU) {np.median(w['gpu']):.1f} ms")

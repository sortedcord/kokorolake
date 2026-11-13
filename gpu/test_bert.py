import os; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, time; sys.path.insert(0, "."); sys.path.insert(0, "gpu")
import numpy as np, openvino as ov
from kdec import KDecoder
d = KDecoder()
c = ov.Core(); c.set_property({"CACHE_DIR": "models/cache"})
for f in ("in_first.npz", "bench_inputs.npz"):
    x = np.load(f); ids = x["ids"].astype(np.int64); L = len(ids)
    got = d.bert(ids)
    ref = x["d_en"][0]
    Lb = next(b for b in (16, 32, 64, 128, 256, 512) if L <= b)
    r = c.compile_model(c.read_model(f"models/bert_L{Lb}.xml"), "GPU", {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()
    ip = np.zeros((1, Lb), np.int64); ip[0, :L] = ids; at = np.zeros((1, Lb), np.int64); at[0, :L] = 1
    w = {"ov": [], "custom": []}
    for _ in range(9):
        t = time.perf_counter(); r.infer([ip, at]); w["ov"].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); d.bert(ids); w["custom"].append((time.perf_counter() - t) * 1e3)
    print(f"L={L}: rel err {np.abs(got - ref).max() / np.abs(ref).max():.1e} | OpenVINO GPU {np.median(w['ov']):.1f} ms, custom {np.median(w['custom']):.1f} ms")

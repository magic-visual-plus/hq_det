"""Multi-threaded stress test for TRT service (port 18901).

Usage: python stress.py <threads> <total_requests> [port]
Reports: p50/p90/p99 latency, throughput, per-stage timing from extra_info.
"""
import base64, json, sys, time, threading, queue
import urllib.request
import numpy as np

PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 18901
THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 40
URL = f"http://127.0.0.1:{PORT}/api/predict"

imgs = [l.strip() for l in open(r"C:\deploy\imglist.txt", encoding="utf-8") if l.strip()]
print(f"threads={THREADS} total={TOTAL} imgs_avail={len(imgs)}")

# pre-encode a pool of images (rotate through them)
POOL = []
for i in range(min(len(imgs), 12)):
    b = open(imgs[i], "rb").read()
    POOL.append(base64.b64encode(b).decode())
print(f"encoded pool: {len(POOL)} imgs")

results = []
results_lock = threading.Lock()
errors = []
work = queue.Queue()
for i in range(TOTAL):
    work.put(i)

def worker():
    while True:
        try:
            i = work.get_nowait()
        except queue.Empty:
            return
        body = json.dumps({
            "request_id": f"stress-{i}",
            "model_id": "swin_l_trt",
            "image_data": [{"image_data": POOL[i % len(POOL)],
                            "source_type": "network_b64",
                            "file_name": f"stress-{i}.jpg"}],
            "threshold": 0.3, "extra": {}}).encode()
        t0 = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"}), timeout=300)
            resp = json.loads(r.read())
            dt = (time.time() - t0) * 1000
            with results_lock:
                results.append({"wall": dt, "cost": resp.get("cost_ms", 0),
                                "extra": resp["results"][0].get("extra_info", {}),
                                "n": len(resp["results"][0].get("predictions", []))})
        except Exception as e:
            with results_lock:
                errors.append(f"{type(e).__name__}: {e}")

t_start = time.time()
threads = [threading.Thread(target=worker) for _ in range(THREADS)]
for t in threads: t.start()
for t in threads: t.join()
wall_total = time.time() - t_start

w = np.array([r["wall"] for r in results])
c = np.array([r["cost"] for r in results])
eng = np.array([r["extra"].get("engine_ms", 0) for r in results])
head = np.array([r["extra"].get("head_ms", 0) for r in results])
n = np.array([r["n"] for r in results])

print(f"\n=== STRESS RESULT threads={THREADS} total={TOTAL} ===")
print(f"wall_total={wall_total:.1f}s errors={len(errors)}")
if errors[:3]: print("err samples:", errors[:3])
print(f"throughput={TOTAL/wall_total:.2f} req/s")
print(f"latency p50={np.percentile(w,50):.0f} p90={np.percentile(w,90):.0f} p99={np.percentile(w,99):.0f} min={w.min():.0f} max={w.max():.0f} (ms)")
print(f"server cost_ms p50={np.percentile(c,50):.0f}")
print(f"engine p50={np.percentile(eng,50):.0f}ms head p50={np.percentile(head,50):.1f}ms")
print(f"detections/img p50={np.percentile(n,50):.0f}")

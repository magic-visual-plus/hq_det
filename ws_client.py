import asyncio, base64, json, sys, time
import websockets

async def test():
    uri = sys.argv[1]
    img_path = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    req = json.dumps({"image_b64": b64, "confidence": conf})
    async with websockets.connect(uri, max_size=64 << 20) as ws:
        await ws.send(req)
        json.loads(await ws.recv())  # warmup
        times = []
        for _ in range(5):
            await ws.send(req)
            t0 = time.time()
            resp = json.loads(await ws.recv())
            times.append((time.time()-t0)*1000)
        print("WS_ROUNDTRIP_MS:", [round(t,1) for t in times])
    if resp.get("error"):
        print("ERROR:", resp["error"])
        return
    print("time_ms:", resp["time_ms"])
    print("n_dets:", len(resp["detections"]))
    for d in sorted(resp["detections"], key=lambda x: -x["score"])[:8]:
        print(f"  [{d['label']}] {d['score']} bbox={d['bbox']}")
    print("CLIENT_OK")

asyncio.run(test())

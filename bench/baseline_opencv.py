"""Baseline: the common OpenCV pipeline camtensor replaces.

N threads, each: cv2.VideoCapture(rtsp) -> letterbox resize -> BGR2RGB ->
float32 / 255 -> HWC->CHW, then batches are stacked in the main thread.
Prints a JSON result comparable to `camtensor bench`.

    python bench/baseline_opencv.py rtsp://127.0.0.1:8554/cam1 --streams 8 --seconds 15
"""

import argparse
import json
import queue
import threading
import time

import cv2
import numpy as np
import psutil


def letterbox(img, size, pad=114):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    out = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def worker(url, size, q, stop, stats):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        t0 = time.perf_counter()
        x = letterbox(frame, size)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        stats["prep_ms"].append((time.perf_counter() - t0) * 1e3)
        try:
            q.put_nowait(x)
        except queue.Full:
            stats["dropped"] += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--json")
    a = ap.parse_args()

    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    stats = {"prep_ms": [], "dropped": 0}
    threads = [
        threading.Thread(target=worker, args=(a.urls[i % len(a.urls)], a.size, q, stop, stats), daemon=True)
        for i in range(a.streams)
    ]
    for t in threads:
        t.start()
    proc = psutil.Process()
    time.sleep(a.warmup)
    while not q.empty():
        q.get_nowait()
    stats["prep_ms"].clear()
    proc.cpu_percent(None)
    images, cpu, rss = 0, [], 0.0
    t0 = time.perf_counter()
    next_sample = t0 + 1
    while time.perf_counter() - t0 < a.seconds:
        items = []
        try:
            items.append(q.get(timeout=0.1))
            while len(items) < a.batch:
                items.append(q.get_nowait())
        except queue.Empty:
            pass
        if items:
            np.stack(items)
            images += len(items)
        if time.perf_counter() >= next_sample:
            cpu.append(proc.cpu_percent(None))
            rss = max(rss, proc.memory_info().rss / 2**20)
            next_sample += 1
    secs = time.perf_counter() - t0
    stop.set()
    prep = sorted(stats["prep_ms"]) or [0]
    r = {
        "pipeline": "opencv",
        "streams": a.streams,
        "images_per_sec": images / secs,
        "per_stream_fps": images / secs / a.streams,
        "preprocess_ms_p50": prep[len(prep) // 2],
        "cpu_percent": sum(cpu) / max(len(cpu), 1),
        "rss_mb": rss,
    }
    print(json.dumps(r, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()

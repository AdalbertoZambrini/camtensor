"""Soak test: many streams for a long time, with churn.

Starts MediaMTX and file-based publishers (stream copy, so publishing costs
almost no CPU), opens `--streams` RTSP sessions spread over the publishers,
consumes every batch, and every `--churn-min` minutes:
  * restarts one publisher (camera reboot),
  * removes one stream from the engine and adds it back.

Every `--sample-s` seconds it records process RSS, threads and handles,
GPU memory, delivered images/s, reconnects and errors. At the end it
writes JSON + CSV and prints whether memory stayed flat.

    python bench/soak.py --minutes 60 --streams 16 --out target/soak
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import psutil

import camtensor as ct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA = os.path.join(ROOT, "testdata", "media")
CLIPS = [
    "h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4",
    "h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4",
    "h264/h264-main-1920x1080-25fps-gop4s-nob-vbr-4m.mkv",
    "mjpeg/mjpeg-1280x720-25fps-q5.mkv",
]


def tool(env, default, name):
    for c in (os.environ.get(env), default, shutil.which(name)):
        if c and os.path.exists(c):
            return c
    raise SystemExit(f"{name} not found (set {env})")


def gpu_mem_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.decoder", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return float("nan"), float("nan")


class Publishers:
    def __init__(self, ffmpeg, port):
        self.ffmpeg, self.port, self.procs = ffmpeg, port, {}

    def start(self, i):
        clip = os.path.join(MEDIA, CLIPS[i % len(CLIPS)])
        self.procs[i] = subprocess.Popen(
            [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-stream_loop", "-1",
             "-i", clip, "-an", "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp",
             f"rtsp://127.0.0.1:{self.port}/src{i}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def restart(self, i, down_s=3):
        self.procs[i].kill()
        self.procs[i].wait()
        time.sleep(down_s)
        self.start(i)

    def stop(self):
        for p in self.procs.values():
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--churn-min", type=float, default=5)
    ap.add_argument("--sample-s", type=float, default=30)
    ap.add_argument("--port", type=int, default=18554)
    ap.add_argument("--out", default=os.path.join(ROOT, "target", "soak"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mediamtx = tool("CAMTENSOR_MEDIAMTX", r"C:\mediamtx-ffmpeg\mediamtx\mediamtx.exe", "mediamtx")
    ffmpeg = tool("CAMTENSOR_FFMPEG", r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe", "ffmpeg")
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "mediamtx.yml"), "w") as f:
        f.write(f"logLevel: error\napi: no\nrtspAddress: 127.0.0.1:{a.port}\nrtspTransports: [tcp]\n"
                "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\npaths:\n  all_others:\n")
    mtx = subprocess.Popen([mediamtx, os.path.join(tmp, "mediamtx.yml")], cwd=tmp,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    pubs = Publishers(ffmpeg, a.port)
    for i in range(len(CLIPS)):
        pubs.start(i)
    time.sleep(3)

    eng = ct.Engine(device=a.device)
    eng.add_output("o", 640, 640, dtype="f16", batch=8, max_wait_ms=20, queue_depth=8)

    def url(i):
        return f"rtsp://127.0.0.1:{a.port}/src{i % len(CLIPS)}"

    for i in range(a.streams):
        eng.add_stream(f"s{i:02d}", url(i), ["o"])

    images = [0]
    stop = threading.Event()

    def consume():
        while not stop.is_set():
            try:
                b = eng.recv("o", timeout=0.5)
                images[0] += b.valid_count
            except TimeoutError:
                pass

    th = threading.Thread(target=consume, daemon=True)
    th.start()

    proc = psutil.Process()
    rows = []
    t0 = time.time()
    next_sample, next_churn, churn_i = t0 + 60, t0 + a.churn_min * 60, 0
    last_images, last_t = 0, t0
    end = t0 + a.minutes * 60
    print(f"soak: {a.streams} streams on {eng.device} for {a.minutes} min", flush=True)
    while time.time() < end:
        time.sleep(1)
        now = time.time()
        if now >= next_churn:
            pubs.restart(churn_i % len(CLIPS))
            sid = f"s{churn_i % a.streams:02d}"
            eng.remove_stream(sid)
            eng.add_stream(sid, url(churn_i % a.streams), ["o"])
            churn_i += 1
            next_churn += a.churn_min * 60
        if now >= next_sample:
            st = eng.stats()
            mem, dec = gpu_mem_mib()
            n = images[0]
            row = {
                "t_min": round((now - t0) / 60, 2),
                "rss_mb": round(proc.memory_info().rss / 2**20, 1),
                "threads": proc.num_threads(),
                "handles": proc.num_handles() if hasattr(proc, "num_handles") else proc.num_fds(),
                "gpu_mem_mib": mem,
                "nvdec_pct": dec,
                "images_per_s": round((n - last_images) / (now - last_t), 1),
                "connected": sum(s["connected"] for s in st["streams"]),
                "reconnects": sum(s["reconnects"] for s in st["streams"]),
                "decode_errors": sum(s["decode_errors"] for s in st["streams"]),
                "dropped": st["outputs"]["o"]["dropped_images"],
                "buffers": st["outputs"]["o"]["buffers"],
                "churn_events": churn_i,
            }
            rows.append(row)
            last_images, last_t = n, now
            print(json.dumps(row), flush=True)
            next_sample += a.sample_s

    stop.set()
    th.join(timeout=2)
    eng.close()
    pubs.stop()
    mtx.kill()

    with open(os.path.join(a.out, "soak.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Memory trend: compare the first and last 20% of samples (after warm-up).
    k = max(1, len(rows) // 5)
    first, last = rows[:k], rows[-k:]
    avg = lambda rs, key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
    hours = max(rows[-1]["t_min"] - rows[0]["t_min"], 1e-9) / 60
    summary = {
        "streams": a.streams,
        "device": a.device,
        "minutes": a.minutes,
        "churn_events": churn_i,
        "samples": len(rows),
        "rss_start_mb": avg(first, "rss_mb"),
        "rss_end_mb": avg(last, "rss_mb"),
        "rss_growth_mb_per_h": (avg(last, "rss_mb") - avg(first, "rss_mb")) / hours,
        "gpu_mem_start_mib": avg(first, "gpu_mem_mib"),
        "gpu_mem_end_mib": avg(last, "gpu_mem_mib"),
        "threads_start": first[0]["threads"],
        "threads_end": last[-1]["threads"],
        "handles_start": first[0]["handles"],
        "handles_end": last[-1]["handles"],
        "images_per_s_avg": avg(rows, "images_per_s"),
        "reconnects_total": rows[-1]["reconnects"],
        "decode_errors_total": rows[-1]["decode_errors"],
    }
    with open(os.path.join(a.out, "soak.json"), "w") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

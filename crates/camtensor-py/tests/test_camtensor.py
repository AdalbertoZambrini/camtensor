"""Tests for the camtensor Python bindings.

The streaming test starts MediaMTX and an FFmpeg publisher when they are
available (CAMTENSOR_MEDIAMTX / CAMTENSOR_FFMPEG or the default dev paths);
otherwise it is skipped.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np
import pytest

import camtensor as ct


def tool(env, default, name):
    for cand in (os.environ.get(env), default, shutil.which(name)):
        if cand and os.path.exists(cand):
            return cand
    return None


MEDIAMTX = tool("CAMTENSOR_MEDIAMTX", r"C:\mediamtx-ffmpeg\mediamtx\mediamtx.exe", "mediamtx")
FFMPEG = tool("CAMTENSOR_FFMPEG", r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe", "ffmpeg")


def test_version_and_capabilities():
    assert ct.__version__ == ct.version()
    caps = ct.capabilities()
    assert set(caps) >= {"cuda", "ffmpeg", "jpeg"}


def test_config_and_errors():
    eng = ct.Engine({"device": "cpu", "outputs": {"a": {"tensor": {"width": 32, "height": 16}}}})
    assert eng.device == "cpu"
    assert eng.outputs == ["a"]
    eng.add_output("b", 64, 64, dtype="f16", layout="nhwc", normalize="imagenet", batch=4, max_wait_ms=5)
    cfg = eng.config()
    assert cfg["outputs"]["b"]["tensor"]["dtype"] == "f16"
    with pytest.raises(TimeoutError):
        eng.recv("a", timeout=0.05)
    with pytest.raises(KeyError):
        eng.recv("missing", timeout=0.01)
    with pytest.raises(ValueError):
        eng.add_output("c", dtype="f64")
    with pytest.raises(ValueError):
        eng.add_stream("x", "http://not-rtsp", ["a"])
    with pytest.raises(ValueError):
        ct.Engine("outputs: {a: {tensr: {}}}")
    assert "camtensor_" in eng.prometheus() or eng.prometheus() != ""
    eng.close()


def free_port_pair():
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port % 2 == 0 and port < 60000:
            return port


@pytest.mark.skipif(not (MEDIAMTX and FFMPEG), reason="mediamtx/ffmpeg not available")
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_stream_to_numpy_and_dlpack(device):
    base = free_port_pair()
    tmp = tempfile.mkdtemp()
    yml = os.path.join(tmp, "mediamtx.yml")
    with open(yml, "w") as f:
        f.write(
            f"logLevel: error\napi: no\nrtspAddress: 127.0.0.1:{base}\nrtspTransports: [tcp]\n"
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n"
        )
    mtx = subprocess.Popen([MEDIAMTX, yml], cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"rtsp://127.0.0.1:{base}/cam"
    pub = None
    try:
        time.sleep(0.5)
        pub = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi",
             "-i", "testsrc2=size=640x360:rate=25", "-c:v", "libx264", "-preset", "ultrafast",
             "-tune", "zerolatency", "-g", "25", "-pix_fmt", "yuv420p", "-f", "rtsp", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        with ct.Engine(device=device) as eng:
            eng.add_output("det", 160, 160, dtype="f32", batch=2, max_wait_ms=50, keep_frame=True,
                           device="cpu")
            eng.add_stream("cam", url, ["det"])
            batch = eng.recv("det", timeout=20)
            assert len(batch) >= 1
            item = batch[0]
            assert item.stream_id == "cam" and item.src_width == 640 and item.src_height == 360
            arr = batch.numpy()
            assert arr.shape == (len(batch), 3, 160, 160) and arr.dtype == np.float32
            assert 0.0 <= arr.min() and arr.max() <= 1.0
            # Letterbox 640x360 -> 160x160: scale 0.25, 35 padding rows on top.
            assert np.allclose(arr[0, :, 0, :], 114 / 255, atol=1e-3)
            dl = np.from_dlpack(batch.tensor)
            assert np.array_equal(dl, arr)
            x1, y1, x2, y2 = item.box_to_source([0, 35, 160, 125])
            assert abs(x2 - 640) < 1e-3 and abs(y2 - 360) < 1e-3
            crops = eng.crop(item.frame, [(0, 0, 320, 180), [[100, 100], [140, 100], [120, 120], [105, 140], [135, 140]]],
                             normalize="symmetric", device="cpu")
            c = crops.numpy()
            assert c.shape == (2, 3, 112, 112) and c.min() >= -1 and c.max() <= 1
            stats = eng.stats()
            assert stats["streams"][0]["connected"]
    finally:
        if pub:
            pub.kill()
        mtx.kill()


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_file_source_region_and_snapshot(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=1280x720:rate=25", "-frames:v", "30", "-c:v", "libx264", "-g", "10",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    with ct.Engine(device="auto") as eng:
        eng.add_output("o", 160, 160, dtype="u8", layout="nhwc", keep_frame=True,
                       backpressure="block")
        eng.add_stream("f", f"file:{clip}", ["o"], realtime=False, loop=False,
                       region=(0.5, 0.0, 0.5, 1.0))
        n, first = 0, None
        deadline = time.time() + 20
        while n < 30 and time.time() < deadline:
            try:
                b = eng.recv("o", timeout=0.5)
            except TimeoutError:
                continue
            n += b.valid_count
            first = first or b
        assert n == 30, "every frame of the file is delivered"
        item = first[0]
        x, y = item.to_source(80, 80)
        assert abs(x - 960) < 1 and abs(y - 360) < 1, "region centre maps back to full-frame pixels"
        jpeg = eng.snapshot(item.frame, quality=85)
        assert jpeg[:2] == b"\xff\xd8" and len(jpeg) > 10_000
        small = eng.snapshot(item.frame, region=(0, 0, 640, 360), width=320)
        assert small[:2] == b"\xff\xd8" and len(small) < len(jpeg)
        with pytest.raises(ValueError):
            eng.snapshot(item.frame, region=(5000, 0, 10, 10))
        with pytest.raises(ValueError):
            eng.add_stream("bad", "file:x.mp4", ["o"], region=(0.8, 0, 0.5, 1))


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_async_iterator(tmp_path):
    import asyncio

    clip = tmp_path / "async.mkv"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=25", "-frames:v", "20", "-c:v", "mjpeg", "-q:v", "5",
         "-pix_fmt", "yuvj420p", str(clip)],
        check=True,
    )

    async def main():
        eng = ct.Engine(device="cpu")
        eng.add_output("o", 64, 64, dtype="u8", backpressure="block")
        # Real-time playback (~0.8 s) so the event loop has time to run.
        eng.add_stream("f", f"file:{clip}", ["o"], realtime=True, loop=False)
        first = await eng.arecv("o", timeout=10)
        n = first.valid_count
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        async for b in eng.abatches("o"):
            n += b.valid_count
            if n >= 20:
                break
        t.cancel()
        eng.close()
        # The loop kept running while waiting for batches.
        return n, ticks

    n, ticks = asyncio.run(main())
    assert n == 20
    assert ticks > 0


def test_list_devices():
    try:
        devices = ct.list_devices()
    except RuntimeError:
        pytest.skip("FFmpeg (libavdevice) not available")
    assert isinstance(devices, list)
    for name, desc in devices:
        assert isinstance(name, str) and isinstance(desc, str)


def test_webcam_if_present():
    try:
        devices = ct.list_devices()
    except RuntimeError:
        pytest.skip("libavdevice not available")
    if not devices:
        pytest.skip("no capture device")
    name, desc = devices[0]
    url = "device:" + (desc if os.name == "nt" and desc else name)
    with ct.Engine(device="auto") as eng:
        eng.add_output("o", 160, 160, dtype="u8", keep_frame=True)
        eng.add_stream("cam", url, ["o"])
        b = eng.recv("o", timeout=20)
        assert b[0].src_width > 0
        assert eng.snapshot(b[0].frame)[:2] == b"\xff\xd8"

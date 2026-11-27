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
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\npaths:\n  all_others:\n"
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
                             normalize="arcface", device="cpu")
            c = crops.numpy()
            assert c.shape == (2, 3, 112, 112) and c.min() >= -1 and c.max() <= 1
            stats = eng.stats()
            assert stats["streams"][0]["connected"]
    finally:
        if pub:
            pub.kill()
        mtx.kill()

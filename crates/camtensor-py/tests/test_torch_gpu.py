"""PyTorch + CUDA interoperability (skipped without torch/CUDA/tools).

A still image is streamed as H.264. The same frame is preprocessed by the
CUDA kernel and handed to PyTorch zero-copy through DLPack. It is also
preprocessed by the CPU sampler of a CPU-only engine, which gives an
independent reference for the values.
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

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ct.capabilities()["cuda"], reason="CUDA not available"
)


def tool(env, default, name):
    for cand in (os.environ.get(env), default, shutil.which(name)):
        if cand and os.path.exists(cand):
            return cand
    return None


MEDIAMTX = tool("CAMTENSOR_MEDIAMTX", r"C:\mediamtx-ffmpeg\mediamtx\mediamtx.exe", "mediamtx")
FFMPEG = tool("CAMTENSOR_FFMPEG", r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe", "ffmpeg")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STILL = os.path.join(ROOT, "testdata", "media", "frames", "bbb-rabbit-1920x1080.jpg")


def free_even_port():
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p % 2 == 0 and p < 60000:
            return p


@pytest.fixture(scope="module")
def still_stream():
    if not (MEDIAMTX and FFMPEG):
        pytest.skip("mediamtx/ffmpeg not available")
    port = free_even_port()
    tmp = tempfile.mkdtemp()
    yml = os.path.join(tmp, "mediamtx.yml")
    with open(yml, "w") as f:
        f.write(
            f"logLevel: error\napi: no\nrtspAddress: 127.0.0.1:{port}\nrtspTransports: [tcp]\n"
            "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n"
        )
    mtx = subprocess.Popen([MEDIAMTX, yml], cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"rtsp://127.0.0.1:{port}/still"
    src = ["-loop", "1", "-framerate", "10", "-i", STILL] if os.path.exists(STILL) else [
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=10"]
    time.sleep(0.5)
    pub = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-re", *src,
         "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-qp", "10", "-g", "10", "-bf", "0", "-f", "rtsp", "-rtsp_transport", "tcp", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    yield url
    pub.kill()
    mtx.kill()


def first_batch(device, url, **kw):
    eng = ct.Engine(device=device)
    eng.add_output("o", 320, 320, keep_frame=True, **kw)
    eng.add_stream("cam", url, ["o"], decoder="cpu" if device == "cpu" else None)
    # Skip the first batch (encoder warm-up), take the next one.
    eng.recv("o", timeout=20)
    return eng, eng.recv("o", timeout=20)


def test_dlpack_zero_copy_on_cuda(still_stream):
    eng, batch = first_batch("cuda:0", still_stream, dtype="f16", batch=1)
    x = torch.from_dlpack(batch)
    assert x.is_cuda and x.device.index == 0
    assert x.dtype == torch.float16
    assert tuple(x.shape) == (1, 3, 320, 320)
    assert x.data_ptr() == batch.tensor.data_ptr, "zero-copy: torch aliases camtensor memory"
    assert x.is_contiguous()
    # Usable by kernels right away (the engine synchronises before delivery).
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert abs(float(x[0, :, 0, 0].float().mean()) - 114 / 255) < 1e-3  # letterbox padding
    # The batch buffer stays valid while torch holds the tensor.
    del batch
    for _ in range(5):
        eng.recv("o", timeout=10)
    assert torch.isfinite(x).all()
    cai = eng.recv("o", timeout=10).tensor.__cuda_array_interface__
    assert cai["typestr"] == "<f2" and cai["shape"] == (1, 3, 320, 320)
    eng.close()


def test_cuda_kernel_matches_cpu_sampler(still_stream):
    g_eng, g = first_batch("cuda:0", still_stream, dtype="f32")
    c_eng, c = first_batch("cpu", still_stream, dtype="f32")
    gpu = torch.from_dlpack(g).cpu().numpy()
    cpu = c.numpy()
    assert gpu.shape == cpu.shape
    diff = np.abs(gpu - cpu)
    # Same still image, decoded by NVDEC vs libavcodec and sampled by the
    # CUDA kernel vs the CPU sampler: differences come from frame-to-frame
    # encoder noise only.
    assert diff.mean() < 2 / 255, diff.mean()
    assert np.percentile(diff, 99) < 8 / 255
    g_eng.close()
    c_eng.close()


def test_crops_to_torch(still_stream):
    eng, batch = first_batch("cuda:0", still_stream, dtype="f16")
    item = batch[0]
    pts = [[x * 4 + 700, y * 4 + 200] for x, y in ct.KEYPOINT_TEMPLATE_112]
    crops = eng.crop(item.frame, [pts, (0, 0, 960, 540)], 112, 112, dtype="f16", normalize="symmetric")
    t = torch.from_dlpack(crops)
    assert t.is_cuda and tuple(t.shape) == (2, 3, 112, 112)
    assert float(t.min()) >= -1.0 and float(t.max()) <= 1.0
    # Run a real op on the crops (what a second model would do).
    pooled = torch.nn.functional.adaptive_avg_pool2d(t.float(), 1)
    assert pooled.shape == (2, 3, 1, 1)
    eng.close()

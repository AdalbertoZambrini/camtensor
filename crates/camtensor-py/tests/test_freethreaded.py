"""Free-threaded CPython (3.14t+): the extension must not re-enable the GIL
and must stay correct with many threads using one engine concurrently.
Skipped on regular (GIL) builds."""

import os
import shutil
import subprocess
import sys
import sysconfig
import threading
import time

import pytest

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
pytestmark = pytest.mark.skipif(not FREE_THREADED, reason="not a free-threaded CPython build")


def tool(env, default, name):
    for cand in (os.environ.get(env), default, shutil.which(name)):
        if cand and os.path.exists(cand):
            return cand
    return None


FFMPEG = tool("CAMTENSOR_FFMPEG", r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe", "ffmpeg")


def test_import_keeps_gil_disabled():
    import camtensor  # noqa: F401

    assert not sys._is_gil_enabled(), "importing camtensor re-enabled the GIL"


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_concurrent_consumers_crops_and_snapshots(tmp_path):
    import camtensor as ct

    clip = tmp_path / "ft.mkv"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=25", "-frames:v", "50", "-c:v", "mjpeg", "-q:v", "5",
         "-pix_fmt", "yuvj420p", str(clip)],
        check=True,
    )
    eng = ct.Engine(device="auto")
    eng.add_output("o", 160, 160, dtype="f32", batch=4, max_wait_ms=20, keep_frame=True,
                   device="cpu", queue_depth=16)
    for i in range(4):
        eng.add_stream(f"s{i}", f"file:{clip}", ["o"], realtime=True, loop=True)

    errors, counts = [], [0] * 6
    stop = time.monotonic() + 4.0

    def consumer(k):
        try:
            while time.monotonic() < stop:
                try:
                    b = eng.recv("o", timeout=0.2)
                except TimeoutError:
                    continue
                arr = b.numpy()
                assert arr.shape[1:] == (3, 160, 160)
                item = b[0]
                crops = eng.crop(item.frame, [(0, 0, 320, 180), (320, 180, 320, 180)], 64, 64,
                                 device="cpu")
                assert crops.numpy().shape == (2, 3, 64, 64)
                assert eng.snapshot(item.frame, width=160)[:2] == b"\xff\xd8"
                counts[k] += len(b)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    def observer(k):
        try:
            while time.monotonic() < stop:
                st = eng.stats()
                assert len(st["streams"]) == 4
                eng.prometheus()
                counts[k] += 1
                time.sleep(0.01)
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=consumer, args=(k,)) for k in range(4)]
    threads += [threading.Thread(target=observer, args=(k,)) for k in (4, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    eng.close()

    assert not errors, errors
    assert sum(counts[:4]) > 50, counts
    assert all(c > 0 for c in counts[4:]), counts
    assert not sys._is_gil_enabled()

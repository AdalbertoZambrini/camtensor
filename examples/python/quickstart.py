"""Quickstart: stream a camera into PyTorch (GPU) or NumPy (CPU).

    python examples/python/quickstart.py rtsp://127.0.0.1:8554/cam1
"""

import sys

import camtensor as ct

url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/cam1"

with ct.Engine(device="auto") as eng:
    print("device:", eng.device, "capabilities:", ct.capabilities())
    eng.add_output("detector", 640, 640, dtype="f16", batch=4, max_wait_ms=20)
    eng.add_stream("cam1", url, ["detector"])
    for i, batch in enumerate(eng.batches("detector")):
        try:
            import torch

            x = torch.from_dlpack(batch)  # zero-copy, on cuda:N or cpu
            info = f"torch {tuple(x.shape)} {x.dtype} {x.device}"
        except ImportError:
            info = f"tensor {batch.tensor.shape} {batch.tensor.dtype} {batch.tensor.device}"
        item = batch[0]
        print(f"batch {batch.seq}: {info} from {item.stream_id} {item.src_width}x{item.src_height}")
        if i == 50:
            break
    print(eng.stats()["streams"][0])

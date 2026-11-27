"""camtensor: camera streams to inference-ready tensors.

RTSP ingest, NVDEC/CPU decoding and fused preprocessing (colour conversion,
resize/letterbox, normalisation, layout) straight into batched tensors on the
GPU or in host memory. Tensors are exported zero-copy via DLPack::

    import camtensor as ct, torch

    eng = ct.Engine(device="auto")
    eng.add_output("detector", 640, 640, dtype="f16", batch=8, max_wait_ms=15, keep_frame=True)
    eng.add_stream("cam1", "rtsp://127.0.0.1:8554/cam1", ["detector"], gating="auto")

    for batch in eng.batches("detector"):
        x = torch.from_dlpack(batch)          # [N, 3, 640, 640] float16 on cuda:0
        for item in batch.items:
            print(item.stream_id, item.latency_ms)
"""

from ._camtensor import (  # noqa: F401
    Batch,
    BatchIterator,
    Engine,
    Frame,
    Item,
    Tensor,
    __version__,
    capabilities,
    version,
)

#: Reference 5-point template (112x112) used for ArcFace alignment.
ARCFACE_112_TEMPLATE = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

__all__ = [
    "ARCFACE_112_TEMPLATE",
    "Batch",
    "BatchIterator",
    "Engine",
    "Frame",
    "Item",
    "Tensor",
    "__version__",
    "capabilities",
    "version",
]

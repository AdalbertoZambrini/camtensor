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

import asyncio as _asyncio

from ._camtensor import (  # noqa: F401
    Batch,
    BatchIterator,
    Frame,
    Item,
    Tensor,
    __version__,
    capabilities,
    list_devices,
    version,
)
from ._camtensor import Engine as _NativeEngine


class Engine(_NativeEngine):
    """Camera streams in, batches of tensors out. See ``help(camtensor)``.

    Adds asyncio support on top of the native engine::

        async for batch in eng.abatches("detector"):
            ...
    """

    async def arecv(self, output, timeout=None, device=None):
        """Awaitable :meth:`recv`: waits in a worker thread (the GIL is
        released while waiting), so the event loop stays responsive."""
        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.recv, output, timeout, device)

    def abatches(self, output, poll=0.25, device=None):
        """Async iterator over the batches of ``output`` (only those made on
        ``device`` if given); ends when the engine is closed."""
        return _AsyncBatches(self, output, poll, device)


class _AsyncBatches:
    def __init__(self, engine, output, poll, device=None):
        self._engine, self._output, self._poll, self._device = engine, output, poll, device

    def __aiter__(self):
        return self

    async def __anext__(self):
        loop = _asyncio.get_running_loop()
        while True:
            try:
                return await loop.run_in_executor(
                    None, self._engine.recv, self._output, self._poll, self._device
                )
            except TimeoutError:
                continue
            except RuntimeError as e:
                if "closed" in str(e):
                    raise StopAsyncIteration from None
                raise

#: Default 5-keypoint reference layout (112x112) used by keypoint-aligned crops.
KEYPOINT_TEMPLATE_112 = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

__all__ = [
    "KEYPOINT_TEMPLATE_112",
    "Batch",
    "BatchIterator",
    "Engine",
    "Frame",
    "Item",
    "Tensor",
    "__version__",
    "capabilities",
    "list_devices",
    "version",
]

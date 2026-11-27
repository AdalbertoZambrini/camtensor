from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Tuple, Union

__version__: str
KEYPOINT_TEMPLATE_112: Tuple[Tuple[float, float], ...]

DType = Literal["u8", "uint8", "f16", "float16", "f32", "float32"]
Layout = Literal["nchw", "nhwc"]
Color = Literal["rgb", "bgr", "gray"]
Resize = Literal["stretch", "letterbox", "letterbox_top_left", "crop_center", "none"]
Interp = Literal["bilinear", "nearest"]
Normalize = Union[
    Literal["none", "unit", "imagenet", "symmetric"],
    Mapping[str, Any],
]
Roi = Union[
    Tuple[float, float, float, float],
    Sequence[Sequence[float]],
    Sequence[float],
    Mapping[str, Any],
]

def version() -> str: ...
def capabilities() -> dict: ...

class Tensor:
    @property
    def shape(self) -> list[int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def data_ptr(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __dlpack__(self, *, stream: Any = None, max_version: Any = None, dl_device: Any = None, copy: Optional[bool] = None) -> Any: ...
    def __dlpack_device__(self) -> Tuple[int, int]: ...
    @property
    def __array_interface__(self) -> dict: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...
    def numpy(self) -> Any: ...

class Frame:
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def pts(self) -> int: ...

class Item:
    stream_id: str
    valid: bool
    seq: int
    pts: int
    keyframe: bool
    src_width: int
    src_height: int
    transform: Tuple[float, float, float, float, float, float]
    capture_time_ns: Optional[int]
    recv_time_ns: int
    ready_time_ns: int
    latency_ms: float
    dropped_since_last: int
    frame: Optional[Frame]
    def to_source(self, x: float, y: float) -> Tuple[float, float]: ...
    def box_to_source(self, box: Sequence[float]) -> list[float]: ...
    def boxes_to_source(self, boxes: Sequence[Sequence[float]]) -> list[list[float]]: ...

class Batch:
    tensor: Tensor
    output: str
    seq: int
    items: list[Item]
    valid_count: int
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Item: ...
    def __dlpack__(self, *, stream: Any = None, max_version: Any = None, dl_device: Any = None, copy: Optional[bool] = None) -> Any: ...
    def __dlpack_device__(self) -> Tuple[int, int]: ...
    def numpy(self) -> Any: ...

class BatchIterator(Iterator[Batch]):
    def __iter__(self) -> "BatchIterator": ...
    def __next__(self) -> Batch: ...

class Engine:
    def __init__(self, config: Union[None, str, Mapping[str, Any]] = None, device: Optional[str] = None) -> None: ...
    @property
    def device(self) -> str: ...
    @property
    def outputs(self) -> list[str]: ...
    @property
    def streams(self) -> list[str]: ...
    def add_output(
        self,
        name: str,
        width: int = 640,
        height: int = 640,
        *,
        dtype: DType = "f32",
        layout: Layout = "nchw",
        color: Color = "rgb",
        resize: Resize = "letterbox",
        interp: Interp = "bilinear",
        normalize: Optional[Normalize] = None,
        pad: Union[None, float, Sequence[float]] = None,
        batch: int = 1,
        max_wait_ms: int = 0,
        pad_to_max: bool = False,
        keep_frame: bool = False,
        max_fps: Optional[float] = None,
        backpressure: Literal["drop_oldest", "latest_only", "block"] = "drop_oldest",
        queue_depth: int = 4,
        device: Optional[str] = None,
    ) -> None: ...
    def add_stream(
        self,
        id: str,
        url: str,
        outputs: Sequence[str],
        *,
        gating: Union[None, Literal["always", "keyframes", "auto"], Mapping[str, Any]] = None,
        transport: Literal["tcp", "udp"] = "tcp",
        decoder: Optional[Literal["auto", "nvdec", "cpu"]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None: ...
    def remove_stream(self, id: str) -> None: ...
    def force_active(self, id: str, seconds: float) -> None: ...
    def recv(self, output: str, timeout: Optional[float] = None) -> Batch: ...
    def try_recv(self, output: str) -> Optional[Batch]: ...
    def batches(self, output: str) -> BatchIterator: ...
    def crop(
        self,
        frame: Frame,
        rois: Sequence[Roi],
        width: int = 112,
        height: int = 112,
        *,
        dtype: DType = "f32",
        layout: Layout = "nchw",
        color: Color = "rgb",
        resize: Resize = "stretch",
        interp: Interp = "bilinear",
        normalize: Optional[Normalize] = None,
        pad: Union[None, float, Sequence[float]] = None,
        device: Optional[str] = None,
    ) -> Tensor: ...
    def stats(self) -> dict: ...
    def prometheus(self) -> str: ...
    def config(self) -> dict: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Engine": ...
    def __exit__(self, *args: Any) -> None: ...

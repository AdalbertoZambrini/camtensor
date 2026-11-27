//! Tensors and batches handed to consumers.

use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;

use camtensor_core::{DType, Device, DeviceMemory, Frame, FrameMeta, TensorDesc};

/// Alignment of host tensor allocations (cache line / AVX-512 friendly).
pub const HOST_ALIGN: usize = 64;

/// Owned, zero-initialised, 64-byte aligned host allocation.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: AlignedBuf exclusively owns its allocation; it is plain bytes.
unsafe impl Send for AlignedBuf {}
// SAFETY: shared access only hands out `&[u8]`; mutation goes through raw
// pointers whose disjointness is guaranteed by the batcher (one slot per writer).
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Allocates `len` zeroed bytes.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        let layout = AllocLayout::from_size_align(len, HOST_ALIGN).expect("tensor too large");
        // SAFETY: layout has non-zero size.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Start address.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Immutable view.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes (or dangling with len 0).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = AllocLayout::from_size_align(self.len, HOST_ALIGN).expect("layout");
            // SAFETY: allocated in `zeroed` with this exact layout.
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuf").field("len", &self.len).finish()
    }
}

/// Memory backing a tensor.
#[derive(Debug)]
pub enum TensorStorage {
    /// Host memory.
    Host(AlignedBuf),
    /// Device memory (CUDA).
    Device(Box<dyn DeviceMemory>),
}

impl TensorStorage {
    /// Device of the storage.
    pub fn device(&self) -> Device {
        match self {
            TensorStorage::Host(_) => Device::Cpu,
            TensorStorage::Device(d) => Device::Cuda(d.device_ordinal()),
        }
    }

    /// Base address (host pointer or device pointer).
    pub fn addr(&self) -> u64 {
        match self {
            TensorStorage::Host(b) => b.as_ptr() as u64,
            TensorStorage::Device(d) => d.device_ptr(),
        }
    }

    /// Size in bytes.
    pub fn byte_len(&self) -> usize {
        match self {
            TensorStorage::Host(b) => b.len(),
            TensorStorage::Device(d) => d.byte_len(),
        }
    }
}

/// A dense tensor (host or device), reference counted.
///
/// Holding a `Tensor` (or any clone) keeps its memory alive; the batch
/// buffer it belongs to is recycled only once every clone is dropped.
#[derive(Clone)]
pub struct Tensor {
    desc: TensorDesc,
    storage: Arc<TensorStorage>,
    offset: usize,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("device", &self.desc.device)
            .field("dtype", &self.desc.dtype)
            .field("shape", &self.desc.shape)
            .finish()
    }
}

impl Tensor {
    pub(crate) fn new(desc: TensorDesc, storage: Arc<TensorStorage>, offset: usize) -> Self {
        debug_assert!(offset + desc.byte_len() <= storage.byte_len());
        Self { desc, storage, offset }
    }

    /// Shape, strides, dtype and device.
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// Shape (`[N, C, H, W]` or `[N, H, W, C]`).
    pub fn shape(&self) -> [usize; 4] {
        self.desc.shape
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.desc.dtype
    }

    /// Device.
    pub fn device(&self) -> Device {
        self.desc.device
    }

    /// Address of the first element (host address or CUDA device pointer).
    pub fn data_ptr(&self) -> u64 {
        self.storage.addr() + self.offset as u64
    }

    /// Size in bytes.
    pub fn byte_len(&self) -> usize {
        self.desc.byte_len()
    }

    /// Raw bytes for host tensors, `None` for device tensors.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &*self.storage {
            TensorStorage::Host(b) => {
                Some(&b.as_slice()[self.offset..self.offset + self.byte_len()])
            }
            TensorStorage::Device(_) => None,
        }
    }

    /// Typed view of a host `f32` tensor.
    pub fn as_f32(&self) -> Option<&[f32]> {
        self.typed(DType::F32)
    }

    /// Typed view of a host `u8` tensor.
    pub fn as_u8(&self) -> Option<&[u8]> {
        if self.desc.dtype != DType::U8 {
            return None;
        }
        self.as_bytes()
    }

    /// Typed view of a host `f16` tensor as raw `u16` bit patterns.
    pub fn as_f16_bits(&self) -> Option<&[u16]> {
        self.typed(DType::F16)
    }

    fn typed<T>(&self, want: DType) -> Option<&[T]> {
        if self.desc.dtype != want {
            return None;
        }
        let bytes = self.as_bytes()?;
        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return None;
        }
        // SAFETY: alignment checked above, length is a multiple of the
        // element size by construction, and all bit patterns are valid for
        // f32/u16.
        Some(unsafe { std::slice::from_raw_parts(ptr as *const T, self.desc.numel()) })
    }

    /// Sub-tensor with the first dimension restricted to `start..start+len`.
    pub fn narrow(&self, start: usize, len: usize) -> Option<Tensor> {
        if start + len > self.desc.shape[0] {
            return None;
        }
        let mut desc = self.desc;
        desc.shape[0] = len;
        let offset = self.offset + start * self.desc.strides[0] * self.desc.dtype.size();
        Some(Tensor { desc, storage: self.storage.clone(), offset })
    }

    /// Keeps the underlying storage alive (used by FFI exports).
    pub fn storage(&self) -> &Arc<TensorStorage> {
        &self.storage
    }
}

/// One image slot of a [`Batch`].
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Metadata (stream, timestamps, transform to map detections back).
    pub meta: FrameMeta,
    /// `false` for padding slots (`batch.pad_to_max`) or failed slots.
    pub valid: bool,
    /// Full-resolution decoded frame, when the output has `keep_frame`.
    pub frame: Option<Frame>,
}

/// A batch of preprocessed images from one output.
///
/// Dropping the batch (and every [`Tensor`] cloned from it) returns its
/// buffer to the output's pool.
#[derive(Debug, Clone)]
pub struct Batch {
    pub(crate) output: Arc<str>,
    pub(crate) tensor: Tensor,
    pub(crate) items: Vec<BatchItem>,
    pub(crate) seq: u64,
}

impl Batch {
    /// Output name.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// The batch tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Per-slot information, `len()` entries.
    pub fn items(&self) -> &[BatchItem] {
        &self.items
    }

    /// Number of slots (first tensor dimension).
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` if the batch has no slots.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Number of valid (non-padding) slots.
    pub fn valid_count(&self) -> usize {
        self.items.iter().filter(|i| i.valid).count()
    }

    /// Monotonic batch counter of this output.
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

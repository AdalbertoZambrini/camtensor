//! Fused GPU preprocessing: sampling (bilinear / nearest through a 2x3
//! sampling matrix), Y'CbCr → RGB, channel order, normalisation and layout,
//! in one kernel launch for a whole batch of output images.
//!
//! # Semantics (shared with the CPU implementation in `camtensor-preprocess`)
//!
//! For destination pixel indices `(x, y)` and sampling matrix
//! `[a, b, c, d, e, f]` (see [`Affine2::sampling_matrix`]):
//! `sx = a*x + b*y + c`, `sy = d*x + e*y + f` (source pixel indices).
//!
//! * If `sx < -0.5 || sx > W - 0.5 || sy < -0.5 || sy > H - 0.5` the pixel
//!   takes `spec.pad` (pixel units, output channel order), then is
//!   normalised like any other pixel.
//! * Bilinear: `x0 = floor(sx)`, `fx = sx - x0`, taps `(x0, y0)`,
//!   `(x0+1, y0)`, `(x0, y0+1)`, `(x0+1, y0+1)` clamped to the plane, weights
//!   `(1-fx)(1-fy)`, `fx(1-fy)`, `(1-fx)fy`, `fx*fy`. Nearest:
//!   `clamp(floor(sx + 0.5))`.
//! * Y'CbCr sources: luma sampled at `(sx, sy)`, chroma at
//!   `((sx + 0.5) / 2^shift_x - 0.5, (sy + 0.5) / 2^shift_y - 0.5)` on the
//!   chroma plane (`ceil(W / 2^shift_x) x ceil(H / 2^shift_y)`); interpolated
//!   in YUV, then converted (limited: `Y' = (Y-16)*255/219`,
//!   `C = (U-128)*255/224`; full: `Y' = Y`, `C = U-128`) with BT.601 or BT.709
//!   and clamped to `[0, 255]`. Gray output from YUV is `clamp(Y')`.
//! * Gray8 sources replicate the value; RGB/BGR sources interpolate per
//!   channel, gray output is `0.299 R + 0.587 G + 0.114 B`.
//! * `out = v * A[c] + B[c]` with `(A, B) = spec.normalize.affine_coeffs()`
//!   for f32/f16 (f16 rounds to nearest even); u8 stores
//!   `clamp(floor(v + 0.5), 0, 255)` without normalisation.
//! * NCHW element offset `c*H*W + y*W + x`, NHWC `(y*W + x)*C + c`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use camtensor_core::{
    Affine2, ColorMatrix, ColorOrder, ColorRange, DType, DeviceMemory, Error, Frame, FrameData,
    FrameDesc, Interp, Layout, PixelFormat, Result, TensorSpec,
};

use crate::device::CudaDevice;
use crate::memory::DeviceBuffer;
use crate::module::{Function, LaunchConfig, Module};
use crate::stream::CudaStream;

/// PTX generated from `kernels/preprocess.cu` (see `cargo run -p xtask -- ptx`).
pub const PREPROCESS_PTX: &str = include_str!("../kernels/preprocess.ptx");

/// Maximum number of items per kernel launch; larger batches are split.
pub const MAX_ITEMS_PER_LAUNCH: usize = 64;

const BLOCK: (u32, u32) = (32, 8);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct KernelItem {
    m: [f32; 6],
    dst: u64,
}

/// Mirror of `CtParams` in `preprocess.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct KernelParams {
    plane: [u64; 3],
    pitch: [i32; 3],
    src_w: i32,
    src_h: i32,
    chroma_w: i32,
    chroma_h: i32,
    chroma_sx: f32,
    chroma_sy: f32,
    src_kind: i32,
    y_off: f32,
    y_scale: f32,
    c_scale: f32,
    cr_r: f32,
    cb_g: f32,
    cr_g: f32,
    cb_b: f32,
    dst_w: i32,
    dst_h: i32,
    out_kind: i32,
    dtype: i32,
    nhwc: i32,
    bilinear: i32,
    norm_a: [f32; 3],
    norm_b: [f32; 3],
    pad: [f32; 3],
    n_items: i32,
    reserved: i32,
    items: [KernelItem; MAX_ITEMS_PER_LAUNCH],
}

/// A source image in device memory: plane base pointers plus layout.
#[derive(Debug, Clone, Copy)]
pub struct SourceImage {
    planes: [u64; 3],
    desc: FrameDesc,
    ordinal: u32,
}

impl SourceImage {
    /// Source backed by a device frame (`FrameData::Device`).
    pub fn from_frame(frame: &Frame) -> Result<Self> {
        match &frame.data {
            FrameData::Device(mem) => Self::from_memory(mem.as_ref(), frame.desc),
            FrameData::Host(_) => Err(Error::Preprocess(
                "CUDA preprocessing needs a device frame; upload host frames first".into(),
            )),
        }
    }

    /// Source backed by a [`DeviceBuffer`] laid out as `desc`.
    pub fn from_buffer(buf: &DeviceBuffer, desc: FrameDesc) -> Result<Self> {
        Self::from_memory(buf, desc)
    }

    /// Source backed by any [`DeviceMemory`] laid out as `desc`
    /// (bounds-checked against `mem.byte_len()`).
    pub fn from_memory(mem: &dyn DeviceMemory, desc: FrameDesc) -> Result<Self> {
        validate_desc(&desc, mem.byte_len())?;
        let base = mem.device_ptr();
        let mut planes = [0u64; 3];
        for (i, p) in planes.iter_mut().enumerate().take(desc.format.plane_count()) {
            *p = base + desc.planes[i].offset as u64;
        }
        Ok(Self { planes, desc, ordinal: mem.device_ordinal() })
    }

    /// Source from raw plane pointers (e.g. a mapped NVDEC surface or a
    /// foreign allocation). `desc.planes[i].offset` is ignored.
    ///
    /// # Safety
    /// Each plane must be readable device memory on device `ordinal`
    /// covering `pitch * rows` bytes for as long as kernels using this
    /// source may run.
    pub unsafe fn from_raw(planes: [u64; 3], desc: FrameDesc, ordinal: u32) -> Self {
        Self { planes, desc, ordinal }
    }

    /// Layout of the source.
    pub fn desc(&self) -> &FrameDesc {
        &self.desc
    }
}

/// One output image of a batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleItem {
    /// Sampling matrix (destination index → source index), see
    /// [`Affine2::sampling_matrix`].
    pub sampling: [f32; 6],
    /// Byte address of this image's slot in the output tensor.
    pub dst_ptr: u64,
}

/// The loaded preprocessing kernel for one device.
pub struct Preprocessor {
    module: Module,
    func: Function,
    device: Arc<CudaDevice>,
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preprocessor").field("device", &self.device.ordinal()).finish()
    }
}

static PREPROCESSORS: OnceLock<Mutex<HashMap<u32, Arc<Preprocessor>>>> = OnceLock::new();

impl Preprocessor {
    /// Returns the kernel for `device`, JIT-compiling the PTX on first use
    /// (cached per device for the lifetime of the process).
    pub fn get(device: &Arc<CudaDevice>) -> Result<Arc<Preprocessor>> {
        let cache = PREPROCESSORS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = map.get(&device.ordinal()) {
            return Ok(p.clone());
        }
        let module = Module::load_ptx(device, PREPROCESS_PTX)?;
        let func = module.function("ct_preprocess")?;
        let p = Arc::new(Preprocessor { module, func, device: device.clone() });
        map.insert(device.ordinal(), p.clone());
        Ok(p)
    }

    /// Device the kernel is loaded on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Preprocesses `sampling.len()` images from `src` into consecutive
    /// slots of `dst` starting at byte `dst_offset` (slot size
    /// `spec.image_bytes()`), asynchronously on `stream`.
    pub fn preprocess_into(
        &self,
        stream: &CudaStream,
        src: &SourceImage,
        sampling: &[Affine2],
        dst: &DeviceBuffer,
        dst_offset: usize,
        spec: &TensorSpec,
    ) -> Result<()> {
        let slot = spec.image_bytes();
        let needed = sampling
            .len()
            .checked_mul(slot)
            .and_then(|n| n.checked_add(dst_offset))
            .ok_or_else(|| Error::Preprocess("output size overflow".into()))?;
        if needed > dst.byte_len() {
            return Err(Error::Preprocess(format!(
                "output buffer too small: need {needed} bytes, have {}",
                dst.byte_len()
            )));
        }
        if dst.device().ordinal() != self.device.ordinal() {
            return Err(Error::Preprocess("output buffer is on another device".into()));
        }
        let items: Vec<SampleItem> = sampling
            .iter()
            .enumerate()
            .map(|(i, m)| SampleItem {
                sampling: m.0,
                dst_ptr: dst.device_ptr() + (dst_offset + i * slot) as u64,
            })
            .collect();
        // SAFETY: every destination slot lies inside `dst` (checked above)
        // and `src` was bounds-checked at construction.
        unsafe { self.preprocess(stream, src, &items, spec) }
    }

    /// Launches the kernel for `items` (split into launches of at most
    /// [`MAX_ITEMS_PER_LAUNCH`]).
    ///
    /// # Safety
    /// Each `dst_ptr` must address `spec.image_bytes()` writable bytes of
    /// device memory on this device, not overlapping each other, valid until
    /// the work on `stream` has completed; the source memory must likewise
    /// stay valid.
    pub unsafe fn preprocess(
        &self,
        stream: &CudaStream,
        src: &SourceImage,
        items: &[SampleItem],
        spec: &TensorSpec,
    ) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        spec.validate()?;
        if src.ordinal != self.device.ordinal()
            || stream.device().ordinal() != self.device.ordinal()
        {
            return Err(Error::Preprocess("source, stream and kernel must share a device".into()));
        }
        let mut params = build_params(src, spec)?;
        let grid_x = spec.width.div_ceil(BLOCK.0);
        let grid_y = spec.height.div_ceil(BLOCK.1);
        for chunk in items.chunks(MAX_ITEMS_PER_LAUNCH) {
            params.n_items = chunk.len() as i32;
            for (slot, it) in params.items.iter_mut().zip(chunk) {
                *slot = KernelItem { m: it.sampling, dst: it.dst_ptr };
            }
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, chunk.len() as u32),
                block: (BLOCK.0, BLOCK.1, 1),
                shared_mem: 0,
            };
            let mut args = [&mut params as *mut KernelParams as *mut c_void];
            // SAFETY: the kernel takes exactly one `CtParams` by value, which
            // `KernelParams` mirrors (layout asserted in tests); memory
            // validity is the caller's contract.
            unsafe { self.module.launch(self.func, cfg, stream, &mut args)? };
        }
        Ok(())
    }
}

fn validate_desc(desc: &FrameDesc, len: usize) -> Result<()> {
    let (w, h) = (desc.width as usize, desc.height as usize);
    if w == 0 || h == 0 {
        return Err(Error::Preprocess("source has zero size".into()));
    }
    let (sx, sy) = desc.format.chroma_shift();
    let (cw, ch) = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
    let planes: &[(usize, usize)] = match desc.format {
        PixelFormat::Nv12 => &[(w, h), (cw * 2, ch)],
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            &[(w, h), (cw, ch), (cw, ch)]
        }
        PixelFormat::Gray8 => &[(w, h)],
        PixelFormat::Rgb24 | PixelFormat::Bgr24 => &[(w * 3, h)],
    };
    for (i, &(row, rows)) in planes.iter().enumerate() {
        let p = desc.planes[i];
        if p.pitch < row || p.pitch > i32::MAX as usize {
            return Err(Error::Preprocess(format!("plane {i}: invalid pitch {}", p.pitch)));
        }
        let end = p.offset + p.pitch * (rows - 1) + row;
        if end > len {
            return Err(Error::Preprocess(format!(
                "plane {i} ends at byte {end} but the buffer has {len} bytes"
            )));
        }
    }
    Ok(())
}

fn build_params(src: &SourceImage, spec: &TensorSpec) -> Result<KernelParams> {
    let d = &src.desc;
    if d.width > i32::MAX as u32 || d.height > i32::MAX as u32 {
        return Err(Error::Preprocess("source too large".into()));
    }
    let (shx, shy) = d.format.chroma_shift();
    let cw = (d.width + (1 << shx) - 1) >> shx;
    let ch = (d.height + (1 << shy) - 1) >> shy;
    let src_kind = match d.format {
        PixelFormat::Nv12 => 0,
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => 1,
        PixelFormat::Gray8 => 2,
        PixelFormat::Rgb24 => 3,
        PixelFormat::Bgr24 => 4,
    };
    let (y_off, y_scale, c_scale) = match d.color.range {
        ColorRange::Limited => (16.0f32, 255.0f32 / 219.0, 255.0f32 / 224.0),
        ColorRange::Full => (0.0, 1.0, 1.0),
    };
    let (cr_r, cb_g, cr_g, cb_b) = match d.color.matrix {
        ColorMatrix::Bt601 => (1.402f32, 0.344136f32, 0.714136f32, 1.772f32),
        ColorMatrix::Bt709 => (1.5748, 0.187324, 0.468124, 1.8556),
    };
    let (norm_a, norm_b) = spec.normalize.affine_coeffs();
    let mut pitch = [0i32; 3];
    for (i, p) in pitch.iter_mut().enumerate().take(d.format.plane_count()) {
        *p = i32::try_from(d.planes[i].pitch)
            .map_err(|_| Error::Preprocess("pitch too large".into()))?;
    }
    Ok(KernelParams {
        plane: src.planes,
        pitch,
        src_w: d.width as i32,
        src_h: d.height as i32,
        chroma_w: cw as i32,
        chroma_h: ch as i32,
        chroma_sx: 1.0 / (1u32 << shx) as f32,
        chroma_sy: 1.0 / (1u32 << shy) as f32,
        src_kind,
        y_off,
        y_scale,
        c_scale,
        cr_r,
        cb_g,
        cr_g,
        cb_b,
        dst_w: spec.width as i32,
        dst_h: spec.height as i32,
        out_kind: match spec.color {
            ColorOrder::Rgb => 0,
            ColorOrder::Bgr => 1,
            ColorOrder::Gray => 2,
        },
        dtype: match spec.dtype {
            DType::U8 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
        },
        nhwc: (spec.layout == Layout::Nhwc) as i32,
        bilinear: (spec.interp == Interp::Bilinear) as i32,
        norm_a,
        norm_b,
        pad: spec.pad,
        n_items: 0,
        reserved: 0,
        items: [KernelItem::default(); MAX_ITEMS_PER_LAUNCH],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Offsets of `CtParams` in preprocess.cu (also visible in the PTX as
    /// `ld.param` displacements).
    #[test]
    fn kernel_params_layout() {
        assert_eq!(size_of::<KernelItem>(), 32);
        assert_eq!(offset_of!(KernelParams, pitch), 24);
        assert_eq!(offset_of!(KernelParams, src_w), 36);
        assert_eq!(offset_of!(KernelParams, chroma_w), 44);
        assert_eq!(offset_of!(KernelParams, chroma_sx), 52);
        assert_eq!(offset_of!(KernelParams, src_kind), 60);
        assert_eq!(offset_of!(KernelParams, y_off), 64);
        assert_eq!(offset_of!(KernelParams, cr_r), 76);
        assert_eq!(offset_of!(KernelParams, dst_w), 92);
        assert_eq!(offset_of!(KernelParams, out_kind), 100);
        assert_eq!(offset_of!(KernelParams, bilinear), 112);
        assert_eq!(offset_of!(KernelParams, norm_a), 116);
        assert_eq!(offset_of!(KernelParams, norm_b), 128);
        assert_eq!(offset_of!(KernelParams, pad), 140);
        assert_eq!(offset_of!(KernelParams, n_items), 152);
        assert_eq!(offset_of!(KernelParams, items), 160);
        assert_eq!(size_of::<KernelParams>(), 160 + 32 * MAX_ITEMS_PER_LAUNCH);
        // CUDA kernel parameters are limited to 4 KiB for compute_52 PTX.
        const { assert!(size_of::<KernelParams>() <= 4096) };
    }

    #[test]
    fn ptx_is_embedded() {
        assert!(PREPROCESS_PTX.contains(".entry ct_preprocess"));
        assert!(PREPROCESS_PTX.contains(".target sm_52"));
    }

    #[test]
    fn desc_validation() {
        let c = camtensor_core::ColorSpec::default();
        let d = FrameDesc::packed(1920, 1080, PixelFormat::Nv12, c);
        assert!(validate_desc(&d, d.byte_len()).is_ok());
        assert!(validate_desc(&d, d.byte_len() - 1).is_err());
        let mut bad = d;
        bad.planes[0].pitch = 100;
        assert!(validate_desc(&bad, d.byte_len()).is_err());
    }
}

//! Output tensor specification: what a consumer asks for and receives.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::Error;

/// Element type of an output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// Raw 8-bit pixels (0..=255). Normalisation is not applied.
    U8,
    /// IEEE half precision.
    F16,
    /// IEEE single precision.
    #[default]
    F32,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            DType::U8 => 1,
            DType::F16 => 2,
            DType::F32 => 4,
        }
    }
}

/// Memory layout of a batch tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// `[N, C, H, W]` (PyTorch, ONNX, TensorRT default).
    #[default]
    Nchw,
    /// `[N, H, W, C]` (TensorFlow, OpenVINO, raw images).
    Nhwc,
}

/// Channel order of the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorOrder {
    /// Red, green, blue.
    #[default]
    Rgb,
    /// Blue, green, red (OpenCV convention).
    Bgr,
    /// Single luma channel.
    Gray,
}

impl ColorOrder {
    /// Number of channels.
    pub fn channels(self) -> usize {
        match self {
            ColorOrder::Gray => 1,
            _ => 3,
        }
    }
}

/// How the source frame is mapped onto the output size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeMode {
    /// Scale each axis independently to fill the output (aspect not kept).
    Stretch,
    /// Keep aspect ratio, fit inside the output, pad the borders, centred
    /// (YOLO-style letterbox).
    #[default]
    Letterbox,
    /// Like [`ResizeMode::Letterbox`] but the image is anchored top-left and
    /// padding is added only to the right/bottom.
    LetterboxTopLeft,
    /// Keep aspect ratio, cover the output, crop the excess (centred).
    CropCenter,
    /// No scaling: the source is copied 1:1 from the top-left corner,
    /// cropped or padded to the output size.
    None,
}

/// Sampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interp {
    /// Nearest neighbour.
    Nearest,
    /// Bilinear (pixel-centre aligned, same convention as OpenCV `INTER_LINEAR`).
    #[default]
    Bilinear,
}

/// Per-channel normalisation: `out = (pixel * scale - mean[c]) / std[c]`,
/// with `pixel` in `0..=255` and channels in *output* order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NormalizeRepr", into = "NormalizeRepr")]
pub struct Normalize {
    /// Multiplier applied to the raw pixel value.
    pub scale: f32,
    /// Value subtracted after scaling, per output channel.
    pub mean: [f32; 3],
    /// Divisor applied last, per output channel.
    pub std: [f32; 3],
}

impl Default for Normalize {
    fn default() -> Self {
        Self::UNIT
    }
}

impl Normalize {
    /// Raw pixel values as floats (`0..=255`).
    pub const NONE: Self = Self { scale: 1.0, mean: [0.0; 3], std: [1.0; 3] };
    /// Pixels scaled to `0..=1` (YOLO family).
    pub const UNIT: Self = Self { scale: 1.0 / 255.0, mean: [0.0; 3], std: [1.0; 3] };
    /// ImageNet mean/std on `0..=1` RGB values.
    pub const IMAGENET: Self =
        Self { scale: 1.0 / 255.0, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
    /// `(pixel - 127.5) / 127.5`, i.e. `-1..=1`.
    pub const SYMMETRIC: Self = Self { scale: 1.0 / 127.5, mean: [1.0; 3], std: [1.0; 3] };

    /// Looks up a named preset (`none`, `unit`, `imagenet`, `symmetric`).
    pub fn preset(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "none" | "raw" => Self::NONE,
            "unit" | "0-1" | "yolo" => Self::UNIT,
            "imagenet" => Self::IMAGENET,
            "symmetric" | "arcface" | "-1-1" => Self::SYMMETRIC,
            _ => return None,
        })
    }

    /// Precomputes `(a, b)` per channel so that `out = pixel * a[c] + b[c]`.
    pub fn affine_coeffs(&self) -> ([f32; 3], [f32; 3]) {
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        for c in 0..3 {
            a[c] = self.scale / self.std[c];
            b[c] = -self.mean[c] / self.std[c];
        }
        (a, b)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum NormalizeRepr {
    Preset(String),
    Custom {
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        mean: [f32; 3],
        #[serde(default = "ones")]
        std: [f32; 3],
    },
}

fn one() -> f32 {
    1.0
}
fn ones() -> [f32; 3] {
    [1.0; 3]
}

impl TryFrom<NormalizeRepr> for Normalize {
    type Error = String;
    fn try_from(r: NormalizeRepr) -> Result<Self, String> {
        match r {
            NormalizeRepr::Preset(p) => {
                Normalize::preset(&p).ok_or_else(|| format!("unknown normalize preset '{p}'"))
            }
            NormalizeRepr::Custom { scale, mean, std } => {
                if std.contains(&0.0) {
                    return Err("normalize.std must not contain zeros".into());
                }
                Ok(Normalize { scale, mean, std })
            }
        }
    }
}

impl From<Normalize> for NormalizeRepr {
    fn from(n: Normalize) -> Self {
        NormalizeRepr::Custom { scale: n.scale, mean: n.mean, std: n.std }
    }
}

/// Specification of one preprocessed image inside an output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TensorSpec {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Element type.
    pub dtype: DType,
    /// Memory layout.
    pub layout: Layout,
    /// Channel order.
    pub color: ColorOrder,
    /// Geometry mapping.
    pub resize: ResizeMode,
    /// Sampling filter.
    pub interp: Interp,
    /// Normalisation (ignored for [`DType::U8`]).
    pub normalize: Normalize,
    /// Padding colour in pixel units (`0..=255`), output channel order.
    pub pad: [f32; 3],
}

impl Default for TensorSpec {
    fn default() -> Self {
        Self {
            width: 640,
            height: 640,
            dtype: DType::F32,
            layout: Layout::Nchw,
            color: ColorOrder::Rgb,
            resize: ResizeMode::Letterbox,
            interp: Interp::Bilinear,
            normalize: Normalize::UNIT,
            pad: [114.0; 3],
        }
    }
}

impl TensorSpec {
    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.color.channels()
    }

    /// Bytes taken by one image.
    pub fn image_bytes(&self) -> usize {
        self.width as usize * self.height as usize * self.channels() * self.dtype.size()
    }

    /// Shape of a batch of `n` images in the configured layout.
    pub fn batch_shape(&self, n: usize) -> [usize; 4] {
        let (c, h, w) = (self.channels(), self.height as usize, self.width as usize);
        match self.layout {
            Layout::Nchw => [n, c, h, w],
            Layout::Nhwc => [n, h, w, c],
        }
    }

    /// Validates the specification.
    pub fn validate(&self) -> crate::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::config("tensor width/height must be > 0"));
        }
        if self.width > 16384 || self.height > 16384 {
            return Err(Error::config("tensor width/height must be <= 16384"));
        }
        Ok(())
    }
}

/// Where a tensor lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Pick CUDA device 0 when NVDEC is usable, CPU otherwise.
    #[default]
    Auto,
    /// Host memory.
    Cpu,
    /// CUDA device memory on the given ordinal.
    Cuda(u32),
}

impl Device {
    /// `true` for [`Device::Cuda`].
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Auto => f.write_str("auto"),
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

impl FromStr for Device {
    type Err = Error;
    fn from_str(s: &str) -> crate::Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => Ok(Device::Cuda(0)),
            _ => {
                let idx = s
                    .strip_prefix("cuda:")
                    .or_else(|| s.strip_prefix("gpu:"))
                    .and_then(|i| i.parse().ok())
                    .ok_or_else(|| Error::config(format!("invalid device '{s}'")))?;
                Ok(Device::Cuda(idx))
            }
        }
    }
}

impl Serialize for Device {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Runtime description of a tensor handed to a consumer.
///
/// `strides` are in **elements** (not bytes), like PyTorch/DLPack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDesc {
    /// Device holding the data.
    pub device: Device,
    /// Element type.
    pub dtype: DType,
    /// Shape (always 4-D: batch first).
    pub shape: [usize; 4],
    /// Strides in elements.
    pub strides: [usize; 4],
}

impl TensorDesc {
    /// Contiguous (row-major) descriptor for `shape`.
    pub fn contiguous(device: Device, dtype: DType, shape: [usize; 4]) -> Self {
        let strides = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1];
        Self { device, dtype, shape, strides }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total number of bytes for a contiguous tensor.
    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parse() {
        assert_eq!("auto".parse::<Device>().unwrap(), Device::Auto);
        assert_eq!("CPU".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:3".parse::<Device>().unwrap(), Device::Cuda(3));
        assert!("tpu".parse::<Device>().is_err());
    }

    #[test]
    fn normalize_presets_and_coeffs() {
        let n = Normalize::SYMMETRIC;
        let (a, b) = n.affine_coeffs();
        assert!((0.0 * a[0] + b[0] + 1.0).abs() < 1e-6);
        assert!((255.0 * a[0] + b[0] - 1.0).abs() < 1e-6);
        assert_eq!(Normalize::preset("ImageNet"), Some(Normalize::IMAGENET));
    }

    #[test]
    fn spec_yaml_roundtrip() {
        let spec: TensorSpec = serde_norway::from_str(
            "width: 112\nheight: 112\ndtype: f16\nnormalize: symmetric\nresize: stretch\n",
        )
        .unwrap();
        assert_eq!(spec.width, 112);
        assert_eq!(spec.dtype, DType::F16);
        assert_eq!(spec.normalize, Normalize::SYMMETRIC);
        assert_eq!(spec.batch_shape(4), [4, 3, 112, 112]);
        let custom: TensorSpec =
            serde_norway::from_str("normalize: {scale: 0.5, mean: [1, 2, 3]}").unwrap();
        assert_eq!(custom.normalize.mean, [1.0, 2.0, 3.0]);
        assert_eq!(custom.normalize.std, [1.0; 3]);
    }

    #[test]
    fn contiguous_strides() {
        let d = TensorDesc::contiguous(Device::Cpu, DType::F16, [2, 3, 4, 5]);
        assert_eq!(d.strides, [60, 20, 5, 1]);
        assert_eq!(d.byte_len(), 240);
    }
}

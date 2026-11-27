//! 2-D geometry: affine transforms, resize mappings, regions of interest and
//! face-alignment helpers.
//!
//! # Coordinate conventions
//!
//! Internally camtensor uses **continuous** coordinates: pixel `i` covers the
//! interval `[i, i + 1)` and its centre is at `i + 0.5`. This is the
//! convention under which scaling is exact (`cv2.resize`, PyTorch
//! `align_corners=False`). Bounding boxes reported by detectors are in this
//! convention too.
//!
//! OpenCV's `warpAffine` matrices use **index** coordinates (pixel centre at
//! `i`). Use [`Affine2::from_opencv`] / [`Affine2::to_opencv`] to convert.

use serde::{Deserialize, Serialize};

use crate::tensor::ResizeMode;

/// 2x3 affine transform `[a, b, c, d, e, f]`:
/// `x' = a*x + b*y + c`, `y' = d*x + e*y + f`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Affine2(pub [f32; 6]);

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    /// Identity transform.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    /// Axis-aligned scale followed by translation.
    pub fn scale_translate(sx: f32, sy: f32, tx: f32, ty: f32) -> Self {
        Self([sx, 0.0, tx, 0.0, sy, ty])
    }

    /// Pure translation.
    pub fn translate(tx: f32, ty: f32) -> Self {
        Self::scale_translate(1.0, 1.0, tx, ty)
    }

    /// Applies the transform to a point.
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    /// `self ∘ other`: first apply `other`, then `self`.
    pub fn compose(&self, other: &Affine2) -> Affine2 {
        let a = &self.0;
        let b = &other.0;
        Affine2([
            a[0] * b[0] + a[1] * b[3],
            a[0] * b[1] + a[1] * b[4],
            a[0] * b[2] + a[1] * b[5] + a[2],
            a[3] * b[0] + a[4] * b[3],
            a[3] * b[1] + a[4] * b[4],
            a[3] * b[2] + a[4] * b[5] + a[5],
        ])
    }

    /// Inverse transform, `None` if singular.
    pub fn invert(&self) -> Option<Affine2> {
        let m = &self.0;
        let det = m[0] as f64 * m[4] as f64 - m[1] as f64 * m[3] as f64;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let (a, b, c, d, e, f) =
            (m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64, m[4] as f64, m[5] as f64);
        Some(Affine2([
            (e * inv) as f32,
            (-b * inv) as f32,
            ((b * f - c * e) * inv) as f32,
            (-d * inv) as f32,
            (a * inv) as f32,
            ((c * d - a * f) * inv) as f32,
        ]))
    }

    /// Converts an OpenCV `warpAffine` matrix (index coordinates,
    /// `dst = M * src`) to a continuous-coordinate transform.
    pub fn from_opencv(m: [f32; 6]) -> Affine2 {
        Affine2::translate(0.5, 0.5).compose(&Affine2(m)).compose(&Affine2::translate(-0.5, -0.5))
    }

    /// Inverse of [`Affine2::from_opencv`].
    pub fn to_opencv(&self) -> [f32; 6] {
        Affine2::translate(-0.5, -0.5).compose(self).compose(&Affine2::translate(0.5, 0.5)).0
    }

    /// Given a continuous `dst_from_src` transform, returns the transform a
    /// sampling kernel needs: destination pixel **index** → source pixel
    /// **index** (centre convention), i.e. `src_idx = S * dst_idx`.
    pub fn sampling_matrix(dst_from_src: &Affine2) -> Option<Affine2> {
        let src_from_dst = dst_from_src.invert()?;
        Some(
            Affine2::translate(-0.5, -0.5)
                .compose(&src_from_dst)
                .compose(&Affine2::translate(0.5, 0.5)),
        )
    }

    /// Continuous `dst_from_src` transform that maps a `src_w x src_h` region
    /// onto a `dst_w x dst_h` image according to `mode`.
    pub fn fit(src_w: f32, src_h: f32, dst_w: f32, dst_h: f32, mode: ResizeMode) -> Affine2 {
        if src_w <= 0.0 || src_h <= 0.0 {
            return Affine2::IDENTITY;
        }
        let (sx, sy) = (dst_w / src_w, dst_h / src_h);
        match mode {
            ResizeMode::Stretch => Affine2::scale_translate(sx, sy, 0.0, 0.0),
            ResizeMode::None => Affine2::IDENTITY,
            ResizeMode::Letterbox => {
                let s = sx.min(sy);
                let (tx, ty) = ((dst_w - src_w * s) * 0.5, (dst_h - src_h * s) * 0.5);
                Affine2::scale_translate(s, s, tx, ty)
            }
            ResizeMode::LetterboxTopLeft => {
                let s = sx.min(sy);
                Affine2::scale_translate(s, s, 0.0, 0.0)
            }
            ResizeMode::CropCenter => {
                let s = sx.max(sy);
                let (tx, ty) = ((dst_w - src_w * s) * 0.5, (dst_h - src_h * s) * 0.5);
                Affine2::scale_translate(s, s, tx, ty)
            }
        }
    }

    /// Least-squares similarity transform (rotation + uniform scale +
    /// translation, no reflection) mapping `src` points onto `dst` points
    /// (Umeyama, 1991). Returns `None` for degenerate input.
    pub fn similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Option<Affine2> {
        let n = src.len();
        if n < 2 || n != dst.len() {
            return None;
        }
        let nf = n as f64;
        let (mut msx, mut msy, mut mdx, mut mdy) = (0.0f64, 0.0, 0.0, 0.0);
        for (s, d) in src.iter().zip(dst) {
            msx += s[0] as f64;
            msy += s[1] as f64;
            mdx += d[0] as f64;
            mdy += d[1] as f64;
        }
        msx /= nf;
        msy /= nf;
        mdx /= nf;
        mdy /= nf;
        // For 2-D similarity the closed form reduces to:
        //   a = sum(xs*xd + ys*yd) / sum(xs^2 + ys^2)
        //   b = sum(xs*yd - ys*xd) / sum(xs^2 + ys^2)
        //   [x'] = [a -b] [x] + t
        //   [y']   [b  a] [y]
        let (mut num_a, mut num_b, mut den) = (0.0f64, 0.0, 0.0);
        for (s, d) in src.iter().zip(dst) {
            let (xs, ys) = (s[0] as f64 - msx, s[1] as f64 - msy);
            let (xd, yd) = (d[0] as f64 - mdx, d[1] as f64 - mdy);
            num_a += xs * xd + ys * yd;
            num_b += xs * yd - ys * xd;
            den += xs * xs + ys * ys;
        }
        if den < 1e-12 {
            return None;
        }
        let a = num_a / den;
        let b = num_b / den;
        let tx = mdx - (a * msx - b * msy);
        let ty = mdy - (b * msx + a * msy);
        Some(Affine2([a as f32, -b as f32, tx as f32, b as f32, a as f32, ty as f32]))
    }
}

/// Reference 5-point face template (left eye, right eye, nose tip, left
/// mouth corner, right mouth corner) for 112x112 ArcFace crops, in OpenCV
/// index coordinates, as published with InsightFace.
pub const ARCFACE_112_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// A region of a source frame to extract as a crop.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Roi {
    /// Axis-aligned box in source pixels (continuous coordinates). The box
    /// is mapped to the output with the output's [`ResizeMode`].
    Box {
        /// Left edge.
        x: f32,
        /// Top edge.
        y: f32,
        /// Width.
        w: f32,
        /// Height.
        h: f32,
    },
    /// Arbitrary affine transform in OpenCV `warpAffine` convention
    /// (`dst = M * src`, index coordinates).
    Affine {
        /// Row-major 2x3 matrix.
        matrix: [f32; 6],
    },
    /// Five facial landmarks in source pixels (OpenCV index coordinates, as
    /// returned by RetinaFace/SCRFD/YOLO-face). The crop is aligned to the
    /// ArcFace template scaled to the output size.
    Landmarks {
        /// Left eye, right eye, nose, left mouth, right mouth.
        points: [[f32; 2]; 5],
    },
}

impl Roi {
    /// Continuous `dst_from_src` transform producing this crop at
    /// `dst_w x dst_h`. `resize` is only used by [`Roi::Box`].
    pub fn dst_from_src(&self, dst_w: u32, dst_h: u32, resize: ResizeMode) -> Option<Affine2> {
        match *self {
            Roi::Box { x, y, w, h } => {
                if w <= 0.0 || h <= 0.0 {
                    return None;
                }
                Some(
                    Affine2::fit(w, h, dst_w as f32, dst_h as f32, resize)
                        .compose(&Affine2::translate(-x, -y)),
                )
            }
            Roi::Affine { matrix } => {
                let a = Affine2::from_opencv(matrix);
                a.invert().map(|_| a)
            }
            Roi::Landmarks { points } => {
                let (sx, sy) = (dst_w as f32 / 112.0, dst_h as f32 / 112.0);
                let template: Vec<[f32; 2]> =
                    ARCFACE_112_TEMPLATE.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
                Affine2::similarity(&points, &template).map(|m| Affine2::from_opencv(m.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn invert_roundtrip() {
        let m = Affine2([0.8, -0.6, 10.0, 0.6, 0.8, -4.0]);
        let inv = m.invert().unwrap();
        let (x, y) = m.apply(3.0, 7.0);
        let (bx, by) = inv.apply(x, y);
        assert!(close(bx, 3.0) && close(by, 7.0));
        let id = m.compose(&inv);
        for (a, b) in id.0.iter().zip(Affine2::IDENTITY.0.iter()) {
            assert!(close(*a, *b));
        }
    }

    #[test]
    fn letterbox_1080p_to_640() {
        let m = Affine2::fit(1920.0, 1080.0, 640.0, 640.0, ResizeMode::Letterbox);
        let s = 640.0 / 1920.0;
        assert!(close(m.0[0], s) && close(m.0[4], s));
        assert!(close(m.0[2], 0.0) && close(m.0[5], (640.0 - 1080.0 * s) / 2.0));
        let (x, y) = m.apply(1920.0, 1080.0);
        assert!(close(x, 640.0) && close(y, 500.0));
    }

    #[test]
    fn crop_center_covers() {
        let m = Affine2::fit(1920.0, 1080.0, 640.0, 640.0, ResizeMode::CropCenter);
        let (x0, _) = m.apply(0.0, 0.0);
        assert!(x0 < 0.0);
        let (_, y1) = m.apply(0.0, 1080.0);
        assert!(close(y1, 640.0));
    }

    #[test]
    fn sampling_matrix_matches_opencv_resize() {
        // cv2.resize maps dst index d to src index (d + 0.5) / s - 0.5.
        let m = Affine2::fit(100.0, 100.0, 50.0, 50.0, ResizeMode::Stretch);
        let s = Affine2::sampling_matrix(&m).unwrap();
        let (x, _) = s.apply(0.0, 0.0);
        assert!(close(x, 0.5));
        let (x, _) = s.apply(10.0, 0.0);
        assert!(close(x, 20.5));
    }

    #[test]
    fn opencv_conversion_roundtrip() {
        let m = [0.9, 0.1, 5.0, -0.1, 0.9, 7.0];
        let back = Affine2::from_opencv(m).to_opencv();
        for (a, b) in m.iter().zip(back.iter()) {
            assert!(close(*a, *b));
        }
    }

    #[test]
    fn similarity_recovers_known_transform() {
        let truth = Affine2([1.2 * 0.8, -1.2 * 0.6, 30.0, 1.2 * 0.6, 1.2 * 0.8, -12.0]);
        let src: Vec<[f32; 2]> = ARCFACE_112_TEMPLATE.to_vec();
        let dst: Vec<[f32; 2]> = src
            .iter()
            .map(|p| {
                let (x, y) = truth.apply(p[0], p[1]);
                [x, y]
            })
            .collect();
        let est = Affine2::similarity(&src, &dst).unwrap();
        for (a, b) in est.0.iter().zip(truth.0.iter()) {
            assert!(close(*a, *b), "{est:?} vs {truth:?}");
        }
    }

    #[test]
    fn landmarks_on_template_give_identity_crop() {
        let roi = Roi::Landmarks { points: ARCFACE_112_TEMPLATE };
        let m = roi.dst_from_src(112, 112, ResizeMode::Stretch).unwrap();
        for (a, b) in m.0.iter().zip(Affine2::IDENTITY.0.iter()) {
            assert!(close(*a, *b), "{m:?}");
        }
    }

    #[test]
    fn box_roi_maps_corners() {
        let roi = Roi::Box { x: 100.0, y: 50.0, w: 200.0, h: 100.0 };
        let m = roi.dst_from_src(64, 32, ResizeMode::Stretch).unwrap();
        let (x, y) = m.apply(100.0, 50.0);
        assert!(close(x, 0.0) && close(y, 0.0));
        let (x, y) = m.apply(300.0, 150.0);
        assert!(close(x, 64.0) && close(y, 32.0));
    }
}

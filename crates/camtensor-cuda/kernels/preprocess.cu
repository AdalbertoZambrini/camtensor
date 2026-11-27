// camtensor fused preprocessing kernel.
//
// One thread produces one output pixel (all channels) of one item:
//   sample the source (bilinear / nearest, affine sampling matrix)
//   -> Y'CbCr to RGB (or gray) -> channel order -> normalise -> store
// in NCHW or NHWC layout as u8, f16 or f32.
//
// The semantics are specified in crates/camtensor-cuda/src/preprocess.rs and
// mirrored by the CPU implementation in camtensor-preprocess.
//
// Compiled to PTX by `cargo run -p xtask -- ptx` (NVRTC, compute_52,
// --fmad=false so arithmetic rounds like the CPU reference). No headers are
// included on purpose: the PTX must build without a CUDA toolkit.

#define CT_MAX_ITEMS 64

// Source kinds.
#define CT_SRC_NV12 0
#define CT_SRC_YUV_PLANAR 1
#define CT_SRC_GRAY 2
#define CT_SRC_RGB 3
#define CT_SRC_BGR 4

// Output colour.
#define CT_OUT_RGB 0
#define CT_OUT_BGR 1
#define CT_OUT_GRAY 2

// Output dtype.
#define CT_U8 0
#define CT_F16 1
#define CT_F32 2

struct CtItem {
    float m[6];               // sampling matrix: src_idx = M * dst_idx
    unsigned long long dst;   // byte address of this item's output image
};

// Mirrors `KernelParams` in preprocess.rs (size and offsets are asserted
// there).
struct CtParams {
    unsigned long long plane[3];
    int pitch[3];
    int src_w, src_h;
    int chroma_w, chroma_h;
    float chroma_sx, chroma_sy;
    int src_kind;
    float y_off, y_scale, c_scale;
    float cr_r, cb_g, cr_g, cb_b;
    int dst_w, dst_h;
    int out_kind;
    int dtype;
    int nhwc;
    int bilinear;
    float norm_a[3];
    float norm_b[3];
    float pad[3];
    int n_items;
    int reserved;
    CtItem items[CT_MAX_ITEMS];
};

struct Taps {
    int o00, o10, o01, o11;   // byte offsets of the four taps
    float w00, w10, w01, w11; // bilinear weights
};

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Tap offsets for sampling a w x h plane at (sx, sy); `step` is the byte
// distance between horizontally adjacent samples of one channel.
__device__ __forceinline__ Taps make_taps(float sx, float sy, int w, int h, int pitch, int step,
                                          int bilinear) {
    Taps t;
    if (!bilinear) {
        int xi = clampi((int)floorf(sx + 0.5f), 0, w - 1);
        int yi = clampi((int)floorf(sy + 0.5f), 0, h - 1);
        t.o00 = yi * pitch + xi * step;
        t.o10 = t.o00;
        t.o01 = t.o00;
        t.o11 = t.o00;
        t.w00 = 1.0f;
        t.w10 = 0.0f;
        t.w01 = 0.0f;
        t.w11 = 0.0f;
        return t;
    }
    float x0f = floorf(sx);
    float y0f = floorf(sy);
    float fx = sx - x0f;
    float fy = sy - y0f;
    int x0 = (int)x0f;
    int y0 = (int)y0f;
    int xa = clampi(x0, 0, w - 1) * step;
    int xb = clampi(x0 + 1, 0, w - 1) * step;
    int ya = clampi(y0, 0, h - 1) * pitch;
    int yb = clampi(y0 + 1, 0, h - 1) * pitch;
    t.o00 = ya + xa;
    t.o10 = ya + xb;
    t.o01 = yb + xa;
    t.o11 = yb + xb;
    float gx = 1.0f - fx;
    float gy = 1.0f - fy;
    t.w00 = gx * gy;
    t.w10 = fx * gy;
    t.w01 = gx * fy;
    t.w11 = fx * fy;
    return t;
}

__device__ __forceinline__ float fetch(const unsigned char* __restrict__ base, const Taps& t,
                                       int bilinear) {
    float p00 = (float)__ldg(base + t.o00);
    if (!bilinear) return p00;
    float p10 = (float)__ldg(base + t.o10);
    float p01 = (float)__ldg(base + t.o01);
    float p11 = (float)__ldg(base + t.o11);
    return t.w00 * p00 + t.w10 * p10 + t.w01 * p01 + t.w11 * p11;
}

__device__ __forceinline__ unsigned short f32_to_f16(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

extern "C" __global__ void ct_preprocess(const CtParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dst_w || y >= p.dst_h) return;
    const int item = blockIdx.z;

    const float* m = p.items[item].m;
    const float fx = (float)x;
    const float fy = (float)y;
    const float sx = m[0] * fx + m[1] * fy + m[2];
    const float sy = m[3] * fx + m[4] * fy + m[5];

    const int nc = p.out_kind == CT_OUT_GRAY ? 1 : 3;
    float v0, v1, v2;

    const bool outside = sx < -0.5f || sx > (float)p.src_w - 0.5f || sy < -0.5f ||
                         sy > (float)p.src_h - 0.5f;
    if (outside) {
        v0 = p.pad[0];
        v1 = p.pad[1];
        v2 = p.pad[2];
    } else {
        float r, g, b, gray;
        bool have_gray = false;
        const int bil = p.bilinear;
        const unsigned char* plane0 = (const unsigned char*)p.plane[0];
        if (p.src_kind == CT_SRC_NV12 || p.src_kind == CT_SRC_YUV_PLANAR) {
            Taps tl = make_taps(sx, sy, p.src_w, p.src_h, p.pitch[0], 1, bil);
            const float Y = fetch(plane0, tl, bil);
            const float yp = (Y - p.y_off) * p.y_scale;
            if (p.out_kind == CT_OUT_GRAY) {
                gray = clampf(yp, 0.0f, 255.0f);
                have_gray = true;
                r = g = b = 0.0f;
            } else {
                const float cx = (sx + 0.5f) * p.chroma_sx - 0.5f;
                const float cy = (sy + 0.5f) * p.chroma_sy - 0.5f;
                float U, V;
                if (p.src_kind == CT_SRC_NV12) {
                    const unsigned char* uv = (const unsigned char*)p.plane[1];
                    Taps tc = make_taps(cx, cy, p.chroma_w, p.chroma_h, p.pitch[1], 2, bil);
                    U = fetch(uv, tc, bil);
                    V = fetch(uv + 1, tc, bil);
                } else {
                    Taps tu = make_taps(cx, cy, p.chroma_w, p.chroma_h, p.pitch[1], 1, bil);
                    U = fetch((const unsigned char*)p.plane[1], tu, bil);
                    Taps tv = make_taps(cx, cy, p.chroma_w, p.chroma_h, p.pitch[2], 1, bil);
                    V = fetch((const unsigned char*)p.plane[2], tv, bil);
                }
                const float cb = (U - 128.0f) * p.c_scale;
                const float cr = (V - 128.0f) * p.c_scale;
                r = clampf(yp + p.cr_r * cr, 0.0f, 255.0f);
                g = clampf(yp - p.cb_g * cb - p.cr_g * cr, 0.0f, 255.0f);
                b = clampf(yp + p.cb_b * cb, 0.0f, 255.0f);
            }
        } else if (p.src_kind == CT_SRC_GRAY) {
            Taps t = make_taps(sx, sy, p.src_w, p.src_h, p.pitch[0], 1, bil);
            gray = fetch(plane0, t, bil);
            have_gray = true;
            r = g = b = gray;
        } else {
            Taps t = make_taps(sx, sy, p.src_w, p.src_h, p.pitch[0], 3, bil);
            const float c0 = fetch(plane0, t, bil);
            const float c1 = fetch(plane0 + 1, t, bil);
            const float c2 = fetch(plane0 + 2, t, bil);
            if (p.src_kind == CT_SRC_RGB) {
                r = c0; g = c1; b = c2;
            } else {
                r = c2; g = c1; b = c0;
            }
        }
        if (p.out_kind == CT_OUT_GRAY) {
            v0 = have_gray ? gray : 0.299f * r + 0.587f * g + 0.114f * b;
            v1 = v2 = 0.0f;
        } else if (p.out_kind == CT_OUT_BGR) {
            v0 = b; v1 = g; v2 = r;
        } else {
            v0 = r; v1 = g; v2 = b;
        }
    }

    const long long hw = (long long)p.dst_w * p.dst_h;
    const long long pix = (long long)y * p.dst_w + x;
    const unsigned long long base = p.items[item].dst;
    for (int c = 0; c < nc; ++c) {
        const float v = c == 0 ? v0 : (c == 1 ? v1 : v2);
        const long long idx = p.nhwc ? pix * nc + c : (long long)c * hw + pix;
        if (p.dtype == CT_U8) {
            const float q = clampf(floorf(v + 0.5f), 0.0f, 255.0f);
            ((unsigned char*)base)[idx] = (unsigned char)q;
        } else {
            const float o = v * p.norm_a[c] + p.norm_b[c];
            if (p.dtype == CT_F16) {
                ((unsigned short*)base)[idx] = f32_to_f16(o);
            } else {
                ((float*)base)[idx] = o;
            }
        }
    }
}

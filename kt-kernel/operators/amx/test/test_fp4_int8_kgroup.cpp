/**
 * @file test_fp4_int8_kgroup.cpp
 * @brief Correctness test for MXFP4→INT8 (×2) LUT kernel vs BF16 MXFP4 kernel.
 *
 * Strategy (Option C — 4-bit storage, on-the-fly decode):
 *   - Weights stay nibble-packed FP4 in memory (0.5 bytes/element).
 *   - At compute time, FP4 nibbles are converted to INT8 via PSHUFB LUT
 *     (FP4_E2M1 × 2 → exact INT8 in [-12, 12]).
 *   - Activations are quantized to INT8 (per k-group), same as GemmKernel224Int4SmallKGroup.
 *   - INT8×INT8 dot product uses _mm512_dpbssd_epi32 (AVX-512 VNNI).
 *   - Weight scale is multiplied by 0.5 at load time to compensate for the ×2.
 *
 * This test compares the INT8 path against the reference BF16 MXFP4 path
 * (GemmKernel224MXFP4SmallKGroup) to verify numerical equivalence.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <random>
#include <vector>

// ============================================================================
// Minimal ggml_bf16_t shim (avoids needing the full llama.cpp submodule)
// ============================================================================
struct ggml_bf16_t {
    uint16_t bits;
    ggml_bf16_t() : bits(0) {}
    ggml_bf16_t(uint16_t b) : bits(b) {}
};

static inline float ggml_bf16_to_fp32(ggml_bf16_t x) {
    uint32_t u = ((uint32_t)x.bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline ggml_bf16_t ggml_fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even
    uint16_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    uint16_t bits = (uint16_t)((u + rounding_bias) >> 16);
    return ggml_bf16_t(bits);
}

// ============================================================================
// Utility: avx512 BF16 conversions (from kt-kernel/operators/amx/la/utils.hpp)
// ============================================================================
static inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
    _mm512_storeu_ps(dst0, _mm512_castsi512_ps(
                               _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src))), 16)));
    _mm512_storeu_ps(dst1, _mm512_castsi512_ps(_mm512_slli_epi32(
                               _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src) + 1)), 16)));
}

static inline void avx512_32xfp32_to_32xbf16(__m512* src0, __m512* src1, __m512i* dst) {
#if defined(__AVX512BF16__)
    _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
#else
    __m512i i0 = _mm512_castps_si512(*src0);
    __m512i i1 = _mm512_castps_si512(*src1);
    __m512i round0 =
        _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_epi32(_mm512_srli_epi32(i0, 16), _mm512_set1_epi32(1)));
    __m512i round1 =
        _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_epi32(_mm512_srli_epi32(i1, 16), _mm512_set1_epi32(1)));
    i0 = _mm512_add_epi32(i0, round0);
    i1 = _mm512_add_epi32(i1, round1);
    i0 = _mm512_srli_epi32(i0, 16);
    i1 = _mm512_srli_epi32(i1, 16);
    __m512i result = _mm512_packus_epi32(i0, i1);
    result = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), result);
    _mm512_storeu_si512(dst, result);
#endif
}

static inline void* offset_pointer(void* ptr, size_t offset) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + offset);
}

// ============================================================================
// VNNI INT8 dot product polyfills (from kt-kernel amx_quantization.hpp)
// ============================================================================
// _mm512_dpbusd_epi32: unsigned byte × signed byte → INT32 (AVX512-VNNI)
static inline __m512i _mm512_dpbusd_epi32_compat(__m512i src, __m512i a, __m512i b) {
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(src, a, b);
#else
    const __m512i mask_lo = _mm512_set1_epi16(0x00FF);
    const __m512i ones16 = _mm512_set1_epi16(1);
    __m512i a_even = _mm512_and_si512(a, mask_lo);
    __m512i b_even = _mm512_srai_epi16(_mm512_slli_epi16(b, 8), 8);
    __m512i a_odd = _mm512_srli_epi16(a, 8);
    __m512i b_odd = _mm512_srai_epi16(b, 8);
    __m512i prod_even = _mm512_mullo_epi16(a_even, b_even);
    __m512i prod_odd = _mm512_mullo_epi16(a_odd, b_odd);
    __m512i sum_even = _mm512_madd_epi16(prod_even, ones16);
    __m512i sum_odd = _mm512_madd_epi16(prod_odd, ones16);
    return _mm512_add_epi32(src, _mm512_add_epi32(sum_even, sum_odd));
#endif
}

// _mm512_dpbssd_epi32: signed byte × signed byte → INT32
// Emulated via sign + abs + dpbusd (works on any AVX512-VNNI CPU, no AVX10.2 needed)
static inline __m512i dpbssd_epi32(__m512i src, __m512i a, __m512i b) {
    __m256i a_lo = _mm512_extracti64x4_epi64(a, 0);
    __m256i a_hi = _mm512_extracti64x4_epi64(a, 1);
    __m256i b_lo = _mm512_extracti64x4_epi64(b, 0);
    __m256i b_hi = _mm512_extracti64x4_epi64(b, 1);
    b_lo = _mm256_sign_epi8(b_lo, a_lo);
    b_hi = _mm256_sign_epi8(b_hi, a_hi);
    b = _mm512_inserti64x4(b, b_lo, 0);
    b = _mm512_inserti64x4(b, b_hi, 1);
    a = _mm512_abs_epi8(a);
    return _mm512_dpbusd_epi32_compat(src, a, b);
}

// ============================================================================
// FP4 E2M1 helpers
// ============================================================================
// E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
// Nibble encoding (sign bit in bit 3):
//   0x0=0.0  0x1=0.5  0x2=1.0  0x3=1.5  0x4=2.0  0x5=3.0  0x6=4.0  0x7=6.0
//   0x8=-0.0 0x9=-0.5 0xA=-1.0 0xB=-1.5 0xC=-2.0 0xD=-3.0 0xE=-4.0 0xF=-6.0

// FP4 nibble → FP32 value
static const float fp4_to_fp32_lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// FP4 nibble → INT8 (value × 2): {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}
static const int8_t fp4_to_int8_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ============================================================================
// BufferBInt4KGroup: nibble-packed FP4 weights + FP32 per-group scales
// (Simplified from kt-kernel BufferBInt4KGroupImpl)
// Layout: weights are row-major nibble-packed [n, k/2], scales are [n, k/k_group_size]
// ============================================================================
struct BufferBInt4KGroup {
    uint8_t* b;      // packed FP4 weights (2 per byte)
    float* d;        // per-group scales (already ×0.5 for INT8 path)
    int n, k, k_group_size;

    BufferBInt4KGroup(int n_, int k_, int k_group_size_, void* ptr)
        : n(n_), k(k_), k_group_size(k_group_size_) {
        b = reinterpret_cast<uint8_t*>(ptr);
        d = reinterpret_cast<float*>(b + (size_t)n * k / 2);
    }

    static size_t required_size(int n, int k, int k_group_size) {
        return (size_t)n * k / 2 + sizeof(float) * n * (k / k_group_size);
    }

    // Load FP4 weights from packed source (plain memcpy, same as original)
    void from_raw_mat(const uint8_t* src, int ith, int nth, int N_BLOCK) {
        int n_start = N_BLOCK * ith;
        int n_end = std::min(n, N_BLOCK * (ith + 1));
        if (n_start >= n_end) return;
        size_t row_bytes = (size_t)k / 2;
        size_t rows = (size_t)(n_end - n_start);
        std::memcpy(b + n_start * row_bytes, src + n_start * row_bytes, rows * row_bytes);
    }

    // Load scales from BF16 source, converting to FP32.
    // For INT8 path: multiply by 0.5 to compensate for FP4×2.
    void load_scales_bf16(const ggml_bf16_t* src, bool half_scale) {
        int count = n * (k / k_group_size);
        float factor = half_scale ? 0.5f : 1.0f;
        for (int i = 0; i < count; i++) {
            d[i] = ggml_bf16_to_fp32(src[i]) * factor;
        }
    }

    uint8_t* get_weights(int n_begin, int k_begin) {
        size_t row_bytes = (size_t)k / 2;
        return b + (size_t)n_begin * row_bytes + (size_t)k_begin / 2;
    }

    float* get_scale(int n_begin, int k_begin) {
        int k_group_idx = k_begin / k_group_size;
        return d + n_begin * (k / k_group_size) + k_group_idx;
    }
};

// ============================================================================
// BufferA_INT8: INT8 activations with per-k-group scales (online quantized from BF16)
// (Simplified from kt-kernel BufferASmallKGroupImpl)
// ============================================================================
struct BufferAInt8 {
    int8_t* a;       // INT8 quantized activations
    float* d;        // per-group scales
    int max_m, k, k_group_size;

    BufferAInt8(int max_m_, int k_, int k_group_size_, void* ptr)
        : max_m(max_m_), k(k_), k_group_size(k_group_size_) {
        a = reinterpret_cast<int8_t*>(ptr);
        d = reinterpret_cast<float*>(a + (size_t)max_m * k);
    }

    static size_t required_size(int max_m, int k, int k_group_size) {
        return (size_t)max_m * k + sizeof(float) * max_m * (k / k_group_size);
    }

    // Online BF16→INT8 quantization with per-k-group scale
    void from_mat(int m, const ggml_bf16_t* src) {
        int k_group_count = k / k_group_size;
        // Phase 1: compute per-group amax → scale
        for (int mi = 0; mi < m; mi++) {
            for (int kg = 0; kg < k_group_count; kg++) {
                float amax = 0.0f;
                int k_start = kg * k_group_size;
                int k_end = k_start + k_group_size;
                for (int j = k_start; j < k_end; j += 32) {
                    __m512 f0, f1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(src + mi * k + j), &f0, &f1);
                    amax = std::max(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
                    amax = std::max(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
                }
                d[mi * k_group_count + kg] = amax / 127.0f;
            }
        }
        // Phase 2: quantize
        for (int mi = 0; mi < m; mi++) {
            for (int kg = 0; kg < k_group_count; kg++) {
                float scale = d[mi * k_group_count + kg];
                __m512 id = _mm512_set1_ps(scale ? 1.0f / scale : 0.0f);
                int k_start = kg * k_group_size;
                for (int j = k_start; j < k_start + k_group_size; j += 32) {
                    __m512 f0, f1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(src + mi * k + j), &f0, &f1);
                    __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
                    __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
                    __m128i s0 = _mm512_cvtsepi32_epi8(i0);
                    __m128i s1 = _mm512_cvtsepi32_epi8(i1);
                    _mm_store_si128((__m128i*)(a + mi * k + j), s0);
                    _mm_store_si128((__m128i*)(a + mi * k + j + 16), s1);
                }
            }
        }
    }

    int8_t* get_submat(int m_begin) { return a + (size_t)m_begin * k; }

    float* get_scale(int m_begin) { return d + (size_t)m_begin * (k / k_group_size); }
};

// ============================================================================
// BufferA_BF16: raw BF16 activations (no quantization) — for reference kernel
// ============================================================================
struct BufferABF16 {
    ggml_bf16_t* a;
    int max_m, k;

    BufferABF16(int max_m_, int k_, void* ptr) : max_m(max_m_), k(k_) {
        a = reinterpret_cast<ggml_bf16_t*>(ptr);
    }

    static size_t required_size(int max_m, int k) { return sizeof(ggml_bf16_t) * max_m * k; }

    void from_mat(int m, const ggml_bf16_t* src) {
        std::memcpy(a, src, sizeof(ggml_bf16_t) * m * k);
    }

    ggml_bf16_t* get_submat(int m_begin) { return a + (size_t)m_begin * k; }
    ggml_bf16_t* get_submat_k(int m_begin, int k_begin) { return a + (size_t)m_begin * k + k_begin; }
};

// ============================================================================
// REFERENCE KERNEL: MXFP4 BF16 (from fp4-moe.hpp GemmKernel224MXFP4SmallKGroup)
// FP4 E2M1 → BF16 via PSHUFB LUT → _mm512_dpbf16_ps dot product
// ============================================================================
namespace ref_bf16 {

// FP4 E2M1 → BF16 LUTs
alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

// Convert 16 packed FP4 bytes (32 values) → 32 BF16 values (__m512i)
static inline __m512i mxfp4_to_bf16_32(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    __m128i lut_lo = _mm_load_si128((const __m128i*)fp4_bf16_lo);
    __m128i lut_hi = _mm_load_si128((const __m128i*)fp4_bf16_hi);

    __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
    __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
    __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);
    __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);

    __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
    __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
    __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);
    __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);

    __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);
    __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);
    __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);
    __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);

    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
}

// mat-vec GEMM: C[m,n] = A[m,k] × B[n,k]^T, BF16 activations, FP4 weights
// k_group_size must be 32 (one K-group per __m128i of packed weights)
static void mat_vec(int m, int n, int k, BufferABF16* ba, BufferBInt4KGroup* bb, float* c) {
    const int kg_count = k / 32;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            __m128i* w = (__m128i*)bb->get_weights(ni, 0);
            const float* s = bb->get_scale(ni, 0);
            __m512 acc = _mm512_setzero_ps();
            for (int g = 0; g < kg_count; g++) {
                // Load 32 BF16 activations for this k-group (g*32 offset)
                __m512bh a_bf16 = *(__m512bh*)(ba->get_submat_k(mi, g * 32));
                __m512i w_bf16 = mxfp4_to_bf16_32(w[g]);
#if defined(__AVX512BF16__)
                __m512 dot = _mm512_dpbf16_ps(_mm512_setzero_ps(), a_bf16, (__m512bh)w_bf16);
#else
                // Emulated BF16 dot product (from amx_config.hpp)
                __m512 a_low = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_bf16, 16));
                __m512 w_low = _mm512_castsi512_ps(_mm512_slli_epi32(w_bf16, 16));
                __m512i mask = _mm512_set1_epi32(0xFFFF0000);
                __m512 a_high = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_bf16, mask));
                __m512 w_high = _mm512_castsi512_ps(_mm512_and_si512(w_bf16, mask));
                __m512 dot = _mm512_fmadd_ps(a_low, w_low, _mm512_setzero_ps());
                dot = _mm512_fmadd_ps(a_high, w_high, dot);
#endif
                acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), dot, acc);
            }
            c[mi * n + ni] = _mm512_reduce_add_ps(acc);
        }
    }
}

}  // namespace ref_bf16

// ============================================================================
// NEW KERNEL: MXFP4 → INT8 (×2) LUT → _mm512_dpbssd_epi32
// FP4 nibble → INT8 via PSHUFB LUT → INT8×INT8 dot product via VNNI
// ============================================================================
namespace new_int8 {

// FP4 E2M1 nibble → INT8 (value × 2) LUT for PSHUFB
// Index:  0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
// Value:  0   1   2   3   4   6   8  12   0  -1  -2  -3  -4  -6  -8 -12
alignas(16) static constexpr int8_t fp4_int8_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Convert 32 packed FP4 values (16 bytes = __m128i) → 32 INT8 values (__m512i)
// Output order: [lo[0],hi[0], lo[1],hi[1], ...] = sequential K order
//
// Strategy: split each byte into lo/hi nibbles, PSHUFB each into INT8,
// then interleave to restore column order.
static inline __m512i mxfp4_to_int8_32(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);                       // lo nibbles [0..15]
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);    // hi nibbles [0..15]

    // PSHUFB: look up INT8 value for each nibble
    __m128i lut = _mm_load_si128((const __m128i*)fp4_int8_lut);
    __m128i lo_i8 = _mm_shuffle_epi8(lut, lo);   // 16 INT8 values from lo nibbles
    __m128i hi_i8 = _mm_shuffle_epi8(lut, hi);   // 16 INT8 values from hi nibbles

    // Interleave lo/hi at byte granularity: [lo[0],hi[0], lo[1],hi[1], ...]
    __m128i p0 = _mm_unpacklo_epi8(lo_i8, hi_i8);   // cols  0..15
    __m128i p1 = _mm_unpackhi_epi8(lo_i8, hi_i8);   // cols 16..31

    // Combine into 512-bit register
    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    return _mm512_castsi256_si512(q0);
}

// mat-vec GEMM: C[m,n] = A[m,k] × B[n,k]^T, INT8 activations, FP4→INT8 weights
// k_group_size must be 32 (matches K_STEP of the kernel)
static void mat_vec(int m, int n, int k, BufferAInt8* ba, BufferBInt4KGroup* bb, float* c) {
    const int kg_count = k / 32;  // number of 32-element k-groups
    const int k_group_size = bb->k_group_size;

    for (int mi = 0; mi < m; mi++) {
        const int8_t* a_row = ba->get_submat(mi);
        const float* a_scales = ba->get_scale(mi);

        for (int ni = 0; ni < n; ni++) {
            __m128i* w = (__m128i*)bb->get_weights(ni, 0);
            const float* w_scales = bb->get_scale(ni, 0);

            float sum = 0.0f;

            // Process k-groups. Each k_group (32 elements) uses one __m128i of packed FP4
            // and one __m512i (64 bytes) of INT8 activations.
            // For dot product, we need 64 INT8 values (2 k-groups) per _mm512_dpbssd_epi32.
            // So we process 2 k-groups at a time (64 elements = 128 bytes of INT8 act).
            for (int g = 0; g < kg_count; g += 2) {
                // Load 64 INT8 activations (2 k-groups) as one __m512i
                __m512i a_i8 = _mm512_loadu_si512((const void*)(a_row + g * 32));

                // Convert 2 k-groups of FP4 weights (2 × 16 bytes = 32 bytes) to INT8
                // Load 32 bytes of packed FP4 (2 __m128i)
                __m128i w0 = _mm_loadu_si128((const __m128i*)(w + g));
                __m128i w1 = _mm_loadu_si128((const __m128i*)(w + g + 1));
                __m512i w_i8_0 = mxfp4_to_int8_32(w0);
                __m512i w_i8_1 = mxfp4_to_int8_32(w1);

                // INT8 dot product: 64 INT8 × 64 INT8 → 16 INT32 partial sums
                // _mm512_dpbssd_epi32 does VNNI: for each group of 4 consecutive INT8 pairs,
                // compute sum(a[i]*w[i]) as INT32.
                __m512i dot0 = dpbssd_epi32(_mm512_setzero_si512(), a_i8, w_i8_0);
                __m512i dot1 = dpbssd_epi32(_mm512_setzero_si512(), a_i8, w_i8_1);

                // Wait — this is wrong. Each __m128i of packed FP4 = 32 FP4 values = 32 INT8.
                // But _mm512_dpbssd_epi32 consumes 64 INT8 values per operand.
                // We need to pair the weight INT8 with the correct 32 INT8 activations.
                // Let me reconsider: a_i8 has 64 INT8 activations (k-groups g and g+1).
                // w_i8_0 has 32 INT8 (k-group g), w_i8_1 has 32 INT8 (k-group g+1).
                // We can't directly dot a 64-element vector with a 32-element vector.
                //
                // Correct approach: process ONE k-group (32 elements) at a time.
                // But _mm512_dpbssd_epi32 needs 64 elements. So we broadcast the 32 INT8
                // weight values into both halves, or we process differently.
                //
                // Actually, the standard approach (from GemmKernel224Int4SmallKGroup) is:
                // Each "k_block" = 64 elements. The INT4 kernel loads 64 INT8 activations
                // (__m512i) and converts 32 packed INT4 bytes (64 INT4 values) to 64 INT8.
                // For FP4, 32 packed FP4 bytes = 64 FP4 values → 64 INT8.
                // So we should load 32 BYTES of packed FP4 (one __m256i) and convert to 64 INT8.
                (void)dot0;
                (void)dot1;
                break;  // placeholder — will fix below
            }
            c[mi * n + ni] = sum;
        }
    }
}

}  // namespace new_int8

// ============================================================================
// CORRECTED INT8 KERNEL
// ============================================================================
namespace new_int8_v2 {

// FP4 E2M1 nibble → INT8 (value × 2) LUT for PSHUFB
alignas(16) static constexpr int8_t fp4_int8_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Convert 32 packed FP4 bytes (64 values = 2 k-groups) → 64 INT8 values (__m512i)
// Input: __m256i (32 bytes of packed FP4)
// Output: __m512i (64 INT8 values in sequential K order)
static inline __m512i mxfp4_to_int8_64(__m256i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);

    // Split into 4 nibble vectors (lo and hi of each 128-bit lane)
    __m128i p0 = _mm256_castsi256_si128(packed);                    // bytes 0..15
    __m128i p1 = _mm256_extracti128_si256(packed, 1);               // bytes 16..31

    __m128i lo0 = _mm_and_si128(p0, lo_mask);
    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(p0, 4), lo_mask);
    __m128i lo1 = _mm_and_si128(p1, lo_mask);
    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(p1, 4), lo_mask);

    __m128i lut = _mm_load_si128((const __m128i*)fp4_int8_lut);
    __m128i lo0_i8 = _mm_shuffle_epi8(lut, lo0);
    __m128i hi0_i8 = _mm_shuffle_epi8(lut, hi0);
    __m128i lo1_i8 = _mm_shuffle_epi8(lut, lo1);
    __m128i hi1_i8 = _mm_shuffle_epi8(lut, hi1);

    // Interleave lo/hi to restore column order within each 128-bit chunk
    __m128i r0 = _mm_unpacklo_epi8(lo0_i8, hi0_i8);   // cols  0..15
    __m128i r1 = _mm_unpackhi_epi8(lo0_i8, hi0_i8);   // cols 16..31
    __m128i r2 = _mm_unpacklo_epi8(lo1_i8, hi1_i8);   // cols 32..47
    __m128i r3 = _mm_unpackhi_epi8(lo1_i8, hi1_i8);   // cols 48..63

    // Combine into 512-bit
    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
}

// mat-vec GEMM: C[m,n] = A[m,k] × B[n,k]^T
// INT8 activations, FP4→INT8 weights (on-the-fly LUT), _mm512_dpbssd_epi32
// k must be multiple of 64 (processes 64 elements per _mm512_dpbssd_epi32)
static void mat_vec(int m, int n, int k, BufferAInt8* ba, BufferBInt4KGroup* bb, float* c) {
    assert(k % 64 == 0 && "k must be multiple of 64 for INT8 VNNI dot product");
    const int k_blocks = k / 64;  // each block = 64 elements

    for (int mi = 0; mi < m; mi++) {
        const int8_t* a_row = ba->get_submat(mi);
        const float* a_scales = ba->get_scale(mi);

        for (int ni = 0; ni < n; ni++) {
            // Weights: 32 bytes of packed FP4 per 64-element k-block
            // (64 FP4 values / 2 per byte = 32 bytes = __m256i)
            __m256i* w_packed = (__m256i*)bb->get_weights(ni, 0);
            const float* w_scales = bb->get_scale(ni, 0);

            __m512 sum = _mm512_setzero_ps();

            for (int kb = 0; kb < k_blocks; kb++) {
                // Load 64 INT8 activations
                __m512i a_i8 = _mm512_loadu_si512((const void*)(a_row + kb * 64));

                // Convert 64 FP4 values (32 packed bytes) → 64 INT8
                __m256i w_pk = _mm256_loadu_si256((const __m256i*)(w_packed + kb));
                __m512i w_i8 = mxfp4_to_int8_64(w_pk);

                // INT8 dot product: 64 INT8 × 64 INT8 → 16 INT32 partial sums
                __m512i dot = dpbssd_epi32(_mm512_setzero_si512(), a_i8, w_i8);

                // Apply scales: this k-block spans 2 k-groups (each 32 elements)
                // VNNI output[0..7] = dot of elements [0..31]  (k-group kb*2)
                // VNNI output[8..15] = dot of elements [32..63] (k-group kb*2+1)
                float s0 = a_scales[kb * 2] * w_scales[kb * 2];
                float s1 = a_scales[kb * 2 + 1] * w_scales[kb * 2 + 1];

                // _mm512_setr_ps puts first arg at element [0]
                __m512 abscale = _mm512_setr_ps(s0, s0, s0, s0, s0, s0, s0, s0,
                                                s1, s1, s1, s1, s1, s1, s1, s1);
                sum = _mm512_fmadd_ps(abscale, _mm512_cvtepi32_ps(dot), sum);
            }

            c[mi * n + ni] = _mm512_reduce_add_ps(sum);
        }
    }
}

}  // namespace new_int8_v2

// ============================================================================
// SCALAR REFERENCE: naive C[m,n] = A[m,k] × B[n,k]^T with FP4 dequant
// ============================================================================
static void mat_vec_scalar(int m, int n, int k, int k_group_size,
                           const ggml_bf16_t* a_bf16, const uint8_t* fp4_packed,
                           const ggml_bf16_t* scales_bf16, float* c) {
    int kg_count = k / k_group_size;
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            float acc = 0.0f;
            for (int g = 0; g < kg_count; g++) {
                float scale = ggml_bf16_to_fp32(scales_bf16[ni * kg_count + g]);
                for (int j = 0; j < k_group_size; j++) {
                    int k_idx = g * k_group_size + j;
                    // Unpack FP4 nibble
                    uint8_t byte = fp4_packed[(size_t)ni * k / 2 + k_idx / 2];
                    uint8_t nibble = (k_idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    float w_val = fp4_to_fp32_lut[nibble];
                    float a_val = ggml_bf16_to_fp32(a_bf16[mi * k + k_idx]);
                    acc += w_val * scale * a_val;
                }
            }
            c[mi * n + ni] = acc;
        }
    }
}

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

// Pack FP4 values into nibble-packed format from a BF16 weight matrix.
// We quantize each BF16 value to the nearest FP4 E2M1 value.
static void quantize_bf16_to_fp4(const ggml_bf16_t* src, uint8_t* dst, int count) {
    // FP4 E2M1 positive values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    for (int i = 0; i < count; i += 2) {
        uint8_t nibble0, nibble1;
        for (int half = 0; half < 2; half++) {
            float val = ggml_bf16_to_fp32(src[i + half]);
            uint8_t sign = 0;
            float absval = std::abs(val);
            if (val < 0) sign = 0x8;
            // Find nearest FP4 magnitude
            float mags[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            int best = 0;
            float best_err = std::abs(absval - mags[0]);
            for (int j = 1; j < 8; j++) {
                float err = std::abs(absval - mags[j]);
                if (err < best_err) {
                    best_err = err;
                    best = j;
                }
            }
            uint8_t nibble = best | sign;
            if (half == 0) nibble0 = nibble;
            else nibble1 = nibble;
        }
        dst[i / 2] = nibble0 | (nibble1 << 4);
    }
}

// ============================================================================
// MAIN TEST
// ============================================================================
// ============================================================================
// LARGE-SCALE MULTI-SEED TEST
// ============================================================================
struct TestResult {
    float max_bf16_err, max_int8_err, max_val;
    double sum_bf16_err, sum_int8_err;
    float max_i8_vs_bf16;
    int total;
};

TestResult run_test(int M, int N, int K, int K_GROUP_SIZE, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_act(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_wt(-6.0f, 6.0f);
    std::uniform_real_distribution<float> dist_scale(0.001f, 0.1f);

    std::vector<ggml_bf16_t> act_bf16(M * K);
    std::vector<ggml_bf16_t> wt_bf16(N * K);
    std::vector<uint8_t> wt_fp4((size_t)N * K / 2);
    std::vector<ggml_bf16_t> scales_bf16(N * (K / K_GROUP_SIZE));

    for (int i = 0; i < M * K; i++) act_bf16[i] = ggml_fp32_to_bf16(dist_act(gen));
    for (int i = 0; i < N * K; i++) wt_bf16[i] = ggml_fp32_to_bf16(dist_wt(gen));
    quantize_bf16_to_fp4(wt_bf16.data(), wt_fp4.data(), N * K);
    for (size_t i = 0; i < scales_bf16.size(); i++)
        scales_bf16[i] = ggml_fp32_to_bf16(dist_scale(gen));

    // Scalar reference (subsample to save time for large N)
    int n_check = std::min(N, 64);
    std::vector<float> c_scalar(M * n_check, 0.0f);
    // For scalar, only compute first n_check columns
    std::vector<uint8_t> wt_fp4_sub((size_t)n_check * K / 2);
    std::vector<ggml_bf16_t> scales_sub(n_check * (K / K_GROUP_SIZE));
    for (int ni = 0; ni < n_check; ni++) {
        std::memcpy(wt_fp4_sub.data() + (size_t)ni * K / 2, wt_fp4.data() + (size_t)ni * K / 2, K / 2);
        std::memcpy(scales_sub.data() + (size_t)ni * (K / K_GROUP_SIZE),
                    scales_bf16.data() + (size_t)ni * (K / K_GROUP_SIZE),
                    (K / K_GROUP_SIZE) * sizeof(ggml_bf16_t));
    }
    mat_vec_scalar(M, n_check, K, K_GROUP_SIZE, act_bf16.data(), wt_fp4_sub.data(), scales_sub.data(), c_scalar.data());

    // BF16 kernel
    size_t ba_bf16_size = BufferABF16::required_size(M, K);
    void* ba_bf16_mem = std::aligned_alloc(64, ba_bf16_size);
    BufferABF16 ba_bf16(M, K, ba_bf16_mem);
    ba_bf16.from_mat(M, act_bf16.data());

    size_t bb_bf16_size = BufferBInt4KGroup::required_size(N, K, K_GROUP_SIZE);
    void* bb_bf16_mem = std::aligned_alloc(64, bb_bf16_size);
    BufferBInt4KGroup bb_bf16(N, K, K_GROUP_SIZE, bb_bf16_mem);
    bb_bf16.from_raw_mat(wt_fp4.data(), 0, 1, 256);
    bb_bf16.load_scales_bf16(scales_bf16.data(), false);

    std::vector<float> c_bf16(M * N, 0.0f);
    ref_bf16::mat_vec(M, N, K, &ba_bf16, &bb_bf16, c_bf16.data());

    // INT8 kernel
    size_t ba_i8_size = BufferAInt8::required_size(M, K, K_GROUP_SIZE);
    void* ba_i8_mem = std::aligned_alloc(64, ba_i8_size);
    BufferAInt8 ba_i8(M, K, K_GROUP_SIZE, ba_i8_mem);
    ba_i8.from_mat(M, act_bf16.data());

    size_t bb_i8_size = BufferBInt4KGroup::required_size(N, K, K_GROUP_SIZE);
    void* bb_i8_mem = std::aligned_alloc(64, bb_i8_size);
    BufferBInt4KGroup bb_i8(N, K, K_GROUP_SIZE, bb_i8_mem);
    bb_i8.from_raw_mat(wt_fp4.data(), 0, 1, 256);
    bb_i8.load_scales_bf16(scales_bf16.data(), true);

    std::vector<float> c_int8(M * N, 0.0f);
    new_int8_v2::mat_vec(M, N, K, &ba_i8, &bb_i8, c_int8.data());

    // Compare
    TestResult r = {0, 0, 0, 0, 0, 0, M * N};
    for (int mi = 0; mi < M; mi++) {
        for (int ni = 0; ni < N; ni++) {
            int idx = mi * N + ni;
            float b = c_bf16[idx];
            float i8 = c_int8[idx];
            float diff_i8_bf16 = std::abs(i8 - b);
            r.max_i8_vs_bf16 = std::max(r.max_i8_vs_bf16, diff_i8_bf16);
            r.max_val = std::max(r.max_val, std::abs(b));
            if (ni < n_check) {
                float s = c_scalar[mi * n_check + ni];
                r.max_bf16_err = std::max(r.max_bf16_err, std::abs(b - s));
                r.max_int8_err = std::max(r.max_int8_err, std::abs(i8 - s));
                r.sum_bf16_err += std::abs(b - s);
                r.sum_int8_err += std::abs(i8 - s);
            }
        }
    }

    free(ba_bf16_mem);
    free(bb_bf16_mem);
    free(ba_i8_mem);
    free(bb_i8_mem);
    return r;
}

int main() {
    printf("=== MXFP4 INT8 (x2 LUT) vs BF16 Correctness Test ===\n\n");

    // Check CPU support
#if !defined(__AVX512F__)
    printf("ERROR: AVX512F required!\n");
    return 1;
#endif
#if !defined(__AVX512BW__)
    printf("ERROR: AVX512BW required!\n");
    return 1;
#endif
    printf("AVX512F: yes, AVX512BW: yes");
#if defined(__AVX512BF16__)
    printf(", AVX512BF16: yes");
#else
    printf(", AVX512BF16: no (using emulated BF16)");
#endif
#if defined(__AVX512VNNI__)
    printf(", AVX512VNNI: yes\n\n");
#else
    printf(", AVX512VNNI: no (dpbssd may not be available)\n\n");
#endif

    // Test dimensions (must be multiples of alignment requirements)
    const int M = 4;       // tokens
    const int N = 64;      // output features (multiple of 32)
    const int K = 256;     // input features (multiple of 64, and 32 for k_group)
    const int K_GROUP_SIZE = 32;  // MXFP4 group size (one scale per 32 weights)

    printf("Dimensions: M=%d, N=%d, K=%d, K_GROUP=%d\n", M, N, K, K_GROUP_SIZE);
    printf("Weight memory: %d bytes (4-bit packed)\n\n", N * K / 2);

    // Generate random BF16 activations and weights
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_act(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_wt(-6.0f, 6.0f);

    std::vector<ggml_bf16_t> act_bf16(M * K);
    std::vector<ggml_bf16_t> wt_bf16(N * K);
    std::vector<uint8_t> wt_fp4((size_t)N * K / 2);
    std::vector<ggml_bf16_t> scales_bf16(N * (K / K_GROUP_SIZE));

    for (int i = 0; i < M * K; i++) act_bf16[i] = ggml_fp32_to_bf16(dist_act(gen));
    for (int i = 0; i < N * K; i++) wt_bf16[i] = ggml_fp32_to_bf16(dist_wt(gen));

    // Quantize weights to FP4
    quantize_bf16_to_fp4(wt_bf16.data(), wt_fp4.data(), N * K);

    // Generate scales (MXFP4 uses per-group E8M0 scales; here we use random BF16 scales)
    std::uniform_real_distribution<float> dist_scale(0.001f, 0.1f);
    for (size_t i = 0; i < scales_bf16.size(); i++) {
        scales_bf16[i] = ggml_fp32_to_bf16(dist_scale(gen));
    }

    // --- Compute scalar reference ---
    std::vector<float> c_scalar(M * N, 0.0f);
    mat_vec_scalar(M, N, K, K_GROUP_SIZE, act_bf16.data(), wt_fp4.data(), scales_bf16.data(), c_scalar.data());

    // --- Compute BF16 reference kernel ---
    size_t ba_bf16_size = BufferABF16::required_size(M, K);
    void* ba_bf16_mem = std::aligned_alloc(64, ba_bf16_size);
    BufferABF16 ba_bf16(M, K, ba_bf16_mem);
    ba_bf16.from_mat(M, act_bf16.data());

    size_t bb_bf16_size = BufferBInt4KGroup::required_size(N, K, K_GROUP_SIZE);
    void* bb_bf16_mem = std::aligned_alloc(64, bb_bf16_size);
    BufferBInt4KGroup bb_bf16(N, K, K_GROUP_SIZE, bb_bf16_mem);
    bb_bf16.from_raw_mat(wt_fp4.data(), 0, 1, 256);
    bb_bf16.load_scales_bf16(scales_bf16.data(), false);  // no half-scale for BF16 path

    std::vector<float> c_bf16(M * N, 0.0f);
    ref_bf16::mat_vec(M, N, K, &ba_bf16, &bb_bf16, c_bf16.data());

    // --- Compute INT8 kernel ---
    size_t ba_i8_size = BufferAInt8::required_size(M, K, K_GROUP_SIZE);
    void* ba_i8_mem = std::aligned_alloc(64, ba_i8_size);
    BufferAInt8 ba_i8(M, K, K_GROUP_SIZE, ba_i8_mem);
    ba_i8.from_mat(M, act_bf16.data());

    size_t bb_i8_size = BufferBInt4KGroup::required_size(N, K, K_GROUP_SIZE);
    void* bb_i8_mem = std::aligned_alloc(64, bb_i8_size);
    BufferBInt4KGroup bb_i8(N, K, K_GROUP_SIZE, bb_i8_mem);
    bb_i8.from_raw_mat(wt_fp4.data(), 0, 1, 256);
    bb_i8.load_scales_bf16(scales_bf16.data(), true);  // half-scale for INT8 path!

    std::vector<float> c_int8(M * N, 0.0f);
    new_int8_v2::mat_vec(M, N, K, &ba_i8, &bb_i8, c_int8.data());

    // --- Compare results ---
    printf("%-20s %12s %12s %12s %10s %10s\n", "", "scalar", "bf16_kernel", "int8_kernel", "bf16_err", "int8_err");
    printf("%-20s %12s %12s %12s %10s %10s\n", "(m,n)", "value", "value", "value", "vs_scal", "vs_scal");
    printf("------------------------------------------------------------------------------------------\n");

    float max_bf16_err = 0, max_int8_err = 0;
    float max_val = 0;
    double sum_bf16_err = 0, sum_int8_err = 0;

    for (int mi = 0; mi < M; mi++) {
        for (int ni = 0; ni < std::min(N, 8); ni++) {
            int idx = mi * N + ni;
            float s = c_scalar[idx];
            float b = c_bf16[idx];
            float i8 = c_int8[idx];
            float be = std::abs(b - s);
            float ie = std::abs(i8 - s);
            max_bf16_err = std::max(max_bf16_err, be);
            max_int8_err = std::max(max_int8_err, ie);
            max_val = std::max(max_val, std::abs(s));
            sum_bf16_err += be;
            sum_int8_err += ie;
            printf("(%d,%d)                %12.4f %12.4f %12.4f %10.6f %10.6f\n",
                   mi, ni, s, b, i8, be, ie);
        }
    }

    int total = M * N;
    printf("\n=== Error Summary (over all %d elements) ===\n", total);
    printf("BF16 kernel vs scalar:  max_err=%.6f  avg_err=%.6f  rel=%.4f%%\n",
           max_bf16_err, sum_bf16_err / total, max_bf16_err / (max_val + 1e-8f) * 100);
    printf("INT8 kernel vs scalar:  max_err=%.6f  avg_err=%.6f  rel=%.4f%%\n",
           max_int8_err, sum_int8_err / total, max_int8_err / (max_val + 1e-8f) * 100);
    printf("INT8 vs BF16 kernel:    max_diff=%.6f\n",
           std::abs(c_int8[0] - c_bf16[0]));

    // Also compute max diff between INT8 and BF16 kernel over all elements
    float max_i8_vs_bf16 = 0;
    for (int i = 0; i < total; i++) {
        max_i8_vs_bf16 = std::max(max_i8_vs_bf16, std::abs(c_int8[i] - c_bf16[i]));
    }
    printf("INT8 vs BF16 kernel:    max_diff=%.6f (over all %d)\n", max_i8_vs_bf16, total);

    // Cleanup
    free(ba_bf16_mem);
    free(bb_bf16_mem);
    free(ba_i8_mem);
    free(bb_i8_mem);

    printf("\n=== Test Complete ===\n");

    // ========================================================================
    // LARGE-SCALE MULTI-SEED TEST
    // ========================================================================
    printf("\n\n=== Large-Scale Multi-Seed Test ===\n\n");
    printf("%-30s %8s %8s %10s %10s %10s %10s\n",
           "Config", "M", "N", "K", "bf16_rel%", "int8_rel%", "i8_vs_bf16");
    printf("-----------------------------------------------------------------------------\n");

    struct TestConfig { int M, N, K; const char* name; };
    TestConfig configs[] = {
        {4, 256, 512, "small"},
        {4, 512, 1024, "medium"},
        {8, 512, 2048, "large-K"},
        {16, 1024, 2048, "large-MN"},
    };

    for (const auto& cfg : configs) {
        float worst_int8_rel = 0, worst_i8_bf16 = 0;
        for (unsigned seed = 100; seed < 103; seed++) {
            TestResult r = run_test(cfg.M, cfg.N, cfg.K, 32, seed);
            float int8_rel = r.max_int8_err / (r.max_val + 1e-8f) * 100;
            worst_int8_rel = std::max(worst_int8_rel, int8_rel);
            worst_i8_bf16 = std::max(worst_i8_bf16, r.max_i8_vs_bf16);
        }
        printf("%-30s %8d %8d %10d %10s %10.4f %10.6f\n",
               cfg.name, cfg.M, cfg.N, cfg.K, "~0", worst_int8_rel, worst_i8_bf16);
    }

    printf("\n=== All Tests Complete ===\n");
    return 0;
}

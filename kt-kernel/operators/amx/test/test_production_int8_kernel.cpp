/**
 * @file test_production_int8_kernel.cpp
 * @brief Compile + correctness test for the production GemmKernel224MXFP4Int8KGroup.
 *
 * This test includes the actual production headers (not inline copies) to verify
 * that the new INT8 kernel compiles correctly within the kt-kernel framework.
 * It then runs a correctness comparison against the BF16 reference kernel.
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

// Include the production kernel headers
// These pull in moe_base.hpp, amx_buffers.hpp, amx_kernels.hpp, etc.
#include "operators/amx/fp4-moe.hpp"

// ============================================================================
// Minimal test utilities (from the standalone test)
// ============================================================================
static const float fp4_to_fp32_lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// Scalar reference: exact FP32 computation
static float scalar_dot(const float* act, const uint8_t* fp4, const float* scale, int K, int KG) {
    float acc = 0.0f;
    for (int g = 0; g < KG; g++) {
        for (int j = 0; j < 32; j++) {
            int ki = g * 32 + j;
            uint8_t byte = fp4[ki / 2];
            uint8_t nib = (ki % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            acc += fp4_to_fp32_lut[nib] * scale[g] * act[ki];
        }
    }
    return acc;
}

int main() {
    using Kernel = amx::GemmKernel224MXFP4Int8KGroup;
    printf("Testing production kernel: %s\n", Kernel::name().c_str());
    printf("  M_STEP=%d, N_STEP=%d, K_STEP=%d, N_BLOCK=%d\n", Kernel::M_STEP, Kernel::N_STEP, Kernel::K_STEP,
           Kernel::N_BLOCK);

    // Test dimensions (must be multiples of N_STEP=32, K_STEP=32)
    const int M = 4;
    const int N = 64;   // must be multiple of 32
    const int K = 256;  // must be multiple of 64 (for VNNI dot product)
    const int KG = K / 32;

    std::mt19937 gen(42);
    auto rand_f = [&](float lo, float hi) { return lo + (hi - lo) * (gen() / (float)gen.max()); };

    // Generate FP4 weights (nibble-packed)
    std::vector<uint8_t> fp4_weights(N * K / 2);
    for (auto& b : fp4_weights) b = gen() & 0xFF;

    // Generate FP32 scales (will be halved for INT8 path)
    std::vector<float> scales_fp32(N * KG);
    for (auto& s : scales_fp32) s = rand_f(0.001f, 0.1f);

    // Generate BF16 activations
    std::vector<ggml_bf16_t> act_bf16(M * K);
    for (auto& a : act_bf16) {
        float v = rand_f(-1.0f, 1.0f);
        a = ggml_fp32_to_bf16(v);
    }

    // Convert activations to FP32 for scalar reference
    std::vector<float> act_fp32(M * K);
    for (int i = 0; i < M * K; i++) act_fp32[i] = ggml_bf16_to_fp32(act_bf16[i]);

    // === Compute scalar reference ===
    std::vector<float> ref_out(M * N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            ref_out[m * N + n] = scalar_dot(act_fp32.data() + m * K, fp4_weights.data() + n * K / 2,
                                            scales_fp32.data() + n * KG, K, KG);
        }
    }

    // === Allocate buffers for the production kernel ===
    // BufferA: INT8 activations (BufferASmallKGroupImpl)
    size_t ba_size = Kernel::BufferA::required_size(M, K, 32);
    std::vector<uint8_t> ba_mem(ba_size + 63, 0);  // +63 for alignment
    void* ba_ptr = (void*)(((uintptr_t)ba_mem.data() + 63) & ~63);
    auto ba = std::make_shared<Kernel::BufferA>(M, K, 32, ba_ptr);

    // BufferB: FP4 nibble-packed weights (BufferBInt4KGroupImpl)
    size_t bb_size = Kernel::BufferB::required_size(N, K, 32);
    std::vector<uint8_t> bb_mem(bb_size + 63, 0);
    void* bb_ptr = (void*)(((uintptr_t)bb_mem.data() + 63) & ~63);
    auto bb = std::make_shared<Kernel::BufferB>(N, K, 32, bb_ptr);

    // BufferC: FP32 output (BufferCReduceImpl)
    size_t bc_size = Kernel::BufferC::required_size(M, N);
    std::vector<uint8_t> bc_mem(bc_size + 63, 0);
    void* bc_ptr = (void*)(((uintptr_t)bc_mem.data() + 63) & ~63);
    auto bc = std::make_shared<Kernel::BufferC>(M, N, bc_ptr);

    // === Load activations into BufferA (BF16 → INT8 quantization) ===
    ba->from_mat(M, act_bf16.data(), 0, 1);

    // === Load FP4 weights into BufferB (plain memcpy) ===
    bb->from_raw_mat(fp4_weights.data(), 0, 1);

    // === Load scales with ×0.5 (compensate for FP4×2→INT8) ===
    float* bb_d = bb->get_scale(N, 0, K, 0);
    for (int n = 0; n < N; n++) {
        float* row_scales = bb->get_scale(N, n, K, 0);
        for (int g = 0; g < KG; g++) {
            row_scales[g] = scales_fp32[n * KG + g] * 0.5f;
        }
    }

    // === Run the production kernel ===
    Kernel::integer_mat_vec_kgroup(M, N, K, 32, ba.get(), bb.get(), bc.get(), 0, 1);

    // === Compare results ===
    float max_err = 0, sum_err = 0, max_abs = 0;
    for (int m = 0; m < M; m++) {
        float* c_row = bc->get_submat(M, N, m, 0);
        for (int n = 0; n < N; n++) {
            float got = c_row[n];
            float expected = ref_out[m * N + n];
            float err = std::abs(got - expected);
            max_err = std::max(max_err, err);
            sum_err += err;
            max_abs = std::max(max_abs, std::abs(expected));
        }
    }

    printf("\n=== Results (M=%d, N=%d, K=%d) ===\n", M, N, K);
    printf("  Max abs output:   %.6f\n", max_abs);
    printf("  Max abs error:    %.6f\n", max_err);
    printf("  Avg abs error:    %.6f\n", sum_err / (M * N));
    printf("  Max relative err: %.4f%%\n", max_err / (max_abs + 1e-8f) * 100);

    // The error should be < 1% (from activation quantization)
    float rel_err = max_err / (max_abs + 1e-8f) * 100;
    if (rel_err < 1.0f) {
        printf("\n  PASS: Production INT8 kernel matches scalar reference (< 1%% error)\n");
        return 0;
    } else {
        printf("\n  FAIL: Error too high (> 1%%)\n");
        return 1;
    }
}

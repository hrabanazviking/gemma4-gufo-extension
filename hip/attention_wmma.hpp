#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_

#include <hip/hip_runtime.h>

#include "hip/kernels.hpp"

namespace gufo::models::gemma4::rocm {

/// Prefill attention on the WMMA matrix cores (binary16 operands, FP32
/// accumulation) for head_dim 256 and 512 with an even number of query heads
/// per KV head. Returns false without launching for other geometries.
/// KV-12: a q8_0 cache (AttentionArgs' *_cache_q8 set) is taken through
/// the kernel's dequant-tile stage for the shapes it stages exactly;
/// other q8_0 shapes return false and keep the non-WMMA road.
[[nodiscard]] bool LaunchWmmaPrefillAttention(const AttentionArgs& args,
                                              hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_

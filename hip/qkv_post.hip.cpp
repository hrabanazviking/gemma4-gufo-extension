/// QkvPost in its own translation unit, compiled with the strict
/// floating-point options (see extension.cmake): R3-E1 part 2
/// LDS-staged reproduction in attention_wmma.hip.cpp recomputes
/// QkvPost norm / rope / q8_0 quantization, and the two must agree
/// bit for bit. Under the fast-math options of kernels.hip.cpp the
/// reproduction differed by +/-1 code / +/-1 ulp scale on real data
/// (KV_PROGRESS.md, R3-E1 part 2 root cause). The wave helpers, RMS
/// scale, rope angle and q8_0 layout helpers below are private
/// copies of the ones in kernels.hip.cpp -- arithmetic, order and
/// rounding unchanged.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "hip/kernels.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;

/// v from lane (lane ^ M) through DPP or a cross-row permute instead of an
/// LDS permute.
template<int M>
__device__ __forceinline__ float XorLane(float v) {
  static_assert(M > 0 && M < kWave && (M & (M - 1)) == 0);
  const int x = __float_as_int(v);
  if constexpr (M == 16) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        x, x, 0x76543210, 0xfedcba98, false, false));
  } else {
    // row_xmask:M. Every source lane exists; mov_dpp lets the compiler
    // fold the exchange into the following add without an old-value copy.
    return __int_as_float(
        __builtin_amdgcn_mov_dpp(x, 0x160 | M, 0xf, 0xf, true));
  }
}

template<int M>
__device__ __forceinline__ float MaxXorLane(float v) {
  static_assert(M > 0 && M < kWave && (M & (M - 1)) == 0);
  if constexpr (M == 16) {
    return fmaxf(v, XorLane<16>(v));
  } else {
    float out;
    asm("v_max_f32_dpp %0, %1, %1 row_xmask:%2 row_mask:0xf bank_mask:0xf "
        "bound_ctrl:1"
        : "=v"(out)
        : "v"(v), "n"(M));
    return out;
  }
}

__device__ __forceinline__ float WaveMax(float v) {
  v = MaxXorLane<16>(v);
  v = MaxXorLane<8>(v);
  v = MaxXorLane<4>(v);
  v = MaxXorLane<2>(v);
  return MaxXorLane<1>(v);
}

__device__ __forceinline__ float WaveSum(float v) {
  v += XorLane<16>(v);
  v += XorLane<8>(v);
  v += XorLane<4>(v);
  v += XorLane<2>(v);
  return v + XorLane<1>(v);
}

__device__ inline float RmsScale(float sum_squares, std::uint32_t dim,
                                 float eps) {
  return rsqrtf(sum_squares / static_cast<float>(dim) + eps);
}

/// mirror is kv_cache_format.hpp KvQ8Layout / Q8_0PayloadOffset /
/// Q8_0ScaleOffset): a row of `values` values (always a multiple
/// of 32 on the device paths) stores its int8 payload contiguously
/// -- 32 bytes per 32-value block, blocks in row order -- then a
/// separate fp16 scale plane, 2 bytes per block, block index =
/// head * blocks-per-head + d_block. Total row bytes are unchanged
/// from layout v1 (blocks * 34).
__device__ __forceinline__ std::size_t Q8RowBytes(std::size_t values) {
  return (values / 32) * 34;
}
__device__ __forceinline__ std::uint8_t* Q8BlockCodes(std::uint8_t* row,
                                                      std::size_t block) {
  return row + block * 32;
}
__device__ __forceinline__ const std::uint8_t* Q8BlockCodes(
    const std::uint8_t* row, std::size_t block) {
  return row + block * 32;
}
__device__ __forceinline__ std::uint8_t* Q8BlockScale(std::uint8_t* row,
                                                      std::size_t values,
                                                      std::size_t block) {
  return row + (values / 32) * 32 + block * 2;
}
__device__ __forceinline__ const std::uint8_t* Q8BlockScale(
    const std::uint8_t* row, std::size_t values, std::size_t block) {
  return row + (values / 32) * 32 + block * 2;
}

/// The angle pair i turns by at `position`.
__device__ inline float RopeAngle(std::uint32_t position, std::uint32_t i,
                                  float theta_scale,
                                  const float* freq_factors) {
  float theta =
      static_cast<float>(position) * powf(theta_scale, static_cast<float>(i));
  if (freq_factors != nullptr) {
    theta /= freq_factors[i];
  }
  return theta;
}

/// R3-E1 part 1: the plain (non-rotated) q8_0 row store of
/// QkvPostKernel, factored into a shared device helper in
/// QkvPost's own one-wave-per-(row, head) mapping. One 32-value
/// block lies in one wave: lane holds dim 32 * (w + kWaves * k) +
/// lane of this head, so WaveMax gives the block amax and each
/// lane writes its own code; lane 0 writes the fp16 scale.
/// Matches QuantizeQ8_0Row on the host. The K-plain (KV-05) and
/// V (KV-06) stores of QkvPostKernel both call this; the rotated-K
/// store (KV-09) keeps its own packed form. A future R3 slice
/// reuses this mapping for the LDS-staged reproduction.
template<int D>
__device__ __forceinline__ void StoreQ8PlainBlocks(
    const float (&v)[kWaves][D / kThreads], int lane, std::uint32_t kvh,
    std::uint32_t slot, std::uint8_t* cache_q8, std::uint32_t kv_heads) {
  constexpr int K = D / kThreads;
  const std::size_t row_values = static_cast<std::size_t>(kv_heads) * D;
  const std::size_t row_bytes = Q8RowBytes(row_values);
  std::uint8_t* row_q8 = cache_q8 + static_cast<std::size_t>(slot) * row_bytes;
  constexpr std::uint32_t kBlocksPerHead = D / 32;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const std::uint32_t blk = kvh * kBlocksPerHead + w + kWaves * k;
      std::uint8_t* out = Q8BlockCodes(row_q8, blk);
      const float amax = WaveMax(fabsf(v[w][k]));
      const float dscale = amax / 127.0F;
      const float inv = dscale != 0.0F ? 1.0F / dscale : 0.0F;
      out[lane] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(v[w][k] * inv)));
      if (lane == 0) {
        *reinterpret_cast<__half*>(Q8BlockScale(row_q8, row_values, blk)) =
            __float2half(dscale);
      }
    }
  }
}

/// One wave per (row, head), eight heads of a row per block. Lane L holds
/// dims 32 w + L + 256 k (w < 8, k < D / 256): exactly the elements thread
/// 32 w + L held in the block-per-head form, so the per-thread partial sums,
/// the per-wave butterflies and the in-order sum over w reproduce BlockSum,
/// and every rotation pair (d, d + D / 2) lies in one lane.
template<int D>
__global__ void __launch_bounds__(kThreads) QkvPostKernel(QkvPostArgs a) {
  constexpr int K = D / kThreads;  // elements per former thread
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y * kWaves + threadIdx.x / kWave;
  if (head >= a.heads + a.kv_heads) {
    return;
  }
  const std::uint32_t chain = a.rows - a.siblings;
  const bool sibling = row >= chain;
  const std::uint32_t position =
      sibling ? a.first_position + 1 + (row - chain) : a.first_position + row;
  const auto dim_of = [&](int w, int k) {
    return static_cast<std::uint32_t>(kWave * w + lane + kThreads * k);
  };
  // v = rms(v) * weight (BlockSum order), then (optionally) the rotation.
  const auto norm_rope = [&](float (&v)[kWaves][K], const float* weight,
                             bool rope) {
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
      float ss = 0.0F;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        ss += v[w][k] * v[w][k];
      }
      total += WaveSum(ss);
    }
    const float r = RmsScale(total, D, a.eps);
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const float x = v[w][k] * r;
        v[w][k] = weight != nullptr ? x * weight[dim_of(w, k)] : x;
      }
    }
    if (!rope) {
      return;
    }
    // The partner of dim d < D / 2 is d + D / 2 = (w + kDw, k + kDk).
    constexpr int kDw = (D / 2) % kThreads / kWave;
    constexpr int kDk = (D / 2) / kThreads;
#pragma unroll
    for (int w = 0; w + kDw < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k + kDk < K; ++k) {
        const std::uint32_t d = dim_of(w, k);
        if (d >= D / 2) {
          continue;
        }
        const float theta =
            RopeAngle(position, d, a.theta_scale, a.freq_factors);
        const float c = __ocml_cos_f32(theta);
        const float sn = __ocml_sin_f32(theta);
        const float lo = v[w][k];
        const float hi = v[w + kDw][k + kDk];
        v[w][k] = lo * c - hi * sn;
        v[w + kDw][k + kDk] = lo * sn + hi * c;
      }
    }
  };
  const auto load = [&](const float* src, float (&v)[kWaves][K]) {
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        v[w][k] = src[dim_of(w, k)];
      }
    }
  };
  float v[kWaves][K];
  const std::size_t q_row = a.row_stride != 0 ? a.row_stride : a.heads * D;
  const std::size_t kv_row = a.row_stride != 0 ? a.row_stride : a.kv_heads * D;
  const std::size_t v_row = a.v_stride != 0 ? a.v_stride : a.kv_heads * D;
  if (head < a.heads) {
    float* q = a.q + row * q_row + std::size_t{head} * D;
    load(q, v);
    if (a.q_out != nullptr) {
      q = a.q_out + (static_cast<std::size_t>(row) * a.heads + head) * D;
    }
    norm_rope(v, a.q_norm, true);
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        // Derived-key layers: the query carries k_norm on unrotated dims.
        const std::uint32_t d = dim_of(w, k);
        if (a.rotated_pairs != 0 && d % (D / 2) >= a.rotated_pairs) {
          v[w][k] *= a.k_norm[d];
        }
        q[d] = v[w][k];
      }
    }
    return;
  }
  const std::uint32_t kvh = head - a.heads;
  const std::size_t in = row * kv_row + std::size_t{kvh} * D;
  const std::uint32_t key = sibling ? a.spare_key + (row - chain) : position;
  const std::uint32_t slot = a.ring != 0 ? key % a.ring : key;
  const std::size_t cache =
      (static_cast<std::size_t>(slot) * a.kv_heads + kvh) * D;
  auto* k_cache = reinterpret_cast<__half*>(a.k_cache);
  auto* v_cache = reinterpret_cast<__half*>(a.v_cache);
  load(a.k + in, v);
  norm_rope(v, a.k_norm, true);
  const std::uint32_t pairs = a.rotated_pairs;
  const std::size_t rot_at =
      (static_cast<std::size_t>(slot) * a.kv_heads + kvh) * 2 * pairs;
  if (a.k_cache_q8 != nullptr && pairs == 0) {
    // KV-05: q8_0 K store, via the shared StoreQ8PlainBlocks
    // helper (R3-E1 part 1; arithmetic unchanged).
    StoreQ8PlainBlocks<D>(v, lane, kvh, slot, a.k_cache_q8, a.kv_heads);
  } else if (a.k_cache_q8 != nullptr && pairs != 0) {
    // KV-09: q8_0 rotated-K store for derived-key layers. The packed
    // row holds, per head, the rotated dims [0, pairs) then
    // [D / 2, D / 2 + pairs), quantized after the rotation. With
    // pairs a multiple of 32, segment k (0 = low dims, 1 = high
    // dims) and wave w < pairs / 32 hold exactly packed block
    // k * (pairs / 32) + w of this head -- one 32-value block per
    // wave, as in the KV-05 store -- so WaveMax gives the block
    // amax, each lane writes its own code, lane 0 the fp16 scale.
    // Matches QuantizeQ8_0Row over the packed row on the host.
    const std::uint32_t seg_blocks = pairs / 32;
    const std::size_t row_values =
        static_cast<std::size_t>(a.kv_heads) * 2 * pairs;
    const std::size_t row_bytes = Q8RowBytes(row_values);
    std::uint8_t* row_q8 =
        a.k_cache_q8 + static_cast<std::size_t>(slot) * row_bytes;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if (static_cast<std::uint32_t>(w) < seg_blocks) {
          const std::uint32_t blk = kvh * 2 * seg_blocks + k * seg_blocks + w;
          std::uint8_t* out = Q8BlockCodes(row_q8, blk);
          const float amax = WaveMax(fabsf(v[w][k]));
          const float dscale = amax / 127.0F;
          const float inv = dscale != 0.0F ? 1.0F / dscale : 0.0F;
          out[lane] = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(v[w][k] * inv)));
          if (lane == 0) {
            *reinterpret_cast<__half*>(Q8BlockScale(row_q8, row_values, blk)) =
                __float2half(dscale);
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const std::uint32_t d = dim_of(w, k);
        if (pairs == 0) {
          k_cache[cache + d] = __float2half(v[w][k]);
        } else if (d % (D / 2) < pairs) {
          // Rotated dims only: [0, pairs) then [D / 2, D / 2 + pairs).
          k_cache[rot_at + (d < D / 2 ? d : pairs + d - D / 2)] =
              __float2half(v[w][k]);
        }
      }
    }
  }
  load(a.v + row * v_row + std::size_t{kvh} * D, v);
  norm_rope(v, nullptr, false);
  if (a.v_cache_q8 != nullptr) {
    // KV-06: q8_0 V store, via the shared StoreQ8PlainBlocks
    // helper (R3-E1 part 1; arithmetic unchanged). KV-09:
    // derived-key layers take this branch too -- their V row is
    // the normalized K projection, stored exactly like a sliding
    // V row.
    StoreQ8PlainBlocks<D>(v, lane, kvh, slot, a.v_cache_q8, a.kv_heads);
  } else {
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        v_cache[cache + dim_of(w, k)] = __float2half(v[w][k]);
      }
    }
  }
}

}  // namespace

void QkvPost(const QkvPostArgs& args, hipStream_t stream) {
  const dim3 grid(args.rows,
                  (args.heads + args.kv_heads + kWaves - 1) / kWaves);
  switch (args.head_dim) {
    case 256:
      QkvPostKernel<256><<<grid, kThreads, 0, stream>>>(args);
      return;
    case 512:
      QkvPostKernel<512><<<grid, kThreads, 0, stream>>>(args);
      return;
    default:
      throw std::invalid_argument("QkvPost supports head_dim 256 and 512");
  }
}

}  // namespace gufo::models::gemma4::rocm

#ifndef GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_
#define GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gufo::models::gemma4 {

/// Storage format of one session's KV cache. The format is a property
/// of the cache: allocation, the write path, and every read path honor
/// the same value (KV Q8_0 campaign, campaign/KV_Q8_ROADMAP.md).
enum class KvCacheFormat : std::uint32_t {
  /// Binary16 rows, the original layout. Always available.
  kF16,
  /// q8_0 blocks: 32 values as 32 int8 codes + one fp16 scale
  /// (34 bytes per 32 values), stored in the split-plane layout
  /// versioned by KvQ8Layout below (kKvQ8LayoutCurrent).
  kQ8_0,
  /// Reserved. ParseKvCacheType rejects it until a q4_0 campaign
  /// builds its write/read paths; the value exists so the format
  /// field never needs renumbering.
  kQ4_0,
};

/// Short name of a format, as accepted by --kv-cache-type /
/// GUFO_KV_CACHE_TYPE: "f16", "q8_0", "q4_0".
[[nodiscard]] inline std::string_view KvCacheFormatName(
    KvCacheFormat format) noexcept {
  switch (format) {
    case KvCacheFormat::kF16:
      return "f16";
    case KvCacheFormat::kQ8_0:
      return "q8_0";
    case KvCacheFormat::kQ4_0:
      return "q4_0";
  }
  return "unknown";
}

/// Physical layout version of a q8_0 KV cache (Phase 5 REDO R1
/// audit, R2-global step 1). v1 was the interleaved ggml layout:
/// each 32-value block stored its fp16 scale inline at bytes
/// [32, 34) of a 34-byte block. v2 splits every row into a
/// contiguous int8 payload plane (32 bytes per block, blocks in
/// row order -- per (head, token, d)) followed by a separate fp16
/// scale plane indexed [token][head][d_block] within the row.
/// Total row bytes are identical; only the arrangement changed.
/// The version is a property of the format: every writer and
/// reader in this tree implements exactly kKvQ8LayoutCurrent and
/// caches never outlive the process, so there is no migration --
/// the flag exists so a future layout change is an explicit
/// version bump, never a silent reinterpretation of stored bytes.
enum class KvQ8Layout : std::uint32_t {
  kInterleavedV1 = 1,
  kSplitScaleV2 = 2,
};

inline constexpr KvQ8Layout kKvQ8LayoutCurrent = KvQ8Layout::kSplitScaleV2;

/// Layout version a cache in `format` is stored in: 0 for formats
/// with no q8_0 layout, else the current q8_0 layout version.
[[nodiscard]] inline std::uint32_t KvCacheQ8LayoutVersion(
    KvCacheFormat format) noexcept {
  return format == KvCacheFormat::kQ8_0
             ? static_cast<std::uint32_t>(kKvQ8LayoutCurrent)
             : 0;
}

/// --kv-cache-type / GUFO_KV_CACHE_TYPE: f16 or q8_0; empty is f16.
/// Throws std::invalid_argument naming the offending value: q4_0 is
/// reserved and not implemented, anything else is unknown. Never a
/// silent fallback.
[[nodiscard]] inline KvCacheFormat ParseKvCacheType(std::string_view name) {
  if (name.empty() || name == "f16") {
    return KvCacheFormat::kF16;
  }
  if (name == "q8_0") {
    return KvCacheFormat::kQ8_0;
  }
  if (name == "q4_0") {
    throw std::invalid_argument(
        "KV cache type 'q4_0' is reserved and not implemented; "
        "use f16 or q8_0");
  }
  throw std::invalid_argument("unknown KV cache type '" + std::string(name) +
                              "'; use f16 or q8_0");
}

/// Bytes of one cache row of `values` values in `format`. q8_0 rows
/// pad a partial tail block (every real row width is a multiple of
/// 32; the padding rule keeps the function total). The q4_0 figure
/// is the reserved ggml layout (16 bytes of packed nibbles + fp16
/// scale per 32 values); it is unreachable until q4_0 is implemented.
[[nodiscard]] inline std::size_t KvCacheRowBytes(std::uint32_t values,
                                                 KvCacheFormat format) {
  switch (format) {
    case KvCacheFormat::kF16:
      return std::size_t{values} * 2;
    case KvCacheFormat::kQ8_0:
      // Layout v2: payload plane (32 int8 bytes per block) plus
      // scale plane (one fp16 per block) -- the same total as the
      // v1 interleaved 34-byte blocks, differently arranged.
      return (std::size_t{values} + 31) / 32 * (32 + 2);
    case KvCacheFormat::kQ4_0:
      return (std::size_t{values} + 31) / 32 * 18;
  }
  throw std::invalid_argument("unknown KV cache format");
}

/// fp16 (binary16) bit conversion, for q8_0 block scales. Standard
/// round-to-nearest-even encoding; subnormals handled both ways.
[[nodiscard]] inline std::uint16_t FloatToFp16Bits(float value) noexcept {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 112;
  const std::uint32_t mant = bits & 0x7FFFFFU;
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (mant != 0 ? 0x200U : 0U));
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t m = mant | 0x800000U;
    const int shift = 14 - exp;
    std::uint32_t half = m >> shift;
    const std::uint32_t rest = m & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1U) != 0)) {
      ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rest = mant & 0x1FFFU;
  if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0)) {
    ++half;
  }
  return static_cast<std::uint16_t>(sign | half);
}

[[nodiscard]] inline float Fp16BitsToFloat(std::uint16_t half) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(half & 0x8000U) << 16);
  const std::uint32_t exp = (half >> 10) & 0x1FU;
  const std::uint32_t mant = half & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mant << 13);
  } else if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      std::uint32_t m = mant;
      std::int32_t e = -14;
      while ((m & 0x400U) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFU;
      bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
    }
  } else {
    bits = sign | ((exp + 112U) << 23) | (mant << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// q8_0 block geometry: 32 values per block; a block occupies 32
/// payload bytes plus a 2-byte fp16 scale (34 bytes total). Since
/// layout v2 the two parts are stored SPLIT: a row is its payload
/// plane (all blocks' codes, contiguous, in row order) followed
/// by its scale plane (all blocks' fp16 scales, in row order).
inline constexpr std::uint32_t kQ8_0BlockValues = 32;
inline constexpr std::uint32_t kQ8_0BlockBytes = 34;
inline constexpr std::uint32_t kQ8_0PayloadBytes = 32;
inline constexpr std::uint32_t kQ8_0ScaleBytes = 2;

/// Number of q8_0 blocks in a row of `values` values.
[[nodiscard]] inline std::size_t Q8_0BlockCount(std::size_t values) noexcept {
  return (values + kQ8_0BlockValues - 1) / kQ8_0BlockValues;
}

/// Byte offset of block `block`'s payload within its row (layout
/// v2 payload plane; the row's value count does not affect it).
[[nodiscard]] inline std::size_t Q8_0PayloadOffset(std::size_t values,
                                                   std::size_t block) noexcept {
  static_cast<void>(values);
  return block * kQ8_0PayloadBytes;
}

/// Byte offset of block `block`'s fp16 scale within a row of
/// `values` values (layout v2 scale plane, after the whole
/// payload plane of the row).
[[nodiscard]] inline std::size_t Q8_0ScaleOffset(std::size_t values,
                                                 std::size_t block) noexcept {
  return Q8_0BlockCount(values) * kQ8_0PayloadBytes + block * kQ8_0ScaleBytes;
}

/// Quantizes one 32-value block: the 32 int8 codes go to `codes`
/// and the fp16 scale bits to `*scale_bits`. scale = max|v| / 127;
/// a zero block stores a zero scale and codes. This is the host
/// reference for the QkvPost device stores (KV-05/KV-06); the
/// kernel computes the same values (its scale passes through
/// __float2half, which may differ by at most the fp16 rounding
/// the host conversion also applies).
inline void QuantizeQ8_0Block(const float* values, std::uint8_t* codes,
                              std::uint16_t* scale_bits) {
  float amax = 0.0F;
  for (std::uint32_t i = 0; i < kQ8_0BlockValues; ++i) {
    amax = std::max(amax, std::fabs(values[i]));
  }
  const float scale = amax / 127.0F;
  const float inv = scale != 0.0F ? 1.0F / scale : 0.0F;
  for (std::uint32_t i = 0; i < kQ8_0BlockValues; ++i) {
    codes[i] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(std::round(values[i] * inv)));
  }
  *scale_bits = FloatToFp16Bits(scale);
}

/// Inverse of QuantizeQ8_0Block.
inline void DequantizeQ8_0Block(const std::uint8_t* codes,
                                std::uint16_t scale_bits, float* values) {
  const float scale = Fp16BitsToFloat(scale_bits);
  for (std::uint32_t i = 0; i < kQ8_0BlockValues; ++i) {
    values[i] = static_cast<float>(static_cast<std::int8_t>(codes[i])) * scale;
  }
}

/// Quantizes a whole row of `values` values (a K or V cache row)
/// into `out`, which must hold KvCacheRowBytes(values, kQ8_0)
/// bytes, in the layout v2 planes: codes at Q8_0PayloadOffset,
/// scales at Q8_0ScaleOffset. A partial tail block is zero-padded
/// before quantizing.
inline void QuantizeQ8_0Row(std::span<const float> row, std::uint8_t* out) {
  float block[kQ8_0BlockValues];
  const std::size_t blocks = Q8_0BlockCount(row.size());
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t at = b * kQ8_0BlockValues;
    const std::size_t n =
        std::min<std::size_t>(kQ8_0BlockValues, row.size() - at);
    for (std::size_t i = 0; i < kQ8_0BlockValues; ++i) {
      block[i] = i < n ? row[at + i] : 0.0F;
    }
    std::uint16_t scale_bits = 0;
    QuantizeQ8_0Block(block, out + Q8_0PayloadOffset(row.size(), b),
                      &scale_bits);
    const std::size_t so = Q8_0ScaleOffset(row.size(), b);
    out[so] = static_cast<std::uint8_t>(scale_bits & 0xFFU);
    out[so + 1] = static_cast<std::uint8_t>(scale_bits >> 8);
  }
}

/// Inverse of QuantizeQ8_0Row for `values` values.
inline void DequantizeQ8_0Row(const std::uint8_t* in, float* row,
                              std::size_t values) {
  const std::size_t blocks = Q8_0BlockCount(values);
  for (std::size_t b = 0; b < blocks; ++b) {
    float block[kQ8_0BlockValues];
    const std::size_t so = Q8_0ScaleOffset(values, b);
    const auto scale_bits = static_cast<std::uint16_t>(
        in[so] | (static_cast<std::uint16_t>(in[so + 1]) << 8));
    DequantizeQ8_0Block(in + Q8_0PayloadOffset(values, b), scale_bits, block);
    const std::size_t at = b * kQ8_0BlockValues;
    const std::size_t n = std::min<std::size_t>(kQ8_0BlockValues, values - at);
    for (std::size_t i = 0; i < n; ++i) {
      row[at + i] = block[i];
    }
  }
}

/// The format one layer's regions use in a cache stored in
/// `format`. Since KV-09 every layer takes the cache format: sliding
/// rows store full K/V rows; global (derived-key) rows store the V
/// row (the normalized K projection) and the packed rotated-K dims
/// row, both as q8_0 blocks, and the attention paths reconstruct
/// K = rope(k_norm . V) from the dequantized stores (rotated dims
/// from their own store, every other dim its dequantized value --
/// rope is orthogonal, so the reconstruction adds no error of its
/// own). The write path (QkvPost), the allocation
/// (Executor::CreateCache/CacheBytes) and every read path agree on
/// this rule. (KV-07/KV-08 shipped a mixed cache -- sliding q8_0,
/// global binary16 -- through this same function.)
[[nodiscard]] inline KvCacheFormat KvCacheLayerFormat(KvCacheFormat format,
                                                      bool sliding) noexcept {
  static_cast<void>(sliding);
  return format;
}

/// One layer's cache shape for storage sizing: `slots` rows of
/// `key_values`-wide K and `v_values`-wide V.
struct KvLayerShape {
  std::uint32_t slots;
  std::uint32_t key_values;
  std::uint32_t v_values;
};

/// Region alignment of the cache allocation (matches the executor's
/// carve-up, hip/executor.cpp kAlign).
inline constexpr std::size_t kKvCacheRegionAlign = 256;

[[nodiscard]] inline std::size_t KvCacheAlignUp(std::size_t n) noexcept {
  return (n + kKvCacheRegionAlign - 1) / kKvCacheRegionAlign *
         kKvCacheRegionAlign;
}

/// Total K+V storage bytes for `layers` in `format`: each region is
/// aligned separately, exactly as the executor carves the allocation.
/// For kF16 this reproduces the pre-campaign CacheBytes loop
/// byte-for-byte.
[[nodiscard]] inline std::size_t KvCacheStorageBytes(
    std::span<const KvLayerShape> layers, KvCacheFormat format) {
  std::size_t bytes = 0;
  for (const KvLayerShape& layer : layers) {
    bytes += KvCacheAlignUp(std::size_t{layer.slots} *
                            KvCacheRowBytes(layer.key_values, format)) +
             KvCacheAlignUp(std::size_t{layer.slots} *
                            KvCacheRowBytes(layer.v_values, format));
  }
  return bytes;
}

/// Total K+V storage bytes for `layers` whose regions use the
/// per-layer formats in `formats` (parallel to `layers`), each
/// region aligned separately as the executor carves the allocation.
/// With every format equal this reproduces KvCacheStorageBytes.
[[nodiscard]] inline std::size_t KvCacheStorageBytesPerLayer(
    std::span<const KvLayerShape> layers,
    std::span<const KvCacheFormat> formats) {
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const KvLayerShape& layer = layers[i];
    bytes += KvCacheAlignUp(std::size_t{layer.slots} *
                            KvCacheRowBytes(layer.key_values, formats[i])) +
             KvCacheAlignUp(std::size_t{layer.slots} *
                            KvCacheRowBytes(layer.v_values, formats[i]));
  }
  return bytes;
}

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_

#ifndef GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_
#define GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_

#include <cstddef>
#include <cstdint>
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
  /// ggml-style q8_0 blocks: 32 values as 32 int8 + one fp16 scale
  /// (34 bytes per 32 values).
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
      return (std::size_t{values} + 31) / 32 * 34;
    case KvCacheFormat::kQ4_0:
      return (std::size_t{values} + 31) / 32 * 18;
  }
  throw std::invalid_argument("unknown KV cache format");
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

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_KV_CACHE_FORMAT_HPP_

#ifndef GUFO_MODELS_GEMMA4_PROVIDER_OPTIONS_HPP_
#define GUFO_MODELS_GEMMA4_PROVIDER_OPTIONS_HPP_

#include <cstdint>
#include <cstdlib>
#include <gufo/ext/options.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include "draft_policy.hpp"
#include "engine.hpp"
#include "kv_cache_format.hpp"
#include "prefill_weights.hpp"
#include "vision/prompt.hpp"

namespace gufo::models::gemma4 {

/// Applies the options `extension.json` declares to `options`; an option that
/// was not given keeps Gemma's default. Throws std::invalid_argument for
/// invalid values.
inline void ApplyProviderOptions(const ext::ParsedOptions& given,
                                 ModelOptions& options) {
  using ext::FindOption;
  if (const auto* scope =
          FindOption<std::string>(given, "--draft-calibration")) {
    options.draft_calibration = ParseDraftCalibrationScope(*scope);
  }
  if (const auto* path = FindOption<std::string>(given, "--draft-trace")) {
    options.draft_trace_path = *path;
  }
  if (const auto* layout =
          FindOption<std::string>(given, "--prefill-weights")) {
    options.prefill_weights = ParsePrefillWeightsOption(*layout);
  }
  // --kv-cache-type wins; GUFO_KV_CACHE_TYPE is the fallback (Ollama
  // style, like GUFO_MODELS_DIR). An explicit selection is reported
  // on stderr once per model load; the unset default stays silent.
  // Invalid values throw naming the value and its source — never a
  // silent fallback.
  if (const auto* type =
          FindOption<std::string>(given, "--kv-cache-type")) {
    try {
      options.kv_cache_format = ParseKvCacheType(*type);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("--kv-cache-type: " +
                                  std::string(e.what()));
    }
    std::cerr << "[gemma4] kv cache type: "
              << KvCacheFormatName(options.kv_cache_format)
              << " (from --kv-cache-type)\n";
  } else if (const char* env = std::getenv("GUFO_KV_CACHE_TYPE");
             env != nullptr && *env != '\0') {
    try {
      options.kv_cache_format = ParseKvCacheType(env);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("GUFO_KV_CACHE_TYPE: " +
                                  std::string(e.what()));
    }
    std::cerr << "[gemma4] kv cache type: "
              << KvCacheFormatName(options.kv_cache_format)
              << " (from GUFO_KV_CACHE_TYPE)\n";
  }
}

/// The soft-token budget per image (`--image-tokens`, default 280). Throws
/// std::invalid_argument for a budget the checkpoint does not support.
[[nodiscard]] inline std::uint32_t ImageTokens(const ext::ParsedOptions& given) {
  const auto* tokens = ext::FindOption<std::int64_t>(given, "--image-tokens");
  if (tokens == nullptr) {
    return vision::kDefaultSoftTokens;
  }
  if (*tokens <= 0 || *tokens > vision::kSoftTokenBudgets.back() ||
      !vision::IsSoftTokenBudget(static_cast<std::uint32_t>(*tokens))) {
    throw std::invalid_argument(
        "--image-tokens must be 70, 140, 280, 560 or 1120");
  }
  return static_cast<std::uint32_t>(*tokens);
}

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_PROVIDER_OPTIONS_HPP_

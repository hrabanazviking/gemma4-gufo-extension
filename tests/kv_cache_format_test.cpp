#include "kv_cache_format.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "tests/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

bool ThrowsInvalid(void (*fn)(), std::string* what) {
  try {
    fn();
  } catch (const std::invalid_argument& e) {
    *what = e.what();
    return true;
  }
  return false;
}

void CheckParse() {
  using g4::KvCacheFormat;
  Require(g4::ParseKvCacheType("") == KvCacheFormat::kF16,
          "empty kv cache type is not f16");
  Require(g4::ParseKvCacheType("f16") == KvCacheFormat::kF16,
          "f16 does not parse");
  Require(g4::ParseKvCacheType("q8_0") == KvCacheFormat::kQ8_0,
          "q8_0 does not parse");
  Require(g4::KvCacheFormatName(KvCacheFormat::kF16) == "f16",
          "f16 name wrong");
  Require(g4::KvCacheFormatName(KvCacheFormat::kQ8_0) == "q8_0",
          "q8_0 name wrong");
  Require(g4::KvCacheFormatName(KvCacheFormat::kQ4_0) == "q4_0",
          "q4_0 name wrong");

  std::string what;
  Require(ThrowsInvalid([] { (void)g4::ParseKvCacheType("q4_0"); }, &what) &&
              what.find("reserved") != std::string::npos &&
              what.find("q4_0") != std::string::npos,
          "q4_0 is not rejected as reserved: " + what);
  Require(ThrowsInvalid([] { (void)g4::ParseKvCacheType("q8"); }, &what) &&
              what.find("q8") != std::string::npos,
          "unknown type does not name the value: " + what);
  Require(ThrowsInvalid([] { (void)g4::ParseKvCacheType("F16"); }, &what),
          "names are case-sensitive; F16 accepted");
}

void CheckRowBytes() {
  using g4::KvCacheFormat;
  Require(g4::KvCacheRowBytes(2048, KvCacheFormat::kF16) == 4096,
          "f16 sliding row bytes wrong");
  Require(g4::KvCacheRowBytes(2048, KvCacheFormat::kQ8_0) == 2176,
          "q8_0 sliding row bytes wrong (64 blocks x 34)");
  Require(g4::KvCacheRowBytes(1024, KvCacheFormat::kQ8_0) == 1088,
          "q8_0 global V row bytes wrong");
  Require(g4::KvCacheRowBytes(512, KvCacheFormat::kQ8_0) == 544,
          "q8_0 derived-K row bytes wrong");
  Require(g4::KvCacheRowBytes(33, KvCacheFormat::kQ8_0) == 68,
          "q8_0 tail block does not pad");
  Require(g4::KvCacheRowBytes(0, KvCacheFormat::kQ8_0) == 0,
          "empty row is not free");
  Require(g4::KvCacheRowBytes(2048, KvCacheFormat::kQ4_0) == 1152,
          "reserved q4_0 layout bytes wrong (64 blocks x 18)");
}

void CheckStorageBytes() {
  using g4::KvCacheFormat;
  // One sliding layer, 3 slots: regions align separately.
  const std::vector<g4::KvLayerShape> one{{3, 2048, 2048}};
  Require(g4::KvCacheStorageBytes(one, KvCacheFormat::kF16) == 24576,
          "f16 storage bytes wrong for one sliding layer");
  Require(g4::KvCacheStorageBytes(one, KvCacheFormat::kQ8_0) == 13312,
          "q8_0 storage bytes wrong for one sliding layer");
  // One global layer (derived K width 512, V 1024), 1 token: the
  // f16 figure is the baseline's per-token global rate, 3,072 B.
  const std::vector<g4::KvLayerShape> global{{1, 512, 1024}};
  Require(g4::KvCacheStorageBytes(global, KvCacheFormat::kF16) == 3072,
          "f16 global layer bytes wrong");
  Require(g4::KvCacheStorageBytes(global, KvCacheFormat::kQ8_0) == 2048,
          "q8_0 global layer bytes wrong (768 K + 1280 V, aligned)");
  // The 26B-A4B geometry for one token: 25 sliding layers (K/V
  // 2048 values) + 5 global layers. The f16 total pins the KV-02
  // baseline figure of 220,160 B/token.
  std::vector<g4::KvLayerShape> model;
  for (int i = 0; i < 25; ++i) {
    model.push_back({1, 2048, 2048});
  }
  for (int i = 0; i < 5; ++i) {
    model.push_back({1, 512, 1024});
  }
  Require(g4::KvCacheStorageBytes(model, KvCacheFormat::kF16) == 220160,
          "f16 model bytes/token drifted from the KV-02 baseline");
  Require(g4::KvCacheStorageBytes(model, KvCacheFormat::kQ8_0) == 125440,
          "q8_0 model bytes/token wrong at 1 slot (alignment-inflated)");
  // At ring scale the alignment padding amortizes: one sliding
  // layer at 16,384 slots is exactly 64 q8_0 blocks per row.
  const std::vector<g4::KvLayerShape> ring{{16384, 2048, 2048}};
  Require(g4::KvCacheStorageBytes(ring, KvCacheFormat::kF16) == 134217728,
          "f16 ring-scale bytes wrong");
  Require(g4::KvCacheStorageBytes(ring, KvCacheFormat::kQ8_0) == 71303168,
          "q8_0 ring-scale bytes wrong");
}

void CheckFp16Bits() {
  const float values[] = {0.0F,   1.0F,          -2.5F,    0.1F,
                          127.0F, 1.0F / 127.0F, 65504.0F, 1e-5F};
  for (const float v : values) {
    const float back = g4::Fp16BitsToFloat(g4::FloatToFp16Bits(v));
    Require(std::fabs(back - v) <= std::fabs(v) * 0.001F + 1e-7F,
            "fp16 bit roundtrip drifted");
  }
  Require(g4::FloatToFp16Bits(0.0F) == 0, "fp16 zero wrong");
  Require(g4::FloatToFp16Bits(1.0F) == 0x3C00, "fp16 one wrong");
}

void CheckQ8BlockLayout() {
  // A constant block: every code saturates at 127, scale is 1/127.
  float ones[32];
  for (float& v : ones) {
    v = 1.0F;
  }
  std::uint8_t bytes[34];
  g4::QuantizeQ8_0Block(ones, bytes);
  for (int i = 0; i < 32; ++i) {
    Require(bytes[i] == 127, "constant block code is not 127");
  }
  const auto scale_bits = static_cast<std::uint16_t>(
      bytes[32] | (static_cast<std::uint16_t>(bytes[33]) << 8));
  Require(scale_bits == g4::FloatToFp16Bits(1.0F / 127.0F),
          "block scale is not fp16(1/127) at bytes 32..33");
  float back[32];
  g4::DequantizeQ8_0Block(bytes, back);
  for (const float v : back) {
    Require(std::fabs(v - 1.0F) < 0.01F, "constant block roundtrip off");
  }
  // A zero block roundtrips to exact zeros with a zero scale.
  float zeros[32] = {};
  g4::QuantizeQ8_0Block(zeros, bytes);
  g4::DequantizeQ8_0Block(bytes, back);
  for (int i = 0; i < 34; ++i) {
    Require(bytes[i] == 0, "zero block stores nonzero bytes");
  }
  for (const float v : back) {
    Require(v == 0.0F, "zero block roundtrip is not exact");
  }
}

// KV-05 gate evidence: a sliding-layer K row (2048 values) through
// QuantizeQ8_0Row / DequantizeQ8_0Row. Deterministic pseudo-random
// values in [-2, 2], one outlier block (index 17, amax 8) to prove
// per-block scales keep the error local. Prints the measured
// max abs error and SNR; the numbers are recorded in
// campaign/KV_PROGRESS.md.
void CheckQ8KRowRoundtrip() {
  constexpr std::size_t kValues = 2048;
  std::vector<float> row(kValues);
  std::uint32_t state = 0x12345678U;
  for (std::size_t i = 0; i < kValues; ++i) {
    state = state * 1664525U + 1013904223U;
    const float u = static_cast<float>(state >> 8) / 16777216.0F;
    row[i] = u * 4.0F - 2.0F;
  }
  for (std::size_t i = 17 * 32; i < 18 * 32; ++i) {
    row[i] = 0.01F;
  }
  row[17 * 32] = 8.0F;
  std::vector<std::uint8_t> packed(
      g4::KvCacheRowBytes(2048, g4::KvCacheFormat::kQ8_0));
  Require(packed.size() == 2176, "q8_0 K row is not 2176 bytes");
  g4::QuantizeQ8_0Row(row, packed.data());
  std::vector<float> back(kValues);
  g4::DequantizeQ8_0Row(packed.data(), back.data(), kValues);
  double max_abs = 0.0;
  double max_abs_calm = 0.0;
  double sig = 0.0;
  double err = 0.0;
  for (std::size_t i = 0; i < kValues; ++i) {
    const double e = std::fabs(static_cast<double>(back[i]) - row[i]);
    max_abs = std::max(max_abs, e);
    if (i / 32 != 17) {
      max_abs_calm = std::max(max_abs_calm, e);
    }
    sig += static_cast<double>(row[i]) * row[i];
    err += e * e;
  }
  const double snr_db = 10.0 * std::log10(sig / err);
  std::printf(
      "q8_0 K-row roundtrip: max_abs_err=%.6f "
      "max_abs_err_calm_blocks=%.6f snr_db=%.2f\n",
      max_abs, max_abs_calm, snr_db);
  // Theory: per-block error <= scale/2 (+ fp16 scale rounding);
  // the outlier block's scale is 8/127 = 0.0630, calm blocks ~2/127.
  Require(max_abs <= 0.04, "q8_0 K roundtrip max abs error too large");
  Require(max_abs_calm <= 0.012, "q8_0 error leaked outside the outlier block");
  Require(snr_db >= 30.0, "q8_0 K roundtrip SNR too low");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckParse();
    CheckRowBytes();
    CheckStorageBytes();
    CheckFp16Bits();
    CheckQ8BlockLayout();
    CheckQ8KRowRoundtrip();
  });
}

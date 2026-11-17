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
  // Layout v2 (Phase 5 REDO R2-global step 1): q8_0 rows are
  // versioned split-plane -- payload first, scales after -- at
  // unchanged totals.
  Require(g4::KvCacheQ8LayoutVersion(KvCacheFormat::kQ8_0) == 2,
          "q8_0 layout version is not v2 (split scale plane)");
  Require(g4::KvCacheQ8LayoutVersion(KvCacheFormat::kF16) == 0,
          "f16 must not carry a q8_0 layout version");
  Require(g4::Q8_0PayloadOffset(2048, 3) == 96, "q8_0 payload offset wrong");
  Require(g4::Q8_0ScaleOffset(2048, 0) == 2048,
          "q8_0 scale plane does not start after the payload");
  Require(g4::Q8_0ScaleOffset(2048, 63) == 2048 + 126,
          "q8_0 scale offset wrong");
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
  std::uint8_t codes[32];
  std::uint16_t scale_bits = 0;
  g4::QuantizeQ8_0Block(ones, codes, &scale_bits);
  for (int i = 0; i < 32; ++i) {
    Require(codes[i] == 127, "constant block code is not 127");
  }
  Require(scale_bits == g4::FloatToFp16Bits(1.0F / 127.0F),
          "block scale is not fp16(1/127)");
  float back[32];
  g4::DequantizeQ8_0Block(codes, scale_bits, back);
  for (const float v : back) {
    Require(std::fabs(v - 1.0F) < 0.01F, "constant block roundtrip off");
  }
  // A zero block roundtrips to exact zeros with a zero scale.
  float zeros[32] = {};
  g4::QuantizeQ8_0Block(zeros, codes, &scale_bits);
  g4::DequantizeQ8_0Block(codes, scale_bits, back);
  for (int i = 0; i < 32; ++i) {
    Require(codes[i] == 0, "zero block stores nonzero codes");
  }
  Require(scale_bits == 0, "zero block stores a nonzero scale");
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

// KV-06 gate evidence, V half: a sliding-layer V row (2048 values)
// through the same host reference the QkvPost V store mirrors.
// Different seed, range and outlier block (41, amax 6) from the K
// test so the two halves are independent measurements.
void CheckQ8VRowRoundtrip() {
  constexpr std::size_t kValues = 2048;
  std::vector<float> row(kValues);
  std::uint32_t state = 0x9E3779B9U;
  for (std::size_t i = 0; i < kValues; ++i) {
    state = state * 1664525U + 1013904223U;
    const float u = static_cast<float>(state >> 8) / 16777216.0F;
    row[i] = u * 4.0F - 2.0F;
  }
  for (std::size_t i = 41 * 32; i < 42 * 32; ++i) {
    row[i] = 0.01F;
  }
  row[41 * 32] = -6.0F;
  std::vector<std::uint8_t> packed(
      g4::KvCacheRowBytes(2048, g4::KvCacheFormat::kQ8_0));
  Require(packed.size() == 2176, "q8_0 V row is not 2176 bytes");
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
    if (i / 32 != 41) {
      max_abs_calm = std::max(max_abs_calm, e);
    }
    sig += static_cast<double>(row[i]) * row[i];
    err += e * e;
  }
  const double snr_db = 10.0 * std::log10(sig / err);
  std::printf(
      "q8_0 V-row roundtrip: max_abs_err=%.6f "
      "max_abs_err_calm_blocks=%.6f snr_db=%.2f\n",
      max_abs, max_abs_calm, snr_db);
  Require(max_abs <= 0.04, "q8_0 V roundtrip max abs error too large");
  Require(max_abs_calm <= 0.012,
          "q8_0 V error leaked outside the outlier block");
  Require(snr_db >= 30.0, "q8_0 V roundtrip SNR too low");
}

// KV-06 ring-layout proof (host reference): rows are stored exactly
// as QkvPost addresses them -- slot = position % ring, row bytes
// from KvCacheRowBytes -- across two full ring wraps. Every slot
// must roundtrip its LAST writer, and a constant row written at a
// wrapped position must land at the exact byte offsets of its slot
// (layout v2: codes at slot*row_bytes + block*32 in the payload
// plane, scales in the scale plane at slot*row_bytes + 2048 +
// block*2) with the neighbouring slots untouched.
void CheckQ8RingLayout() {
  constexpr std::uint32_t kRing = 8;
  constexpr std::size_t kValues = 2048;
  const std::size_t row_bytes =
      g4::KvCacheRowBytes(2048, g4::KvCacheFormat::kQ8_0);
  Require(row_bytes == 2176, "ring test row bytes wrong");
  const auto make_row = [](std::uint32_t pos) {
    std::vector<float> row(kValues);
    std::uint32_t state = 0x85EBCA6BU + pos * 0x27D4EB2FU;
    for (std::size_t i = 0; i < kValues; ++i) {
      state = state * 1664525U + 1013904223U;
      const float u = static_cast<float>(state >> 8) / 16777216.0F;
      row[i] = u * 3.0F - 1.5F;
    }
    return row;
  };
  std::vector<std::uint8_t> cache(kRing * row_bytes, 0);
  std::vector<std::vector<float>> last(kRing);
  for (std::uint32_t pos = 0; pos < 20; ++pos) {
    const std::uint32_t slot = pos % kRing;
    std::vector<float> row = make_row(pos);
    g4::QuantizeQ8_0Row(row, cache.data() + slot * row_bytes);
    last[slot] = row;
  }
  double max_abs = 0.0;
  std::vector<float> back(kValues);
  for (std::uint32_t slot = 0; slot < kRing; ++slot) {
    g4::DequantizeQ8_0Row(cache.data() + slot * row_bytes, back.data(),
                          kValues);
    for (std::size_t i = 0; i < kValues; ++i) {
      max_abs = std::max(
          max_abs, std::fabs(static_cast<double>(back[i]) - last[slot][i]));
    }
  }
  std::printf("q8_0 ring layout: slots=%u wraps=2 max_abs_err=%.6f\n", kRing,
              max_abs);
  // Row values are in [-1.5, 1.5]: per-block scale <= 1.5/127, so
  // the roundtrip error stays under half a step plus fp16 rounding.
  Require(max_abs <= 0.012, "ring-wrapped q8_0 rows roundtrip badly");
  // Byte-offset pin on a fresh region: a constant 0.5 row stored at
  // position kRing + 2 lands in slot 2; in layout v2 every block's
  // codes 127 sit in the payload plane (block * 32) and the fp16
  // scale of 0.5/127 sits in the scale plane (2048 + block * 2).
  std::vector<std::uint8_t> pin(kRing * row_bytes, 0);
  std::vector<float> flat(kValues, 0.5F);
  const std::uint32_t wrapped_pos = kRing + 2;
  g4::QuantizeQ8_0Row(flat, pin.data() + (wrapped_pos % kRing) * row_bytes);
  const std::size_t base = 2 * row_bytes;
  const auto want_scale = g4::FloatToFp16Bits(0.5F / 127.0F);
  for (std::size_t blk = 0; blk < kValues / 32; ++blk) {
    const std::size_t at = base + blk * 32;
    for (int i = 0; i < 32; ++i) {
      Require(pin[at + i] == 127, "ring pin: code not at its byte offset");
    }
    const std::size_t sat = base + kValues + blk * 2;
    const auto scale_bits = static_cast<std::uint16_t>(
        pin[sat] | (static_cast<std::uint16_t>(pin[sat + 1]) << 8));
    Require(scale_bits == want_scale, "ring pin: scale not in scale plane");
  }
  for (std::size_t i = 0; i < pin.size(); ++i) {
    if (i < base || i >= base + row_bytes) {
      Require(pin[i] == 0, "ring pin: write leaked into another slot");
    }
  }
}

// KV-09: the cache the executor actually allocates in q8_0 mode --
// every layer q8_0 since the derived-K design landed (KV-07/KV-08
// shipped the mixed cache: sliding q8_0, global binary16, 130,560
// B/token). Per token: 25 sliding x (2,304 + 2,304) aligned q8_0
// bytes + 5 global x (768 rotated-K + 1,280 V) aligned q8_0 bytes
// = 125,440 B, against 220,160 B all-f16.
void CheckLayerStorageBytes() {
  using g4::KvCacheFormat;
  Require(g4::KvCacheLayerFormat(KvCacheFormat::kQ8_0, true) ==
              KvCacheFormat::kQ8_0,
          "sliding layer does not take the cache format");
  Require(g4::KvCacheLayerFormat(KvCacheFormat::kQ8_0, false) ==
              KvCacheFormat::kQ8_0,
          "global layer does not take the cache format");
  Require(
      g4::KvCacheLayerFormat(KvCacheFormat::kF16, true) == KvCacheFormat::kF16,
      "f16 cache layer format wrong");
  std::vector<g4::KvLayerShape> model;
  std::vector<KvCacheFormat> formats;
  for (int i = 0; i < 25; ++i) {
    model.push_back({1, 2048, 2048});
    formats.push_back(KvCacheFormat::kQ8_0);
  }
  for (int i = 0; i < 5; ++i) {
    model.push_back({1, 512, 1024});
    formats.push_back(KvCacheFormat::kQ8_0);
  }
  Require(g4::KvCacheStorageBytesPerLayer(model, formats) == 125440,
          "full q8_0 model bytes/token wrong");
  // Uniform formats through the per-layer road reproduce the
  // uniform figures.
  std::vector<KvCacheFormat> all_f16(model.size(), KvCacheFormat::kF16);
  Require(g4::KvCacheStorageBytesPerLayer(model, all_f16) == 220160,
          "per-layer f16 bytes drifted from the KV-02 baseline");
  std::vector<KvCacheFormat> all_q8(model.size(), KvCacheFormat::kQ8_0);
  Require(g4::KvCacheStorageBytesPerLayer(model, all_q8) ==
              g4::KvCacheStorageBytes(model, KvCacheFormat::kQ8_0),
          "per-layer q8_0 bytes differ from the uniform road");
}

// KV-09 gate evidence (host half): the derived-K reconstruction
// over q8_0 stores. One global head (D = 512, pairs = 64): V is a
// deterministic pseudo-random row in [-2, 2]; the true key is
// K = rope(k_norm . V) at a fixed position's angles. The stores are
// the q8_0 V row and the q8_0 packed rotated-dims row (quantized
// after rotation, as QkvPost stores them). Reconstruction reads
// rotated dims from the rotated store and every other dim as
// k_norm * dequantized V (in the kernels the query carries k_norm;
// the product is the same) -- the exact rule the attention paths
// implement. Rope is orthogonal and applied before quantization,
// never to quantized values, so the reconstruction must land within
// the campaign's recorded bounds with no amplification of its own.
void CheckDerivedKeyReconstruction() {
  constexpr std::uint32_t kDim = 512;
  constexpr std::uint32_t kPairs = 64;
  std::vector<float> v(kDim);
  std::uint32_t state = 0x85EBCA6BU;
  for (std::uint32_t i = 0; i < kDim; ++i) {
    state = state * 1664525U + 1013904223U;
    const float u = static_cast<float>(state >> 8) / 16777216.0F;
    v[i] = u * 4.0F - 2.0F;
  }
  std::vector<float> k_norm(kDim);
  for (std::uint32_t i = 0; i < kDim; ++i) {
    k_norm[i] = 0.5F + static_cast<float>(i % 7) * 0.25F;
  }
  std::vector<float> key(kDim);
  for (std::uint32_t i = 0; i < kDim; ++i) {
    key[i] = k_norm[i] * v[i];
  }
  for (std::uint32_t i = 0; i < kPairs; ++i) {
    const double theta = 0.013 * static_cast<double>(i + 1) * 7.0;
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(std::sin(theta));
    const float lo = key[i];
    const float hi = key[i + kDim / 2];
    key[i] = lo * c - hi * s;
    key[i + kDim / 2] = lo * s + hi * c;
  }
  std::vector<std::uint8_t> v_store(
      g4::KvCacheRowBytes(kDim, g4::KvCacheFormat::kQ8_0));
  g4::QuantizeQ8_0Row(v, v_store.data());
  std::vector<float> packed(2 * kPairs);
  for (std::uint32_t i = 0; i < kPairs; ++i) {
    packed[i] = key[i];
    packed[kPairs + i] = key[i + kDim / 2];
  }
  std::vector<std::uint8_t> k_store(
      g4::KvCacheRowBytes(2 * kPairs, g4::KvCacheFormat::kQ8_0));
  g4::QuantizeQ8_0Row(packed, k_store.data());
  std::vector<float> v_back(kDim);
  g4::DequantizeQ8_0Row(v_store.data(), v_back.data(), kDim);
  std::vector<float> packed_back(2 * kPairs);
  g4::DequantizeQ8_0Row(k_store.data(), packed_back.data(), 2 * kPairs);
  std::vector<float> rec(kDim);
  for (std::uint32_t i = 0; i < kDim; ++i) {
    const std::uint32_t seg = i % (kDim / 2);
    if (seg < kPairs) {
      rec[i] = packed_back[i < kDim / 2 ? i : kPairs + i - kDim / 2];
    } else {
      rec[i] = k_norm[i] * v_back[i];
    }
  }
  double max_abs = 0.0;
  double sig = 0.0;
  double err = 0.0;
  for (std::uint32_t i = 0; i < kDim; ++i) {
    const double e = std::fabs(static_cast<double>(rec[i]) - key[i]);
    max_abs = std::max(max_abs, e);
    sig += static_cast<double>(key[i]) * key[i];
    err += e * e;
  }
  const double snr_db = 10.0 * std::log10(sig / err);
  std::printf("q8_0 derived-K reconstruction: max_abs_err=%.6f snr_db=%.2f\n",
              max_abs, snr_db);
  Require(max_abs <= 0.04, "derived-K reconstruction max abs error too large");
  Require(snr_db >= 30.0, "derived-K reconstruction SNR too low");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckParse();
    CheckRowBytes();
    CheckStorageBytes();
    CheckLayerStorageBytes();
    CheckDerivedKeyReconstruction();
    CheckFp16Bits();
    CheckQ8BlockLayout();
    CheckQ8KRowRoundtrip();
    CheckQ8VRowRoundtrip();
    CheckQ8RingLayout();
  });
}

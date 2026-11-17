#include "kv_cache_format.hpp"

#include <cstddef>
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

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckParse();
    CheckRowBytes();
    CheckStorageBytes();
  });
}

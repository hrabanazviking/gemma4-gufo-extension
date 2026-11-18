// KV-13 (Phase 6, machinery re-proof): MoveKey over the quantized
// layout. Executor::MoveKey fills MoveKeyArgs with widths in uint16
// units of the STORED row -- KvCacheRowBytes(values, layer format) / 2
// -- so the kernel moves a q8_0 row's exact stored bytes. This test
// drives rocm::MoveKey directly with exactly those widths over
// production-shaped rows (a sliding row of 2,048 values = 2,176 stored
// bytes, a derived-K row of 256 values = 272 bytes beside its 1,024-
// value V row = 1,088 bytes, a plain global row), including a move
// whose key indices wrap the sliding ring, and checks byte-exact
// movement with every other row untouched. An f16 run is the control:
// the same kernel with binary16 widths, as before KV-07.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <iostream>
#include <vector>

#include "hip/kernels.hpp"
#include "kv_cache_format.hpp"
#include "tests/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::uint16_t* Device(const std::vector<std::uint16_t>& host) {
  std::uint16_t* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(std::uint16_t) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  return ptr;
}

std::vector<std::uint16_t> Host(const std::uint16_t* ptr, std::size_t n) {
  std::vector<std::uint16_t> out(n);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out.data(), ptr, n * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  return out;
}

struct LayerSpec {
  std::uint32_t k_values;
  std::uint32_t v_values;
  std::uint32_t ring;  // 0 = linear (global), else the sliding ring
  bool sliding;
};

constexpr std::uint32_t kRows = 32;

void RunFormat(g4::KvCacheFormat format, const char* name) {
  const LayerSpec specs[] = {
      {2048, 2048, 8, true},  // sliding layer, ring of 8 keys
      {256, 1024, 0, false},  // global derived-K layer
      {512, 512, 0, false},   // plain global layer
  };
  g4::rocm::MoveKeyArgs args{};
  args.layers = 3;
  struct Side {
    std::uint16_t* device;
    std::vector<std::uint16_t> mirror;
    std::uint32_t width;  // uint16 units of the stored row
    std::uint32_t ring;
  };
  std::vector<Side> sides;
  for (std::uint32_t l = 0; l < args.layers; ++l) {
    const g4::KvCacheFormat layer =
        g4::KvCacheLayerFormat(format, specs[l].sliding);
    for (const bool values : {false, true}) {
      const std::uint32_t count =
          values ? specs[l].v_values : specs[l].k_values;
      // Exactly Executor::MoveKey's width formula (KV-07/KV-09).
      const std::uint32_t width =
          static_cast<std::uint32_t>(g4::KvCacheRowBytes(count, layer) / 2);
      std::vector<std::uint16_t> host(std::size_t{kRows} * width);
      for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t i = 0; i < width; ++i) {
          host[std::size_t{r} * width + i] = static_cast<std::uint16_t>(
              0x5A00 + r * 977 + i * 131 + l * 17 + (values ? 0x40 : 0));
        }
      }
      Side side{Device(host), host, width, specs[l].ring};
      if (values) {
        args.v[l] = side.device;
        args.v_width[l] = width;
      } else {
        args.k[l] = side.device;
        args.k_width[l] = width;
      }
      args.ring[l] = specs[l].ring;
      sides.push_back(std::move(side));
    }
  }
  const auto move = [&](std::uint32_t from, std::uint32_t to) {
    args.from = from;
    args.to = to;
    g4::rocm::MoveKey(args, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    for (Side& side : sides) {
      const std::uint32_t src = side.ring != 0 ? from % side.ring : from;
      const std::uint32_t dst = side.ring != 0 ? to % side.ring : to;
      std::memcpy(&side.mirror[std::size_t{dst} * side.width],
                  &side.mirror[std::size_t{src} * side.width],
                  std::size_t{side.width} * 2);
      Require(Host(side.device, side.mirror.size()) == side.mirror,
              std::string("MoveKey moved wrong bytes (") + name + ")");
    }
  };
  move(3, 12);   // direct: sliding slots 3 -> 4, global rows 3 -> 12
  move(19, 26);  // wrapped: sliding slots 3 -> 2, global rows 19 -> 26
  for (Side& side : sides) {
    HIP_CHECK(hipFree(side.device));
  }
  std::cout << "MoveKey " << name << ": byte-exact, ring wrap included\n";
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    RunFormat(g4::KvCacheFormat::kF16, "f16 control");
    RunFormat(g4::KvCacheFormat::kQ8_0, "q8_0");
  });
}

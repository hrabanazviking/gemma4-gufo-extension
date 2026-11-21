// R3-E1 part 2 differential test, shapes edition: like
// r3e1p2_diff.hip.cpp but parameterized over heads / kv_heads
// and the executor's two scratch layouts (packed rows, and the
// fused QKV projection with row_stride = qkv_stride and V
// beside K in the same rows). Bypass (current unset) vs staged
// (current set) must be bit-identical in every shape.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "hip/attention_wmma.hpp"
#include "hip/kernels.hpp"
#include "kv_cache_format.hpp"

namespace k = gufo::models::gemma4::rocm;
using gufo::models::gemma4::KvCacheFormat;
using gufo::models::gemma4::KvCacheRowBytes;

namespace {

constexpr std::uint32_t D = 256;
constexpr float kEps = 1e-6F;
const float kThetaScale = std::pow(1000000.0F, -2.0F / 256.0F);

template <class T>
T* Dev(const std::vector<T>& h) {
  T* p = nullptr;
  if (hipMalloc(&p, h.size() * sizeof(T) + 16) != hipSuccess) return nullptr;
  hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
  return p;
}

struct Fixture {
  std::uint32_t heads, kvh, ring;
  bool fused;
  std::uint32_t stride;  // fused row width in floats (0 = packed)
  float *dbase = nullptr;              // fused buffer (or Q buffer)
  float *dk = nullptr, *dv = nullptr;  // K / V views (offsets applied)
  float *dqn = nullptr, *dkn = nullptr;
  float* dqout = nullptr;
  std::uint8_t *dkc = nullptr, *dvc = nullptr;
  std::uint16_t *dkf = nullptr, *dvf = nullptr;
  float* dpart = nullptr;

  std::size_t QRow() const { return fused ? stride : std::size_t{heads} * D; }
  std::size_t KvRow() const {
    return fused ? stride : std::size_t{kvh} * D;
  }
  std::size_t VRow() const { return fused ? stride : std::size_t{kvh} * D; }

  int Build(std::uint32_t total, std::uint32_t ring_, std::uint32_t heads_,
            std::uint32_t kvh_, bool fused_, std::uint32_t seed) {
    heads = heads_;
    kvh = kvh_;
    ring = ring_;
    fused = fused_;
    stride = fused ? (heads + 2 * kvh) * D : 0;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 0.6F);
    const std::size_t roww = fused ? stride : (heads + 2 * kvh) * D;
    std::vector<float> buf(total * roww);
    for (auto& x : buf) x = nd(rng);
    dbase = Dev(buf);
    dk = dbase + std::size_t{heads} * D;
    dv = dk + std::size_t{kvh} * D;
    std::vector<float> qn(D), kn(D);
    std::uniform_real_distribution<float> ud(0.8F, 1.2F);
    for (auto& x : qn) x = ud(rng);
    for (auto& x : kn) x = ud(rng);
    dqn = Dev(qn);
    dkn = Dev(kn);
    std::vector<float> qo(total * heads * D, 0.0F);
    dqout = Dev(qo);
    const std::size_t rowb = KvCacheRowBytes(kvh * D, KvCacheFormat::kQ8_0);
    std::vector<std::uint8_t> cache(ring * rowb, 0);
    dkc = Dev(cache);
    dvc = Dev(cache);
    std::vector<std::uint16_t> f16c(ring * kvh * D, 0);
    dkf = Dev(f16c);
    dvf = Dev(f16c);
    std::vector<float> part(k::AttentionPartialFloats(256, heads, D, 8192));
    dpart = Dev(part);
    return (dbase && dqn && dkn && dqout && dkc && dvc && dkf && dvf &&
            dpart)
               ? 0
               : 1;
  }

  void Post(std::uint32_t fp, std::uint32_t rows) {
    k::QkvPostArgs post{};
    post.q = dbase + std::size_t{fp} * QRow();
    post.k = dk + std::size_t{fp} * KvRow();
    post.v = dv + std::size_t{fp} * VRow();
    post.q_norm = dqn;
    post.k_norm = dkn;
    post.theta_scale = kThetaScale;
    post.freq_factors = nullptr;
    post.k_cache = dkf;
    post.v_cache = dvf;
    post.k_cache_q8 = dkc;
    post.v_cache_q8 = dvc;
    post.rows = rows;
    post.heads = heads;
    post.kv_heads = kvh;
    post.head_dim = D;
    post.first_position = fp;
    post.ring = ring;
    post.eps = kEps;
    post.rotated_pairs = 0;
    post.row_stride = fused ? stride : 0;
    post.v_stride = fused ? stride : 0;
    post.q_out = dqout + std::size_t{fp} * heads * D;
    post.siblings = 0;
    post.spare_key = 0;
    k::QkvPost(post, nullptr);
  }

  std::vector<float> Attend(std::uint32_t fp, std::uint32_t rows,
                            std::uint32_t window, bool staged) {
    std::vector<float> out(std::size_t{rows} * heads * D, -777.0F);
    float* dout = Dev(out);
    k::AttentionArgs a{};
    a.q = dqout + std::size_t{fp} * heads * D;
    a.k_cache = dkf;
    a.v_cache = dvf;
    a.k_cache_q8 = dkc;
    a.v_cache_q8 = dvc;
    a.out = dout;
    a.partials = dpart;
    a.rows = rows;
    a.heads = heads;
    a.kv_heads = kvh;
    a.head_dim = D;
    a.first_position = fp;
    a.shared_position = false;
    a.key_limit = std::numeric_limits<std::uint32_t>::max();
    a.window = window;
    a.ring = ring;
    a.rope_pairs = 0;
    a.key_ends = nullptr;
    a.out_half = nullptr;
    a.siblings = 0;
    a.spare_key = 0;
    if (staged) {
      a.current.k = dk + std::size_t{fp} * KvRow();
      a.current.v = dv + std::size_t{fp} * VRow();
      a.current.k_norm = dkn;
      a.current.theta_scale = kThetaScale;
      a.current.freq_factors = nullptr;
      a.current.eps = kEps;
      a.current.first_position = fp;
      a.current.rows = rows;
      a.current.row_stride = fused ? stride : 0;
      a.current.v_stride = fused ? stride : 0;
    }
    k::Attention(a, nullptr);
    hipDeviceSynchronize();
    hipMemcpy(out.data(), dout, out.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    hipFree(dout);
    return out;
  }
};

int g_fail = 0;

void Compare(const char* name, const std::vector<float>& x,
             const std::vector<float>& y) {
  std::size_t bad = 0, first = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (std::memcmp(&x[i], &y[i], 4) != 0) {
      if (bad == 0) first = i;
      ++bad;
    }
  }
  std::cout << name << ": " << (bad == 0 ? "BIT-IDENTICAL" : "MISMATCH")
            << " (" << bad << "/" << x.size() << " floats differ";
  if (bad != 0) {
    std::cout << ", first at " << first << ": " << x[first] << " vs " << y[first];
    ++g_fail;
  }
  std::cout << ")\n";
}

int Run(const char* name, std::uint32_t heads, std::uint32_t kvh, bool fused,
        std::uint32_t hist, std::uint32_t fp, std::uint32_t rows,
        std::uint32_t window, std::uint32_t ring) {
  Fixture f;
  if (f.Build(fp + rows, ring, heads, kvh, fused, 777)) {
    std::cout << name << ": FIXTURE FAILED\n";
    return 1;
  }
  if (hist != 0) f.Post(0, hist);
  f.Post(fp, rows);
  hipDeviceSynchronize();
  if (hipGetLastError() != hipSuccess) {
    std::cout << name << ": QkvPost HIP error\n";
    return 1;
  }
  const auto staged = f.Attend(fp, rows, window, true);
  const auto bypass = f.Attend(fp, rows, window, false);
  if (hipGetLastError() != hipSuccess) {
    std::cout << name << ": Attention HIP error\n";
    return 1;
  }
  Compare(name, staged, bypass);
  return 0;
}

}  // namespace

int main() {
  if (hipSetDevice(0) != hipSuccess) return 1;
  int rc = 0;
  // N: exact COH-1 shape: h16 kv8 fused, rows27, win1024, ring3328.
  rc |= Run("N fused h16 kv8 rows27 ring3328", 16, 8, true, 0, 0, 27, 1024, 3328);
  // O: same but ring1024.
  rc |= Run("O fused h16 kv8 rows27 ring1024", 16, 8, true, 0, 0, 27, 1024, 1024);
  // P: rows27 packed.
  rc |= Run("P packed h16 kv8 rows27", 16, 8, false, 0, 0, 27, 1024, 3328);
  // Q: rows64 ring3328 (full tiles, partial query block none).
  rc |= Run("Q fused h16 kv8 rows64 ring3328", 16, 8, true, 0, 0, 64, 1024, 3328);
  // R: rows27 with history 0 but second segment style: fp27 rows27.
  rc |= Run("R fused h16 kv8 fp27 rows27", 16, 8, true, 27, 27, 27, 1024, 3328);
  std::cout << (g_fail == 0 && rc == 0 ? "ALL BIT-IDENTICAL\n" : "FAILURES\n");
  return (g_fail == 0 && rc == 0) ? 0 : 1;
}

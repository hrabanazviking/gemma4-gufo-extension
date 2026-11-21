// R3-E1 part 2 differential device test: WMMA prefill attention
// over a q8_0 sliding D=256 cache, bypass (AttentionArgs.current
// unset, today's cache-read road) vs LDS-staged reproduction
// (current set, R3-E1 part 2). The cache is filled by QkvPost
// itself from the same FP32 projections the descriptor points
// at, so both roads must produce bit-identical FP32 outputs if
// the reproduction is exact. Case A also corrupts the current
// keys' cache rows after the reference runs: the staged output
// must not move (it never reads those rows), the bypass output
// must move (it does) -- proof the staged road really engages.
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

#define HIP_CHECK(x)                                                     \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      std::cerr << "HIP error " << hipGetErrorString(e_) << " at line "  \
                << __LINE__ << "\n";                                     \
      return 1;                                                          \
    }                                                                    \
  } while (0)

namespace {

constexpr std::uint32_t D = 256;
constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kKvHeads = 1;
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
  std::uint32_t total;  // positions 0..total-1 have projections
  std::uint32_t ring;
  float *dq = nullptr, *dk = nullptr, *dv = nullptr;    // projections
  float *dqn = nullptr, *dkn = nullptr;                 // norms
  float* dqout = nullptr;                               // packed Q
  std::uint8_t *dkc = nullptr, *dvc = nullptr;          // q8 caches
  std::uint16_t *dkf = nullptr, *dvf = nullptr;         // dummy f16
  float* dpart = nullptr;

  int Build(std::uint32_t total_, std::uint32_t ring_, std::uint32_t seed) {
    total = total_;
    ring = ring_;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0F, 0.6F);
    std::vector<float> q(total * kHeads * D), kk(total * D), vv(total * D);
    for (auto& x : q) x = nd(rng);
    for (auto& x : kk) x = nd(rng);
    for (auto& x : vv) x = nd(rng);
    std::vector<float> qn(D), kn(D);
    std::uniform_real_distribution<float> ud(0.8F, 1.2F);
    for (auto& x : qn) x = ud(rng);
    for (auto& x : kn) x = ud(rng);
    dq = Dev(q);
    dk = Dev(kk);
    dv = Dev(vv);
    dqn = Dev(qn);
    dkn = Dev(kn);
    std::vector<float> qo(total * kHeads * D, 0.0F);
    dqout = Dev(qo);
    const std::size_t rowb = KvCacheRowBytes(kKvHeads * D, KvCacheFormat::kQ8_0);
    std::vector<std::uint8_t> cache(ring * rowb, 0);
    dkc = Dev(cache);
    dvc = Dev(cache);
    std::vector<std::uint16_t> f16c(ring * kKvHeads * D, 0);
    dkf = Dev(f16c);
    dvf = Dev(f16c);
    std::vector<float> part(k::AttentionPartialFloats(256, kHeads, D, 8192));
    dpart = Dev(part);
    return (dq && dk && dv && dqn && dkn && dqout && dkc && dvc && dkf &&
            dvf && dpart)
               ? 0
               : 1;
  }

  // QkvPost one segment [fp, fp+rows) exactly as the executor does.
  void Post(std::uint32_t fp, std::uint32_t rows, std::uint32_t window) {
    k::QkvPostArgs post{};
    post.q = dq + std::size_t{fp} * kHeads * D;
    post.k = dk + std::size_t{fp} * D;
    post.v = dv + std::size_t{fp} * D;
    post.q_norm = dqn;
    post.k_norm = dkn;
    post.theta_scale = kThetaScale;
    post.freq_factors = nullptr;  // sliding layers
    post.k_cache = dkf;
    post.v_cache = dvf;
    post.k_cache_q8 = dkc;
    post.v_cache_q8 = dvc;
    post.rows = rows;
    post.heads = kHeads;
    post.kv_heads = kKvHeads;
    post.head_dim = D;
    post.first_position = fp;
    post.ring = ring;
    post.eps = kEps;
    post.rotated_pairs = 0;
    post.row_stride = 0;
    post.v_stride = 0;
    post.q_out = dqout + std::size_t{fp} * kHeads * D;
    post.siblings = 0;
    post.spare_key = 0;
    (void)window;
    k::QkvPost(post, nullptr);
  }

  // Attention for the segment [fp, fp+rows); staged toggles the
  // current-chunk descriptor. direct_wmma forces the WMMA launch
  // (the production dispatch sends rows <= 16 to split attention).
  std::vector<float> Attend(std::uint32_t fp, std::uint32_t rows,
                            std::uint32_t window, bool staged,
                            bool direct_wmma, bool* launched = nullptr) {
    std::vector<float> out(std::size_t{rows} * kHeads * D, -777.0F);
    float* dout = Dev(out);
    k::AttentionArgs a{};
    a.q = dqout + std::size_t{fp} * kHeads * D;
    a.k_cache = dkf;
    a.v_cache = dvf;
    a.k_cache_q8 = dkc;
    a.v_cache_q8 = dvc;
    a.out = dout;
    a.partials = dpart;
    a.rows = rows;
    a.heads = kHeads;
    a.kv_heads = kKvHeads;
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
      a.current.k = dk + std::size_t{fp} * D;
      a.current.v = dv + std::size_t{fp} * D;
      a.current.k_norm = dkn;
      a.current.theta_scale = kThetaScale;
      a.current.freq_factors = nullptr;
      a.current.eps = kEps;
      a.current.first_position = fp;
      a.current.rows = rows;
      a.current.row_stride = 0;
      a.current.v_stride = 0;
    }
    if (direct_wmma) {
      const bool ok = k::LaunchWmmaPrefillAttention(a, nullptr);
      if (launched != nullptr) *launched = ok;
    } else {
      k::Attention(a, nullptr);
    }
    hipDeviceSynchronize();
    hipMemcpy(out.data(), dout, out.size() * sizeof(float),
              hipMemcpyDeviceToHost);
    hipFree(dout);
    return out;
  }

  void CorruptCurrent(std::uint32_t fp, std::uint32_t rows) {
    const std::size_t rowb = KvCacheRowBytes(kKvHeads * D, KvCacheFormat::kQ8_0);
    std::vector<std::uint8_t> zeros(rowb, 0);
    for (std::uint32_t key = fp; key < fp + rows; ++key) {
      const std::uint32_t slot = ring != 0 ? key % ring : key;
      hipMemcpy(dkc + std::size_t{slot} * rowb, zeros.data(), rowb,
                hipMemcpyHostToDevice);
      hipMemcpy(dvc + std::size_t{slot} * rowb, zeros.data(), rowb,
                hipMemcpyHostToDevice);
    }
    hipDeviceSynchronize();
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
    std::cout << ", first at " << first << ": " << x[first] << " vs "
              << y[first];
    ++g_fail;
  }
  std::cout << ")\n";
}

int RunCase(const char* name, std::uint32_t hist, std::uint32_t fp,
            std::uint32_t rows, std::uint32_t window, std::uint32_t ring,
            bool direct_wmma, bool corrupt_check) {
  Fixture f;
  if (f.Build(fp + rows, ring, 1234)) {
    std::cout << name << ": FIXTURE FAILED\n";
    return 1;
  }
  if (hist != 0) f.Post(0, hist, window);
  f.Post(fp, rows, window);
  hipDeviceSynchronize();
  if (hipGetLastError() != hipSuccess) {
    std::cout << name << ": QkvPost HIP error\n";
    return 1;
  }
  bool launched = false;
  const auto staged = f.Attend(fp, rows, window, true, direct_wmma, &launched);
  const auto bypass = f.Attend(fp, rows, window, false, direct_wmma);
  if (hipGetLastError() != hipSuccess) {
    std::cout << name << ": Attention HIP error\n";
    return 1;
  }
  if (direct_wmma && !launched) {
    std::cout << name << ": WMMA launch declined (unexpected)\n";
    ++g_fail;
  }
  Compare(name, staged, bypass);
  if (corrupt_check) {
    f.CorruptCurrent(fp, rows);
    const auto staged_c = f.Attend(fp, rows, window, true, direct_wmma);
    const auto bypass_c = f.Attend(fp, rows, window, false, direct_wmma);
    Compare("A-corrupt staged==staged", staged, staged_c);
    std::size_t moved = 0;
    for (std::size_t i = 0; i < bypass.size(); ++i) {
      if (std::memcmp(&bypass[i], &bypass_c[i], 4) != 0) ++moved;
    }
    std::cout << "A-corrupt bypass moved: " << moved << "/" << bypass.size()
              << (moved == 0 ? " (FAIL: corruption had no effect)\n"
                             : " floats (expected: bypass reads cache)\n");
    if (moved == 0) ++g_fail;
  }
  return 0;
}

}  // namespace

int main() {
  HIP_CHECK(hipSetDevice(0));
  int rc = 0;
  // A: first_position 0, all keys current (+ corruption proof).
  rc |= RunCase("A fp0 rows64", 0, 0, 64, 1024, 1024, false, true);
  // B: unaligned 17 -- history segment, tile at 16 mixes both.
  rc |= RunCase("B unaligned17 rows48", 17, 17, 48, 1024, 1024, false, false);
  // C: mid-window sliding -- window 64 over positions 200..263.
  rc |= RunCase("C midwindow rows64", 200, 200, 64, 64, 64, false, false);
  // D: partial final tile -- 40 keys, last tile holds 8.
  rc |= RunCase("D partialtile rows40", 0, 0, 40, 1024, 1024, false, false);
  // E1: rows < Bc(16) through the production dispatch (split
  // attention; the descriptor is not consumed there by design).
  rc |= RunCase("E1 rows8 dispatch", 0, 0, 8, 1024, 1024, false, false);
  // E2: rows < Bc(16) forced through the WMMA kernel directly.
  rc |= RunCase("E2 rows8 wmma", 0, 0, 8, 1024, 1024, true, false);
  // F: partial query block + partial tile, 24 rows at fp 0.
  rc |= RunCase("F rows24", 0, 0, 24, 1024, 1024, false, false);
  std::cout << (g_fail == 0 && rc == 0 ? "ALL BIT-IDENTICAL\n" : "FAILURES\n");
  return (g_fail == 0 && rc == 0) ? 0 : 1;
}

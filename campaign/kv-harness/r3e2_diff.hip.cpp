// R3-E2 differential test: global D=512 derived-key shapes.
// Bypass (current unset) vs staged (current set) must be
// bit-identical in every shape.
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

constexpr std::uint32_t D = 512;
constexpr std::uint32_t kPairs = 64;
constexpr float kEps = 1e-6F;
const float kThetaScale = std::pow(1000000.0F, -2.0F / 512.0F);

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
  float* dff = nullptr;
  bool use_ff = false;
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
            std::uint32_t kvh_, bool fused_, std::uint32_t seed, bool ff_) {
    use_ff = ff_;
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
    const std::size_t krowb =
        KvCacheRowBytes(kvh * 2 * kPairs, KvCacheFormat::kQ8_0);
    const std::size_t vrowb = KvCacheRowBytes(kvh * D, KvCacheFormat::kQ8_0);
    std::vector<std::uint8_t> kcache(ring * krowb, 0);
    std::vector<std::uint8_t> vcache(ring * vrowb, 0);
    dkc = Dev(kcache);
    dvc = Dev(vcache);
    std::vector<std::uint16_t> kf16(ring * kvh * 2 * kPairs, 0);
    std::vector<std::uint16_t> vf16(ring * kvh * D, 0);
    dkf = Dev(kf16);
    dvf = Dev(vf16);
    std::vector<float> ff(256);
    std::uniform_real_distribution<float> fud(1.0F, 2.0F);
    for (auto& x : ff) x = fud(rng);
    dff = Dev(ff);
    std::vector<float> part(k::AttentionPartialFloats(256, heads, D, 8192));
    dpart = Dev(part);
    return (dbase && dqn && dkn && dqout && dkc && dvc && dkf && dvf && dff &&
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
    post.freq_factors = use_ff ? dff : nullptr;
    post.k_cache = dkf;
    post.v_cache = dvf;
    post.k_cache_q8 = dkc;
    post.v_cache_q8 = dvc;
    post.rows = rows;
    post.heads = heads;
    post.kv_heads = kvh;
    post.head_dim = D;
    post.first_position = fp;
    post.ring = 0;
    post.eps = kEps;
    post.rotated_pairs = kPairs;
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
    a.window = 0;
    a.ring = 0;
    a.rope_pairs = kPairs;
    a.key_ends = nullptr;
    a.out_half = nullptr;
    a.siblings = 0;
    a.spare_key = 0;
    if (staged) {
      a.current.k = dk + std::size_t{fp} * KvRow();
      a.current.v = dv + std::size_t{fp} * VRow();
      a.current.k_norm = dkn;
      a.current.theta_scale = kThetaScale;
      a.current.freq_factors = use_ff ? dff : nullptr;
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
        std::uint32_t window, std::uint32_t ring, bool ff = false) {
  Fixture f;
  if (f.Build(fp + rows, ring, heads, kvh, fused, 777, ff)) {
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
  // A: fp0 rows27 -- full tile + partial final tile, fused.
  rc |= Run("A fused h32 kv4 rows27", 32, 4, true, 0, 0, 27, 0, 27);
  // B: fp0 rows64 -- full tiles only, fused.
  rc |= Run("B fused h32 kv4 rows64", 32, 4, true, 0, 0, 64, 0, 64);
  // C: rows27 packed layout.
  rc |= Run("C packed h32 kv4 rows27", 32, 4, false, 0, 0, 27, 0, 27);
  // D: unaligned first_position 27 with 27 keys of history.
  rc |= Run("D fused h32 kv4 hist27 fp27 rows27", 32, 4, true, 27, 27, 27, 0, 54);
  // E: freq_factors on, rows33.
  rc |= Run("E fused h16 kv2 rows33 ff", 16, 2, true, 0, 0, 33, 0, 33, true);
  // F: freq_factors on, rows17 (one full tile + 1-key tile).
  rc |= Run("F fused h32 kv4 rows17 ff", 32, 4, true, 0, 0, 17, 0, 17, true);
  // G: single KV head, packed, rows48.
  rc |= Run("G packed h8 kv1 rows48", 8, 1, false, 0, 0, 48, 0, 48);
  // H: history 64, fp64, rows27 (mixed history/current tiles).
  rc |= Run("H fused h32 kv4 hist64 fp64 rows27", 32, 4, true, 64, 64, 27, 0, 91);
  std::cout << (g_fail == 0 && rc == 0 ? "ALL BIT-IDENTICAL\n" : "FAILURES\n");
  return (g_fail == 0 && rc == 0) ? 0 : 1;
}

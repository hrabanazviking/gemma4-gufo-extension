// KV REDO R6 E2 -- int8-WMMA QK spike (MICROBENCHMARK ONLY).
// One 16x64 score tile: S[16][64] = Q[16][D] * K[64][D]^T, D in {512, 256}.
// Variants: A bf16/f16 WMMA reference (fused R2 path proxy),
// B pre-quantized Q (int8 WMMA, per-32 i32 partials flushed to an
// fp32 master with separable scale sQ*sK), C runtime-quantized Q
// (Q quantized inside the timed loop, then exactly B's QK path).
// NOT a shipping kernel. Nothing here is integrated anywhere.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using v4i = __attribute__((__vector_size__(4 * sizeof(std::int32_t)))) std::int32_t;
using v8i = __attribute__((__vector_size__(8 * sizeof(std::int32_t)))) std::int32_t;

__device__ __forceinline__ v8f WmmaF16(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8i WmmaI8(v4i a, v4i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

// mode: 0=A, 1=B, 2=C. D is a template parameter (512 or 256).
template <int D>
__global__ void TileKernel(int mode, const __half* __restrict__ qh,
                           const __half* __restrict__ kh,
                           const std::int8_t* __restrict__ qi,
                           const std::int8_t* __restrict__ ki,
                           const float* __restrict__ sq,
                           const float* __restrict__ sk, int iters,
                           float* __restrict__ out, float* __restrict__ sink) {
  const int lane = threadIdx.x & 31;
  const int rowl = lane & 15;          // fragment row / column lane
  __shared__ std::int8_t qsh[16 * 512];
  __shared__ float sqsh[16 * 16];
  float tot = 0.0f;
  for (int it = 0; it < iters; ++it) {
    const std::int8_t* qsrc = qi;
    const float* sqsrc = sq;
    int sqstride = D / 32;
    if (mode == 2) {
      // Runtime quantization of Q: per (row, 32-block) amax scale.
      const int nb = D / 32;
      for (int idx = lane; idx < 16 * nb; idx += 32) {
        const int r = idx / nb, b = idx % nb;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j)
          amax = fmaxf(amax, fabsf(__half2float(qh[r * D + b * 32 + j])));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        sqsh[r * 16 + b] = s;
        const float inv = 1.0f / s;
        for (int j = 0; j < 32; ++j)
          qsh[r * D + b * 32 + j] = (std::int8_t)rintf(
              __half2float(qh[r * D + b * 32 + j]) * inv);
      }
      __syncthreads();
      qsrc = qsh;
      sqsrc = sqsh;
      sqstride = 16;
    }
    for (int kb = 0; kb < 4; ++kb) {  // 4 key blocks of 16 -> 64 keys
      float master[8];
      for (int i = 0; i < 8; ++i) master[i] = 0.0f;
      const int key0 = kb * 16 + rowl;  // this lane's B column (key)
      for (int blk = 0; blk < D / 32; ++blk) {
        if (mode == 0) {
          v8f acc{};
          for (int s = 0; s < 2; ++s) {
            const int k0 = blk * 32 + s * 16;
            v16h af{}, bf{};
            for (int i = 0; i < 16; ++i) {
              af[i] = qh[rowl * D + k0 + i];
              bf[i] = kh[key0 * D + k0 + i];
            }
            acc = WmmaF16(af, bf, acc);
          }
          for (int i = 0; i < 8; ++i) master[i] += acc[i];
        } else {
          v8i iacc{};
          for (int s = 0; s < 2; ++s) {
            const int k0 = blk * 32 + s * 16;
            v4i af{}, bf{};
            const auto* qa = reinterpret_cast<const std::int32_t*>(qsrc + (std::size_t)rowl * D + k0);
            const auto* ka = reinterpret_cast<const std::int32_t*>(ki + (std::size_t)key0 * D + k0);
            for (int i = 0; i < 4; ++i) { af[i] = qa[i]; bf[i] = ka[i]; }
            iacc = WmmaI8(af, bf, iacc);
          }
          // Per-element flush: C element i is row 2i+lane/16, col lane%16.
          for (int i = 0; i < 8; ++i) {
            const int r = 2 * i + (lane >> 4);
            master[i] += (float)iacc[i] * (sqsrc[r * sqstride + blk] *
                                           sk[key0 * (D / 32) + blk]);
          }
        }
      }
      if (iters == 1 && blockIdx.x == 0) {
        for (int i = 0; i < 8; ++i) {
          const int r = 2 * i + (lane >> 4);
          out[r * 64 + kb * 16 + rowl] = master[i];
        }
      }
      for (int i = 0; i < 8; ++i) tot += master[i];
    }
    if (mode == 2) __syncthreads();
  }
  if (tot == 3.4028235e38f) *sink = tot;  // never true; defeats DCE
}

static float QuantBlock(const float* x, std::int8_t* q) {
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
  const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
  for (int j = 0; j < 32; ++j) q[j] = (std::int8_t)std::rint(x[j] / s);
  return s;
}

template <int D>
int RunD() {
  const int nb = D / 32;
  std::vector<float> qf(16 * D), kf(64 * D);
  for (int i = 0; i < 16 * D; ++i) qf[i] = std::sin(i * 0.371f) * 1.7f + std::cos(i * 0.013f) * 0.4f;
  for (int i = 0; i < 64 * D; ++i) kf[i] = std::sin(i * 0.529f + 1.3f) * 1.3f + std::cos(i * 0.007f) * 0.5f;
  std::vector<__half> qh(16 * D), kh(64 * D);
  for (int i = 0; i < 16 * D; ++i) qh[i] = __float2half(qf[i]);
  for (int i = 0; i < 64 * D; ++i) kh[i] = __float2half(kf[i]);
  // Host-side per-32 quantization (B's pre-quantized inputs; K same for C).
  std::vector<std::int8_t> qi(16 * D), ki(64 * D);
  std::vector<float> sq(16 * nb), sk(64 * nb);
  for (int r = 0; r < 16; ++r)
    for (int b = 0; b < nb; ++b) sq[r * nb + b] = QuantBlock(&qf[r * D + b * 32], &qi[r * D + b * 32]);
  for (int r = 0; r < 64; ++r)
    for (int b = 0; b < nb; ++b) sk[r * nb + b] = QuantBlock(&kf[r * D + b * 32], &ki[r * D + b * 32]);

  __half *dqh, *dkh; std::int8_t *dqi, *dki; float *dsq, *dsk, *dout, *dsink;
  hipMalloc(&dqh, qh.size() * 2); hipMalloc(&dkh, kh.size() * 2);
  hipMalloc(&dqi, qi.size()); hipMalloc(&dki, ki.size());
  hipMalloc(&dsq, sq.size() * 4); hipMalloc(&dsk, sk.size() * 4);
  hipMalloc(&dout, 16 * 64 * 4); hipMalloc(&dsink, 4);
  hipMemcpy(dqh, qh.data(), qh.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(dkh, kh.data(), kh.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(dqi, qi.data(), qi.size(), hipMemcpyHostToDevice);
  hipMemcpy(dki, ki.data(), ki.size(), hipMemcpyHostToDevice);
  hipMemcpy(dsq, sq.data(), sq.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dsk, sk.data(), sk.size() * 4, hipMemcpyHostToDevice);

  const char* names[3] = {"A bf16-ref", "B prequant-Q", "C runtime-quant-Q"};
  double medians[3] = {0, 0, 0};
  for (int mode = 0; mode < 3; ++mode) {
    // Correctness: single tile, compare vs fp32 CPU reference.
    hipMemset(dout, 0, 16 * 64 * 4);
    TileKernel<D><<<1, 32>>>(mode, dqh, dkh, dqi, dki, dsq, dsk, 1, dout, dsink);
    auto err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::printf("d%d %s CHECK LAUNCH FAIL: %s\n", D, names[mode], hipGetErrorString(err)); return 1; }
    std::vector<float> out(16 * 64);
    hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost);
    double maxerr = 0.0;
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 64; ++c) {
        double ref = 0.0;
        for (int j = 0; j < D; ++j) ref += (double)qf[r * D + j] * kf[c * D + j];
        maxerr = std::max(maxerr, std::fabs(out[r * 64 + c] - ref));
      }
    std::printf("d%d %-16s check maxabs err vs fp32 ref: %.4f\n", D, names[mode], maxerr);
    // Timing: grid of independent waves, iters tiles each.
    const int grid = 256, iters = 200;
    TileKernel<D><<<grid, 32>>>(mode, dqh, dkh, dqi, dki, dsq, dsk, 20, dout, dsink);
    hipDeviceSynchronize();
    std::vector<double> ts;
    for (int rep = 0; rep < 7; ++rep) {
      hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
      hipEventRecord(a);
      TileKernel<D><<<grid, 32>>>(mode, dqh, dkh, dqi, dki, dsq, dsk, iters, dout, dsink);
      hipEventRecord(b); hipEventSynchronize(b);
      float ms = 0; hipEventElapsedTime(&ms, a, b);
      ts.push_back((double)ms * 1e6 / ((double)grid * iters));  // ns per tile
      hipEventDestroy(a); hipEventDestroy(b);
    }
    std::sort(ts.begin(), ts.end());
    medians[mode] = ts[3];
    std::printf("d%d %-16s ns/tile reps:", D, names[mode]);
    for (double v : ts) std::printf(" %.1f", v);
    std::printf("  MEDIAN %.1f  spread %.1f%%\n", ts[3],
                100.0 * (ts[6] - ts[0]) / ts[3]);
  }
  std::printf("d%d VERDICT B vs A: B %.1f ns vs A %.1f ns -> B %s A (%.2fx)\n",
              D, medians[1], medians[0],
              medians[1] < medians[0] ? "BEATS" : "does NOT beat",
              medians[0] / medians[1]);
  hipFree(dqh); hipFree(dkh); hipFree(dqi); hipFree(dki);
  hipFree(dsq); hipFree(dsk); hipFree(dout); hipFree(dsink);
  return 0;
}

int main() {
  int n = 0; auto e = hipGetDeviceCount(&n);
  if (e != hipSuccess || n < 1) { std::printf("no HIP device: %s\n", hipGetErrorString(e)); return 1; }
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  std::printf("device: %s gcnArch=%s\n", p.name, p.gcnArchName);
  int rc = 0;
  rc |= RunD<512>();
  rc |= RunD<256>();
  return rc;
}

// Split-KV attention for decoding: one query token per sequence, FP16 or BF16, grouped-query heads.
//
// The throughput test (throughput/decode_bench.py) runs it in every decode step of the dense and the W2 model alike,
// in place of PyTorch's SDPA: at batch 1 SDPA picks its memory-efficient kernel, whose one block per query head leaves
// most SMs idle, and transformers first copies K and V once per query head.
//
// Block (split s, KV head h, sequence b) covers keys [s * chunk, min(L, (s + 1) * chunk)) for the G query heads of
// KV head h, in tiles of 64 keys with an online softmax, and leaves the unnormalized output, the running maximum and
// the sum of exponentials of each head in a workspace. The last block of (b, h) to finish (a counter per (b, h),
// which that block resets) merges the splits and writes the output. At batch 1 the kernel's time is a chain of
// memory round trips, so it keeps that chain short: the loads of q overlap those of the first tile, and the merge
// takes two rounds of loads (one warp per head turns the maxima and sums of its splits into weights, then each
// thread reads its outputs of up to 16 splits at once). Keys whose mask byte is 0 are skipped.

#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
#else
#include <cstdint>

#include <cuda_runtime.h>
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 128;
constexpr int kTile = 64;   // keys per tile; score phase: two threads per key, output phase: 32 keys per 64 threads
constexpr int kMaxCounters = 4096;
constexpr int kMaxSplits = 64;  // blocks per (b, KV head)
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

template <typename T>
__device__ __forceinline__ float2 to_float2(uint32_t u);
template <>
__device__ __forceinline__ float2 to_float2<__half>(uint32_t u) {
  return __half22float2(*reinterpret_cast<const __half2*>(&u));
}
template <>
__device__ __forceinline__ float2 to_float2<__nv_bfloat16>(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}

template <typename T>
__device__ __forceinline__ float to_float(unsigned short u);
template <>
__device__ __forceinline__ float to_float<__half>(unsigned short u) {
  return __half2float(__ushort_as_half(u));
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(unsigned short u) {
  return __bfloat162float(__ushort_as_bfloat16(u));
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

// DPT consecutive values of one V row (DPT = D / 64 dims per thread in the output phase)
template <int DPT>
struct VWord;
template <>
struct VWord<1> {
  typedef unsigned short type;
};
template <>
struct VWord<2> {
  typedef uint32_t type;
};
template <>
struct VWord<4> {
  typedef uint2 type;
};

template <typename T>
__device__ __forceinline__ void unpack(unsigned short w, float (&f)[1]) {
  f[0] = to_float<T>(w);
}
template <typename T>
__device__ __forceinline__ void unpack(uint32_t w, float (&f)[2]) {
  const float2 a = to_float2<T>(w);
  f[0] = a.x;
  f[1] = a.y;
}
template <typename T>
__device__ __forceinline__ void unpack(uint2 w, float (&f)[4]) {
  const float2 a = to_float2<T>(w.x), b = to_float2<T>(w.y);
  f[0] = a.x;
  f[1] = a.y;
  f[2] = b.x;
  f[3] = b.y;
}

}  // namespace

// Grid (splits, Hkv, B), kThreads threads. q [B, Hq, D] with strides (qb, qh); k and v [B, Hkv, L, D] with strides
// (kb, kh, kl) and (vb, vh, vl); mask: L bytes per sequence (stride mb) or null; out [B, Hq, D] contiguous;
// work: B * Hq * splits records of D + 2 floats; counters: B * Hkv zeros (left zero again).
template <typename T, int D, int G>
__global__ void __launch_bounds__(kThreads)
    sph_decode_attention_run(const T* __restrict__ q, long long qb, long long qh, const T* __restrict__ k,
                             long long kb, long long kh, long long kl, const T* __restrict__ v, long long vb,
                             long long vh, long long vl, const unsigned char* __restrict__ mask, long long mb, int L,
                             int chunk, float scale_log2, float* __restrict__ work, unsigned int* __restrict__ counters,
                             T* __restrict__ out) {
  constexpr int DH = D / 2;   // dims per thread in the score phase
  constexpr int KW = DH / 8;  // uint4 words per thread and key
  constexpr int DPT = D / 64;
  constexpr int QS = DH + 4;  // padded half row of q: the two halves read by a lane pair fall in different banks
  typedef typename VWord<DPT>::type VW;
  __shared__ __align__(16) float qs[G][2][QS];
  __shared__ __align__(16) float ps[G][kTile];
  __shared__ float red[2][G][kThreads / 32];
  __shared__ float accs[G][D];
  __shared__ float wsplit[G][kMaxSplits];
  __shared__ bool last;

  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int split = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int splits = gridDim.x, hkv = gridDim.y, hq = hkv * G;
  const int kt = t >> 1, half = t & 1;    // score phase: key kt of the tile, dims half * DH ..
  const int kg = t >> 6, dc = t & 63;     // output phase: keys kg * 32 .. of the tile, dims dc * DPT ..
  const T* kbase = k + b * kb + hk * kh;
  const T* vbase = v + b * vb + hk * vh;
  const unsigned char* mrow = mask == nullptr ? nullptr : mask + b * mb;

  float m_run[G], l_run[G], acc[G][DPT];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = neg_inf();
    l_run[g] = 0.0f;
#pragma unroll
    for (int e = 0; e < DPT; ++e) acc[g][e] = 0.0f;
  }

  const int key_lo = split * chunk, key_hi = min(L, key_lo + chunk);
  for (int k0 = key_lo; k0 < key_hi; k0 += kTile) {
    // every load of the tile first: this thread's half row of K and its 32 V slices
    const int key = k0 + kt;
    const bool in = key < key_hi;
    uint4 kr[KW];
    const uint4* krow = reinterpret_cast<const uint4*>(kbase + key * kl + half * DH);
#pragma unroll
    for (int i = 0; i < KW; ++i) kr[i] = in ? __ldg(krow + i) : make_uint4(0, 0, 0, 0);
    VW vr[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int vk = k0 + kg * 32 + j;
      vr[j] = vk < key_hi ? __ldg(reinterpret_cast<const VW*>(vbase + vk * vl + dc * DPT)) : VW{};
    }
    const bool valid = in && (mrow == nullptr || mrow[key] != 0);
    if (k0 == key_lo) {  // q, scaled, once; its loads overlap those of the first tile
      for (int i = t; i < G * D; i += kThreads) {
        const int g = i / D, d = i % D;
        const unsigned short u = reinterpret_cast<const unsigned short*>(q)[b * qb + (hk * G + g) * qh + d];
        qs[g][d / DH][d % DH] = to_float<T>(u) * scale_log2;
      }
    }
    __syncthreads();  // q staged (first tile); ps and red free again (later tiles)

    float sc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) sc[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < KW; ++i) {
      const uint32_t w[4] = {kr[i].x, kr[i].y, kr[i].z, kr[i].w};
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const float2 k01 = to_float2<T>(w[2 * p]), k23 = to_float2<T>(w[2 * p + 1]);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const float4 q4 = *reinterpret_cast<const float4*>(&qs[g][half][8 * i + 4 * p]);
          sc[g] = fmaf(q4.x, k01.x, fmaf(q4.y, k01.y, fmaf(q4.z, k23.x, fmaf(q4.w, k23.y, sc[g]))));
        }
      }
    }
    float mx[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      sc[g] += __shfl_xor_sync(0xffffffffu, sc[g], 1);
      if (!valid) sc[g] = neg_inf();
      mx[g] = sc[g];
#pragma unroll
      for (int o = 16; o > 1; o >>= 1) mx[g] = fmaxf(mx[g], __shfl_xor_sync(0xffffffffu, mx[g], o));
    }
    if (lane == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g) red[0][g][warp] = mx[g];
    }
    __syncthreads();

    float alpha[G], sm[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float tile_max = red[0][g][0];
#pragma unroll
      for (int w = 1; w < kThreads / 32; ++w) tile_max = fmaxf(tile_max, red[0][g][w]);
      const float m_new = fmaxf(m_run[g], tile_max);
      const float base = m_new == neg_inf() ? 0.0f : m_new;  // no key valid yet: every weight stays 0
      alpha[g] = exp2f(m_run[g] - base);
      m_run[g] = m_new;
      const float p = exp2f(sc[g] - base);
      if (half == 0) ps[g][kt] = p;
      sm[g] = half == 0 ? p : 0.0f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sm[g] += __shfl_xor_sync(0xffffffffu, sm[g], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g) red[1][g][warp] = sm[g];
    }
    __syncthreads();

#pragma unroll
    for (int g = 0; g < G; ++g) {
      float tile_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kThreads / 32; ++w) tile_sum += red[1][g][w];
      l_run[g] = l_run[g] * alpha[g] + tile_sum;
#pragma unroll
      for (int e = 0; e < DPT; ++e) acc[g][e] *= alpha[g];
    }
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      float vf[4][DPT];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) unpack<T>(vr[j + jj], vf[jj]);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float4 p4 = *reinterpret_cast<const float4*>(&ps[g][kg * 32 + j]);
#pragma unroll
        for (int e = 0; e < DPT; ++e)
          acc[g][e] = fmaf(p4.x, vf[0][e], fmaf(p4.y, vf[1][e], fmaf(p4.z, vf[2][e], fmaf(p4.w, vf[3][e], acc[g][e]))));
      }
    }
  }

  // the two key groups meet; one record per query head: D unnormalized outputs, the maximum, the sum
  if (kg == 1) {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int e = 0; e < DPT; ++e) accs[g][dc * DPT + e] = acc[g][e];
  }
  __syncthreads();
  const long long rec0 = (static_cast<long long>(b) * hq + hk * G) * splits + split;  // record of head g: + g * splits
  if (kg == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float* r = work + (rec0 + static_cast<long long>(g) * splits) * (D + 2);
#pragma unroll
      for (int e = 0; e < DPT; ++e) r[dc * DPT + e] = acc[g][e] + accs[g][dc * DPT + e];
      if (dc == 0) {
        r[D] = m_run[g];
        r[D + 1] = l_run[g];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (t == 0) last = atomicAdd(&counters[b * hkv + hk], 1u) == static_cast<unsigned int>(splits - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  // weight of split s for head g: exp2(m_s - max) / sum over the splits of exp2(m_s - max) * l_s (0 if no key valid)
  // records of the G heads; head g, split s at + (g * splits + s) * (D + 2)
  const float* rbase = work + (static_cast<long long>(b) * hq + hk * G) * splits * (D + 2);
  for (int g = warp; g < G; g += kThreads / 32) {
    const float* r = rbase + static_cast<long long>(g) * splits * (D + 2);
    float mv[kMaxSplits / 32], lv[kMaxSplits / 32], mx = neg_inf();
#pragma unroll
    for (int j = 0; j < kMaxSplits / 32; ++j) {
      const int s = lane + 32 * j;
      mv[j] = s < splits ? __ldcg(r + s * (D + 2) + D) : neg_inf();
      lv[j] = s < splits ? __ldcg(r + s * (D + 2) + D + 1) : 0.0f;
      mx = fmaxf(mx, mv[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxSplits / 32; ++j) {
      mv[j] = mx == neg_inf() ? 0.0f : exp2f(mv[j] - mx);
      sum = fmaf(mv[j], lv[j], sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxSplits / 32; ++j)
      if (lane + 32 * j < splits) wsplit[g][lane + 32 * j] = mv[j] * inv;
  }
  __syncthreads();
  constexpr int NO = (G * D + kThreads - 1) / kThreads;  // outputs per thread
  float o[NO];
#pragma unroll
  for (int j = 0; j < NO; ++j) o[j] = 0.0f;
  constexpr int SU = NO <= 4 ? 16 : 32 / NO;  // splits per round of loads (up to 64 records in flight)
  for (int s0 = 0; s0 < splits; s0 += SU) {
    float ov[NO][SU];
#pragma unroll
    for (int u = 0; u < SU; ++u)
#pragma unroll
      for (int j = 0; j < NO; ++j) {
        const int i = t + j * kThreads, g = i / D, d = i % D;
        ov[j][u] = i < G * D && s0 + u < splits
                       ? __ldcg(rbase + (static_cast<long long>(g) * splits + s0 + u) * (D + 2) + d) : 0.0f;
      }
#pragma unroll
    for (int u = 0; u < SU; ++u)
#pragma unroll
      for (int j = 0; j < NO; ++j) {
        const int i = t + j * kThreads, g = i / D;
        if (s0 + u < splits) o[j] = fmaf(wsplit[g][s0 + u], ov[j][u], o[j]);
      }
  }
#pragma unroll
  for (int j = 0; j < NO; ++j) {
    const int i = t + j * kThreads, g = i / D, d = i % D;
    if (i < G * D) out[(static_cast<long long>(b) * hq + hk * G + g) * D + d] = from_float<T>(o[j]);
  }
  if (t == 0) counters[b * hkv + hk] = 0;
}

#ifndef __CUDACC_RTC__
namespace {

struct Args {
  const void *q, *k, *v;
  long long qb, qh, kb, kh, kl, vb, vh, vl;
  const unsigned char* mask;
  long long mb;
  int B, Hkv, L, chunk, splits;
  float scale_log2;
  float* work;
  unsigned int* counters;
  void* out;
  cudaStream_t stream;
};

template <typename T, int D, int G>
int launch(const Args& a) {
  const dim3 grid(static_cast<unsigned>(a.splits), static_cast<unsigned>(a.Hkv), static_cast<unsigned>(a.B));
  sph_decode_attention_run<T, D, G><<<grid, kThreads, 0, a.stream>>>(
      static_cast<const T*>(a.q), a.qb, a.qh, static_cast<const T*>(a.k), a.kb, a.kh, a.kl,
      static_cast<const T*>(a.v), a.vb, a.vh, a.vl, a.mask, a.mb, a.L, a.chunk, a.scale_log2, a.work, a.counters,
      static_cast<T*>(a.out));
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <typename T, int D>
int by_group(const Args& a, int G) {
  switch (G) {
    case 1: return launch<T, D, 1>(a);
    case 2: return launch<T, D, 2>(a);
    case 4: return launch<T, D, 4>(a);
    case 8: return launch<T, D, 8>(a);
    default: return 1;
  }
}

template <typename T>
int by_dim(const Args& a, int D, int G) {
  switch (D) {
    case 64: return by_group<T, 64>(a, G);
    case 128: return by_group<T, 128>(a, G);
    case 256: return by_group<T, 256>(a, G);
    default: return 1;
  }
}

bool aligned(const void* p, uintptr_t bytes) { return p != nullptr && reinterpret_cast<uintptr_t>(p) % bytes == 0; }

}  // namespace

// out = softmax(scale * q k^T, keys with mask byte != 0) v for one query token per sequence. dtype 0: FP16,
// 1: BF16. Head dim D in {64, 128, 256}, Hq / Hkv in {1, 2, 4, 8}; strides in elements, the last dim contiguous.
// splits * chunk >= L, splits at most 64. work: B * Hq * splits * (D + 2) floats; counters: B * Hkv zeroed ints, at
// most 4096.
// Returns 0 on success, 1 bad shape or argument, 2 null or misaligned pointer, 3 launch error.
extern "C" int sph_decode_attention(int dtype, const void* q, long long qb, long long qh, const void* k, long long kb,
                                    long long kh, long long kl, const void* v, long long vb, long long vh,
                                    long long vl, const void* mask, long long mb, int B, int Hq, int Hkv, int L,
                                    int D, int chunk, int splits, float scale, void* work, void* counters, void* out,
                                    void* stream) {
  if (B < 1 || B > 65535 || Hkv < 1 || Hkv > 65535 || Hq % Hkv != 0 || L < 1 || chunk < 1 || splits < 1 ||
      splits > kMaxSplits ||
      static_cast<long long>(chunk) * splits < L || static_cast<long long>(chunk) * (splits - 1) >= L ||
      static_cast<long long>(B) * Hkv > kMaxCounters || (dtype != 0 && dtype != 1))
    return 1;
  if (!aligned(q, 2) || !aligned(k, 16) || !aligned(v, 8) || !aligned(work, 16) || !aligned(counters, 4) ||
      !aligned(out, 2) || (kl * 2) % 16 != 0 || (kb * 2) % 16 != 0 || (kh * 2) % 16 != 0 || (vl * 2) % 8 != 0 ||
      (vb * 2) % 8 != 0 || (vh * 2) % 8 != 0)
    return 2;
  Args a;
  a.q = q, a.k = k, a.v = v;
  a.qb = qb, a.qh = qh, a.kb = kb, a.kh = kh, a.kl = kl, a.vb = vb, a.vh = vh, a.vl = vl;
  a.mask = static_cast<const unsigned char*>(mask), a.mb = mb;
  a.B = B, a.Hkv = Hkv, a.L = L, a.chunk = chunk, a.splits = splits;
  a.scale_log2 = scale * kLog2e;
  a.work = static_cast<float*>(work), a.counters = static_cast<unsigned int*>(counters), a.out = out;
  a.stream = static_cast<cudaStream_t>(stream);
  return dtype == 0 ? by_dim<__half>(a, D, Hq / Hkv) : by_dim<__nv_bfloat16>(a, D, Hq / Hkv);
}
#endif

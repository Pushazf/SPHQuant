// Fused SPHQuant W2 kernels for batch-1 decoding. They read the runtime layout that
// cuda_kernel/packed_linear.py:repack builds at load time from the sphquant-w2-v1 export. That layout has the same
// 2.5 bits per weight and is arranged so that every lane decodes with wide loads and few instructions.
//
// Runtime layout. For row n and input group g (128 inputs, i.e. 16 vectors of 8), 40 bytes are spread over three
// row-major arrays [N][K/128]:
//   A uint4   P0 P1 P2 P3   sign planes: bit j of P_k is the sign of coordinate 2k of vector j, bit 16+j that of 2k+1
//   B uint4   D0 D1 D2 Q0
//   C uint2   Q1 M
// D_r holds the direction indices of vectors 5r..5r+4 in bits 6i..6i+5 (i = 0..4). The index of vector 15 is split
// over bits 30-31 of D0 (its bits 0-1), D1 (bits 2-3) and D2 (bits 4-5).
// Q_h holds the 4-bit radius codes of vectors 8h..8h+7: vector 8h+2p in bits 4p..4p+3, vector 8h+2p+1 in bits
// 16+4p..16+4p+3 (p = 0..3).
// M = half2(step, rmin) of the group.
// The weight of row n at input 128g + 8j + i is (rmin + code_j * step) * codebook[s][dir_j][i] * (-1 if sign bit),
// where s is the segment of row n. A fused module (q, k, v or gate, up) stacks the rows of its parts, each part
// with its own 64 x 8 codebook; ends[s] is the first row after segment s.
//
// sph_gemv_run: a block stages x and the codebooks in shared memory once and covers a range of rows; each warp
// takes one input slice of 4096 (lane l: group 32 * slice + l, its 128 inputs in registers) and every
// (8 / slices)-th row of the range, reading the 40 bytes of the row's group and adding up its 16 vector dots. The
// codebook sits in shared memory as 8 copies, one per lane % 8, so the 16-byte row reads of a quarter warp never
// conflict. Signs are XORed into the FP16 codebook values (all codebook entries are nonnegative), the 8 products of
// a vector are summed in FP16 (two partial sums of 4), scaled by the FP16 radius, summed over four vectors in FP16
// and accumulated in FP32. Warps of the same rows but different slices meet in shared memory.
// GLU mode (a fused gate/up projection of N = 2H rows): output r < H is silu(row r) * (row H + r), so the MLP needs
// no separate activation kernel; a warp computes the two rows of an output one after the other.
// With programmatic dependent launch (sph_gemv's pdl, compute capability 9.0 and up) a block requests its first
// weight rows and the codebooks while the previous kernel in the stream finishes, and waits for that kernel only
// before it reads x.
// sph_dequant_run writes the dense FP16 weight, for prompts of many tokens (then a regular GEMM).

#ifdef __CUDACC_RTC__
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <climits>
#include <cstdint>

#include <cuda_runtime.h>
#endif
#include <cuda_fp16.h>

namespace {

constexpr int kGroup = 128;          // inputs per group (16 vectors)
constexpr int kSliceGroups = 32;     // groups per slice, one per lane
constexpr int kDirs = 64;
#ifndef SPH_COPIES
#define SPH_COPIES 8
#endif
constexpr int kCopies = SPH_COPIES;  // codebook copies in shared memory, one per lane % kCopies
constexpr int kCopyShift = SPH_COPIES == 8 ? 7 : SPH_COPIES == 16 ? 8 : 9;  // log2(kCopies * 16)
constexpr uint32_t kDirMask = 63u << kCopyShift;
constexpr int kMaxSegs = 4;
constexpr int kSegBytes = kDirs * kCopies * 16;  // 8192 for 8 copies
constexpr int kWarps = 8;            // warps per block (fewer when K / 4096 does not divide 8)
constexpr int kMaxSlices = 8;        // K <= 32768
constexpr int kMaxPartialBytes = 16384;  // rows x slices FP32 partial sums when K > 4096

#ifndef SPH_WEIGHT_LOAD
#define SPH_WEIGHT_LOAD __ldcs  // weights are read once per token: evict them from L2 first
#endif

struct Ends {
  int e[kMaxSegs];
};

__device__ __forceinline__ __half2 h2(uint32_t u) { return *reinterpret_cast<const __half2*>(&u); }

// Segment of row n: unused ends equal N, which no row reaches.
__device__ __forceinline__ int segment_of(int n, const Ends& ends) {
  return (n >= ends.e[0]) + (n >= ends.e[1]) + (n >= ends.e[2]);
}

// Programmatic dependent launch: wait until the previous kernel in the stream has completed and its writes are
// visible (no-op without PDL), and let the next kernel start launching.
__device__ __forceinline__ void wait_for_previous_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void allow_next_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

struct Row {
  uint4 a, b;
  uint2 c;
};

__device__ __forceinline__ Row load_row(const uint4* __restrict__ A, const uint4* __restrict__ B,
                                        const uint2* __restrict__ C, long long idx) {
  Row r;
#ifdef SPH_SYNTHETIC  // development: weights made up in registers, to time the arithmetic alone
  const uint32_t h = static_cast<uint32_t>(idx) * 2654435761u;
  r.a = make_uint4(h, h ^ 0x5bd1e995u, h * 3u, h + 0x9e3779b9u);
  r.b = make_uint4(h * 5u, h ^ 0x27d4eb2du, h * 7u, h & 0x0F0F0F0Fu);
  r.c = make_uint2(h & 0x0F0F0F0Fu, 0x24002000u);
  return r;
#endif
  r.a = SPH_WEIGHT_LOAD(A + idx);
  r.b = SPH_WEIGHT_LOAD(B + idx);
  r.c = SPH_WEIGHT_LOAD(C + idx);
  return r;
}

// (a & 0x000F000F) | c in one LOP3 (c in a register): the 4-bit codes of a vector pair as half2(1024 + code)
__device__ __forceinline__ uint32_t codes_as_half2(uint32_t a, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, 0x000F000F, %2, 0xEA;" : "=r"(r) : "r"(a), "r"(c));
  return r;
}

// Dot of one group of row r with the lane's inputs xv. BASE is the 32-bit shared address of the row's codebook
// copy for this lane: table start (8 KB aligned) + segment * 8192 + (lane % 8) * 16, so OR adds the row offset.
// MAGIC holds 0x64006400 (half2(1024, 1024)).
__device__ __forceinline__ float group_dot(const Row& r, const __half2 (&xv)[16][4], uint32_t base, uint32_t magic) {
  const uint32_t P[4] = {r.a.x, r.a.y, r.a.z, r.a.w};
  const uint32_t D[3] = {r.b.x, r.b.y, r.b.z};
  const uint32_t Q[2] = {r.b.w, r.c.x};
  const __half2 step2 = __low2half2(h2(r.c.y));
  const __half2 rmin2 = __high2half2(h2(r.c.y));
  // (d << kCopyShift) of vector 15, from its three 2-bit pieces
  const uint32_t d15 = (((D[0] >> 30) | ((D[1] >> 28) & 0xCu) | ((D[2] >> 26) & 0x30u)) << kCopyShift);
  float acc = 0.0f;
#pragma unroll
  for (int quad = 0; quad < 4; ++quad) {  // 4 vectors: their radius-scaled dots are summed in FP16
    __half2 t;
#pragma unroll
    for (int pq = 0; pq < 2; ++pq) {
      const int pair = 2 * quad + pq;
      const uint32_t bits = codes_as_half2(Q[pair >> 2] >> (4 * (pair & 3)), magic);
      const __half2 radius = __hfma2(__hsub2(h2(bits), h2(magic)), step2, rmin2);
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const int j = 2 * pair + v;
        uint32_t off;
        if (j == 15) {
          off = d15;
        } else {
          const int f = 6 * (j % 5);
          off = (f <= kCopyShift ? D[j / 5] << (kCopyShift - f) : D[j / 5] >> (f - kCopyShift)) & kDirMask;
        }
        const uint4 cw = lds128(off | base);
        const uint32_t c[4] = {cw.x, cw.y, cw.z, cw.w};
        __half2 h = __hmul2(xv[j][0], h2(c[0] ^ ((P[0] << (15 - j)) & 0x80008000u)));
#pragma unroll
        for (int k = 1; k < 4; ++k) h = __hfma2(xv[j][k], h2(c[k] ^ ((P[k] << (15 - j)) & 0x80008000u)), h);
        const __half2 rj = v ? __high2half2(radius) : __low2half2(radius);
        t = (pq == 0 && v == 0) ? __hmul2(h, rj) : __hfma2(h, rj, t);
      }
    }
    acc += __low2float(t) + __high2float(t);
  }
  return acc;
}

}  // namespace

// y[m, n] = sum_k x[m, k] W[n, k] for n < N, or with glu = 1 y[m, r] = silu(W[r] x[m]) * (W[N / 2 + r] x[m]) for
// r < N / 2. Grid (ceil(outputs / rows_per_block), M); block 32 * slices * wps threads with slices = ceil(K / 4096)
// and wps = max(1, 8 / slices) warps per slice. Dynamic shared memory: up to 8 KB of alignment padding, the codebook
// table (nseg * 8192 bytes), x[m] (2K bytes, swizzled) and, when slices > 1, rows_per_block * slices * (1 + glu)
// FP32 partial sums.
// Warp w covers slice w % slices and outputs rb0 + w / slices + i * wps of the block's outputs [rb0, rb1).
extern "C" __global__ void __launch_bounds__(256, 2)
    sph_gemv_run(const uint4* __restrict__ x, const uint4* __restrict__ A, const uint4* __restrict__ B,
                 const uint2* __restrict__ C, const uint4* __restrict__ codebooks, int nseg, Ends ends,
                 __half* __restrict__ y, int K, int N, int rows_per_block, int glu) {
  extern __shared__ uint4 smem[];
  const int G = K / kGroup;
  const int slices = (G + kSliceGroups - 1) / kSliceGroups;
  const int wps = slices >= kWarps ? 1 : kWarps / slices;
  const int nthreads = 32 * slices * wps;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int slice = warp % slices, wr = warp / slices;
  const int g = slice * kSliceGroups + lane;
  const bool active = g < G;
  const long long m = blockIdx.y;

  // Item i of this warp: output r = rb0 + wr + (i >> glu) * wps, weight row r (+ N / 2 for the odd items with
  // GLU). Two items ahead are in flight while one is summed; the first two are requested before the block stages
  // its shared memory.
  const int outputs = N >> glu, half_rows = glu ? N / 2 : 0;
  const int rb0 = blockIdx.x * rows_per_block, rb1 = min(outputs, rb0 + rows_per_block);
  const int nitems = rb1 - rb0 > wr ? ((rb1 - rb0 - wr + wps - 1) / wps) << glu : 0;
  const auto out_of = [&](int i) { return rb0 + wr + (i >> glu) * wps; };
  const auto idx_of = [&](int i) {
    return static_cast<long long>(out_of(i) + ((i & glu) ? half_rows : 0)) * G + g;
  };
  Row cur{}, nxt{};
  if (active && nitems > 0) cur = load_row(A, B, C, idx_of(0));
  if (active && nitems > 1) nxt = load_row(A, B, C, idx_of(1));

  // Shared memory: [pad to 8 KB][codebook table, nseg * 8192][x, swizzled][partial sums]
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t tab32 = (sbase + (kSegBytes - 1)) & ~static_cast<uint32_t>(kSegBytes - 1);
  uint4* tab4 = smem + (tab32 - sbase) / 16;
  uint4* xs = tab4 + nseg * kDirs * kCopies;
  float* part = reinterpret_cast<float*>(xs + K / 8);
  // Every codebook entry is loaded once (at most two per thread: nseg * 64 <= 256 <= 2 * nthreads) and stored as
  // kCopies copies. x, written by the previous kernel, is read after waiting for it; chunk j of group g goes to
  // g * 16 + (j ^ (g % 8)), so the 16-byte reads of 8 consecutive lanes hit 8 bank groups.
  const int ncb = nseg * kDirs, e0 = threadIdx.x, e1 = threadIdx.x + nthreads;
  uint4 cb0 = make_uint4(0, 0, 0, 0), cb1 = cb0;
  if (e0 < ncb) cb0 = codebooks[e0];
  if (e1 < ncb) cb1 = codebooks[e1];
  wait_for_previous_kernel();
  const uint4* xm = x + m * (K / 8);
  const int nx = K / 8;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // asynchronous copies straight into shared memory: all of them in flight at once, no registers
  const uint32_t xs32 = static_cast<uint32_t>(__cvta_generic_to_shared(xs));
  for (int i = threadIdx.x; i < nx; i += nthreads) {
    const uint32_t dst = xs32 + 16u * static_cast<uint32_t>((i & ~15) | ((i & 15) ^ ((i >> 4) & 7)));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(xm + i) : "memory");
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
#else
  for (int i = threadIdx.x; i < nx; i += nthreads) xs[(i & ~15) | ((i & 15) ^ ((i >> 4) & 7))] = xm[i];
#endif
#pragma unroll
  for (int c = 0; c < kCopies; ++c) {
    if (e0 < ncb) tab4[e0 * kCopies + c] = cb0;
    if (e1 < ncb) tab4[e1 * kCopies + c] = cb1;
  }
  __syncthreads();

  const uint32_t lane_off = static_cast<uint32_t>(lane % kCopies) * 16u;
  __half2 xv[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint4 u = active ? xs[(g << 4) | (j ^ (g & 7))] : make_uint4(0, 0, 0, 0);
    xv[j][0] = h2(u.x);
    xv[j][1] = h2(u.y);
    xv[j][2] = h2(u.z);
    xv[j][3] = h2(u.w);
  }

  const uint32_t magic = 0x64006400u;
  float gate = 0.0f;
  for (int i = 0; i < nitems; ++i) {
    Row nn{};
    if (active && i + 2 < nitems) nn = load_row(A, B, C, idx_of(i + 2));
    const int r = out_of(i), n = r + ((i & glu) ? half_rows : 0);
    const uint32_t base = (tab32 + static_cast<uint32_t>(segment_of(n, ends)) * kSegBytes) | lane_off;
    float s = active ? group_dot(cur, xv, base, magic) : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (slices > 1) {
      if (lane == 0) part[(((r - rb0) << glu) | (i & glu)) * slices + slice] = s;
    } else if (glu && (i & 1) == 0) {
      gate = s;
    } else if (lane == 0) {
      y[m * outputs + r] = __float2half_rn(glu ? silu(gate) * s : s);
    }
    cur = nxt;
    nxt = nn;
  }
  allow_next_kernel();
  if (slices > 1) {
    __syncthreads();
    for (int t = threadIdx.x; t < rb1 - rb0; t += nthreads) {
      float s0 = 0.0f, s1 = 0.0f;
      for (int sl = 0; sl < slices; ++sl) {
        s0 += part[(t << glu) * slices + sl];
        if (glu) s1 += part[((t << 1) | 1) * slices + sl];
      }
      y[m * outputs + rb0 + t] = __float2half_rn(glu ? silu(s0) * s1 : s0);
    }
  }
}

// W[n, :] in FP16 from the runtime layout; one thread per (row, group).
extern "C" __global__ void __launch_bounds__(256)
    sph_dequant_run(const uint4* __restrict__ A, const uint4* __restrict__ B, const uint2* __restrict__ C,
                    const uint4* __restrict__ codebooks, int nseg, Ends ends, uint4* __restrict__ W, int K, int N) {
  const int G = K / kGroup;
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(N) * G) return;
  const int n = static_cast<int>(t / G);
  const uint4* cb = codebooks + segment_of(n, ends) * kDirs;
  const uint4 a = A[t], b = B[t];
  const uint2 c = C[t];
  const uint32_t P[4] = {a.x, a.y, a.z, a.w};
  const uint32_t D[3] = {b.x, b.y, b.z};
  const uint32_t Q[2] = {b.w, c.x};
  const float step = __low2float(h2(c.y)), rmin = __high2float(h2(c.y));
  uint4* out = W + t * 16;
#pragma unroll 4
  for (int j = 0; j < 16; ++j) {
    const uint32_t d = j == 15 ? ((D[0] >> 30) | ((D[1] >> 28) & 0xCu) | ((D[2] >> 26) & 0x30u))
                               : (D[j / 5] >> (6 * (j % 5))) & 63u;
    const int h = j >> 3, p = (j & 7) >> 1;
    const uint32_t code = (Q[h] >> (4 * p + 16 * (j & 1))) & 15u;
    const __half2 r = __float2half2_rn(fmaf(static_cast<float>(code), step, rmin));
    const uint4 cw = cb[d];
    const uint32_t cv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __half2 w = __hmul2(h2(cv[k] ^ (((P[k] >> j) & 1u) << 15) ^ (((P[k] >> (16 + j)) & 1u) << 31)), r);
      o[k] = *reinterpret_cast<const uint32_t*>(&w);
    }
    out[j] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

#ifndef __CUDACC_RTC__
namespace {

bool aligned(const void* p, uintptr_t bytes) { return p != nullptr && reinterpret_cast<uintptr_t>(p) % bytes == 0; }

bool load_ends(const int* ends_in, int nseg, long long N, Ends* ends) {
  if (nseg < 1 || nseg > kMaxSegs || ends_in == nullptr) return false;
  int prev = 0;
  for (int i = 0; i < kMaxSegs; ++i) ends->e[i] = static_cast<int>(N);
  for (int i = 0; i < nseg; ++i) {
    if (ends_in[i] <= prev || ends_in[i] > N) return false;
    ends->e[i] = prev = ends_in[i];
  }
  return prev == N;
}

// Compute capability 9.0 or higher on the current device (programmatic dependent launch), cached per device.
bool pdl_supported() {
  static int major[64] = {};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return false;
  if (dev >= 64) return false;
  if (major[dev] == 0 && cudaDeviceGetAttribute(&major[dev], cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
    return false;
  return major[dev] >= 9;
}

}  // namespace

// Both return 0 on success, 1 bad shape or argument, 2 null or misaligned pointer, 3 launch error.
// ends: nseg increasing row ends (host memory), the last one N. glu != 0: y [M, N / 2] = silu(first half of the
// rows) * (second half), rows_per_block then counts outputs. pdl != 0: allow programmatic dependent launch (ignored
// below compute capability 9.0).
extern "C" int sph_gemv(const void* x, long long M, long long K, const void* a, const void* b, const void* c,
                        const void* codebooks, int nseg, const int* ends_in, long long N, int rows_per_block, int glu,
                        int pdl, void* y, void* stream) {
  Ends ends;
  if (K <= 0 || K % kGroup != 0 || K > kMaxSlices * kSliceGroups * kGroup || M < 0 || M > 65535 || N <= 0 ||
      N > INT_MAX || rows_per_block < 1 || (glu != 0 && N % 2 != 0) || !load_ends(ends_in, nseg, N, &ends))
    return 1;
  if (M == 0) return 0;
  if (!aligned(x, 16) || !aligned(a, 16) || !aligned(b, 16) || !aligned(c, 8) || !aligned(codebooks, 16) ||
      !aligned(y, 2))
    return 2;
  const int g1 = glu != 0 ? 1 : 0;
  const long long outputs = N >> g1;
  const int G = static_cast<int>(K / kGroup);
  const int slices = (G + kSliceGroups - 1) / kSliceGroups;
  const int wps = slices >= kWarps ? 1 : kWarps / slices;
  const long long partial_bytes = slices > 1 ? 4LL * rows_per_block * slices << g1 : 0;
  if (partial_bytes > kMaxPartialBytes) return 1;
  const size_t smem = static_cast<size_t>(nseg + 1) * kSegBytes + 2 * static_cast<size_t>(K) + partial_bytes;
  static size_t smem_opted = 48 * 1024;  // above 48 KB the kernel must opt in (once per larger size)
  if (smem > smem_opted) {
    if (cudaFuncSetAttribute(sph_gemv_run, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) !=
        cudaSuccess)
      return 3;
    smem_opted = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>((outputs + rows_per_block - 1) / rows_per_block), static_cast<unsigned>(M));
  cfg.blockDim = dim3(32 * slices * wps);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl != 0 && pdl_supported() ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, sph_gemv_run, static_cast<const uint4*>(x), static_cast<const uint4*>(a),
                         static_cast<const uint4*>(b), static_cast<const uint2*>(c),
                         static_cast<const uint4*>(codebooks), nseg, ends, static_cast<__half*>(y), static_cast<int>(K),
                         static_cast<int>(N), rows_per_block, g1) != cudaSuccess)
    return 3;
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int sph_dequant(const void* a, const void* b, const void* c, const void* codebooks, int nseg,
                           const int* ends_in, long long K, long long N, void* w, void* stream) {
  Ends ends;
  if (K <= 0 || K % kGroup != 0 || K > INT_MAX || N <= 0 || N > INT_MAX || !load_ends(ends_in, nseg, N, &ends))
    return 1;
  if (!aligned(a, 16) || !aligned(b, 16) || !aligned(c, 8) || !aligned(codebooks, 16) || !aligned(w, 16)) return 2;
  const long long threads = N * (K / kGroup);
  sph_dequant_run<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(a), static_cast<const uint4*>(b), static_cast<const uint2*>(c),
      static_cast<const uint4*>(codebooks), nseg, ends, static_cast<uint4*>(w), static_cast<int>(K),
      static_cast<int>(N));
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
#endif

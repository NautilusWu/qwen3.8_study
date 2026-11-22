// PLAN_22 M1: how fast a tuned CUDA kernel reads GB10 unified memory (no model, no engine).
//
// A read kernel loads 16-byte vectors (uint4), U loads in flight per thread, XORs them into a register and writes
// nothing unless the XOR hits a magic value (so no load can be removed). Knobs: U (1,2,4,8), load hint (plain,
// __ldg, __ldcs), layout (grid-stride or one contiguous chunk per block), threads per block (256, 512, 1024),
// blocks per SM (1, 2, 4, 8). Timing: CUDA events, warm-ups first, median of the timed repetitions.
//
//   bw_tuned sweep   <size_gb> <reps>                     one line of JSON per configuration
//   bw_tuned single  <size_gb> <reps> <cfg>               one configuration, every repetition
//   bw_tuned forward <sizes.txt> <min_bytes> <reps> <cfg> <arena|sep>
//        the weight groups of one decode forward (bytes per line), each in its own buffer (sep) or packed into one
//        allocation at 256-byte alignment (arena); groups below min_bytes skipped; one kernel launch per group, in
//        order; the time of the whole sequence.
// cfg = U,hint,layout,threads,blocks_per_sm   (e.g. 4,0,0,512,4)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static const unsigned MAGIC = 0x9e3779b9u;

template <int HINT>
__device__ __forceinline__ uint4 ld(const uint4* q) {
  if (HINT == 1) return __ldg(q);
  if (HINT == 2) return __ldcs(q);
  return *q;
}

template <int U, int HINT, int LAYOUT>
__global__ void readk(const uint4* __restrict__ p, size_t n, unsigned* out) {
  size_t beg, end, i, stride;
  if (LAYOUT == 0) {            // grid-stride over the whole buffer
    beg = 0; end = n;
    i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    stride = (size_t)gridDim.x * blockDim.x;
  } else {                      // one contiguous chunk per block
    size_t per = (n + gridDim.x - 1) / gridDim.x;
    beg = blockIdx.x * per; end = beg + per < n ? beg + per : n;
    i = beg + threadIdx.x;
    stride = blockDim.x;
  }
  unsigned acc = 0;
  for (; i + (size_t)(U - 1) * stride < end; i += (size_t)U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld<HINT>(p + i + (size_t)u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < end; i += stride) { uint4 v = ld<HINT>(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == MAGIC) out[0] = acc;
}

typedef void (*Kern)(const uint4*, size_t, unsigned*);
static const int US[4] = {1, 2, 4, 8};

template <int U, int H, int L> Kern k3() { return readk<U, H, L>; }
template <int U, int H> Kern k2(int L) { return L ? k3<U, H, 1>() : k3<U, H, 0>(); }
template <int U> Kern k1(int H, int L) { return H == 0 ? k2<U, 0>(L) : (H == 1 ? k2<U, 1>(L) : k2<U, 2>(L)); }
static Kern pick(int U, int H, int L) {
  switch (U) { case 1: return k1<1>(H, L); case 2: return k1<2>(H, L); case 4: return k1<4>(H, L); default: return k1<8>(H, L); }
}

struct Cfg { int U, hint, layout, threads, bps; };
static Cfg parse(const char* s) {
  Cfg c; if (sscanf(s, "%d,%d,%d,%d,%d", &c.U, &c.hint, &c.layout, &c.threads, &c.bps) != 5) { fprintf(stderr, "bad cfg\n"); exit(2); }
  return c;
}

static int g_sms = 0;
static unsigned* g_out = nullptr;

static void launch(const Cfg& c, const uint4* p, size_t n, cudaStream_t s) {
  pick(c.U, c.hint, c.layout)<<<g_sms * c.bps, c.threads, 0, s>>>(p, n, g_out);
}

static std::vector<float> time_it(int reps, int warm, std::function<void()> body) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int w = 0; w < warm; ++w) body();
  CK(cudaDeviceSynchronize());
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a)); body(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float t; CK(cudaEventElapsedTime(&t, a, b)); ms.push_back(t);
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms;
}
static double median(std::vector<float> v) { std::sort(v.begin(), v.end()); size_t m = v.size() / 2; return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]); }

static void print_row(const char* mode, const Cfg& c, double gb, size_t bytes, const std::vector<float>& ms, bool all) {
  double med = median(ms), best = *std::min_element(ms.begin(), ms.end());
  printf("{\"mode\":\"%s\",\"cfg\":\"%d,%d,%d,%d,%d\",\"size_gb\":%.4f,\"bytes\":%zu,\"ms_median\":%.4f,\"ms_best\":%.4f,"
         "\"gb_s_median\":%.2f,\"gb_s_best\":%.2f", mode, c.U, c.hint, c.layout, c.threads, c.bps, gb, bytes, med, best,
         bytes / med / 1e6, bytes / best / 1e6);
  if (all) { printf(",\"ms\":["); for (size_t i = 0; i < ms.size(); ++i) printf("%s%.4f", i ? "," : "", ms[i]); printf("]"); }
  printf("}\n"); fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 2; }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); g_sms = pr.multiProcessorCount;
  CK(cudaMalloc(&g_out, 64));
  fprintf(stderr, "{\"device\":\"%s\",\"sms\":%d,\"max_threads_per_sm\":%d,\"cc\":\"%d.%d\",\"l2_bytes\":%d}\n", pr.name, g_sms,
          pr.maxThreadsPerMultiProcessor, pr.major, pr.minor, pr.l2CacheSize);
  std::string mode = argv[1];
  if (mode == "sweep" || mode == "single") {
    double gb = atof(argv[2]); int reps = atoi(argv[3]);
    size_t bytes = (size_t)(gb * 1e9) / 4096 * 4096, n = bytes / 16;
    uint4* p; CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 1, bytes)); CK(cudaDeviceSynchronize());
    std::vector<Cfg> cfgs;
    if (mode == "single") cfgs.push_back(parse(argv[4]));
    else for (int U : US) for (int h = 0; h < 3; ++h) for (int L = 0; L < 2; ++L) for (int t : {256, 512, 1024}) for (int b : {1, 2, 4, 8})
      cfgs.push_back({U, h, L, t, b});
    for (const Cfg& c : cfgs) {
      auto ms = time_it(reps, 2, [&] { launch(c, p, n, 0); });
      print_row(mode.c_str(), c, gb, bytes, ms, mode == "single");
    }
    CK(cudaFree(p));
  } else if (mode == "forward") {
    FILE* f = fopen(argv[2], "r"); if (!f) { fprintf(stderr, "no sizes file\n"); return 2; }
    size_t minb = strtoull(argv[3], nullptr, 10); int reps = atoi(argv[4]); Cfg c = parse(argv[5]); bool arena = std::string(argv[6]) == "arena";
    std::vector<size_t> sz; unsigned long long v;
    while (fscanf(f, "%llu", &v) == 1) if (v >= minb) sz.push_back((size_t)v / 16 * 16);
    fclose(f);
    size_t total = 0; for (size_t s : sz) total += s;
    std::vector<uint4*> bufs(sz.size());
    uint4* base = nullptr;
    if (arena) {
      size_t off = 0; std::vector<size_t> offs; for (size_t s : sz) { offs.push_back(off); off += (s + 255) / 256 * 256; }
      CK(cudaMalloc(&base, off)); CK(cudaMemset(base, 1, off));
      for (size_t i = 0; i < sz.size(); ++i) bufs[i] = (uint4*)((char*)base + offs[i]);
    } else {
      for (size_t i = 0; i < sz.size(); ++i) { CK(cudaMalloc(&bufs[i], sz[i])); CK(cudaMemset(bufs[i], 1, sz[i])); }
    }
    CK(cudaDeviceSynchronize());
    auto ms = time_it(reps, 2, [&] { for (size_t i = 0; i < sz.size(); ++i) launch(c, bufs[i], sz[i] / 16, 0); });
    char m[64]; snprintf(m, sizeof m, "forward_%s_%zu", arena ? "arena" : "sep", sz.size());
    print_row(m, c, total / 1e9, total, ms, true);
    if (arena) CK(cudaFree(base)); else for (auto b : bufs) CK(cudaFree(b));
  } else { fprintf(stderr, "unknown mode\n"); return 2; }
  CK(cudaFree(g_out));
  return 0;
}

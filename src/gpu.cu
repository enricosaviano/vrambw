// vrambw - real VRAM and PCIe bandwidth of NVIDIA GPUs (CUDA 11.4 / 12.x)
//
// VRAM: read-only streaming kernel over a buffer much larger than L2.
//       Same access pattern as LLM decode (weights streamed once per token).
// PCIe: cudaMemcpy host<->device with pinned memory.
// Units: GB/s = 1e9 bytes/s (same as marketing).

#include "gpu.h"
#include "progress.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x)                                                                 \
    do {                                                                      \
        cudaError_t e_ = (x);                                                 \
        if (e_ != cudaSuccess) {                                              \
            progress_clear();                                                 \
            fprintf(stderr, "CUDA error: %s (%s:%d): %s\n", #x, __FILE__,    \
                    __LINE__, cudaGetErrorString(e_));                        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static const int BLOCK = 256;

__global__ void k_init(int4 *a, size_t n)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned h = (unsigned)i * 2654435761u;
        a[i] = make_int4((int)h, (int)(h ^ 0x5bd1e995u), (int)(h + 1), (int)~h);
    }
}

// 16-byte loads, 4 independent loads in flight per thread, grid-stride.
__global__ void k_read(const int4 *__restrict__ a, size_t n, int *__restrict__ sink, int flag)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    int acc = 0;
    for (; i + 3 * stride < n; i += 4 * stride) {
        int4 v0 = a[i];
        int4 v1 = a[i + stride];
        int4 v2 = a[i + 2 * stride];
        int4 v3 = a[i + 3 * stride];
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w;
        acc ^= v1.x ^ v1.y ^ v1.z ^ v1.w;
        acc ^= v2.x ^ v2.y ^ v2.z ^ v2.w;
        acc ^= v3.x ^ v3.y ^ v3.z ^ v3.w;
    }
    for (; i < n; i += stride) {
        int4 v = a[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == flag) sink[0] = acc; // keeps loads alive, practically never writes
}

static float elapsed(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// ---------------------------------------------------------------- VRAM
static double bench_vram(int sms, size_t bytes, int samples)
{
    size_t n = bytes / sizeof(int4);
    int4 *buf = nullptr;
    int *sink = nullptr;
    CK(cudaMalloc(&buf, n * sizeof(int4)));
    CK(cudaMalloc(&sink, sizeof(int)));

    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_read, BLOCK, 0));
    int grid = sms * (per_sm > 0 ? per_sm : 1);

    progress("  VRAM  preparing %zu MiB buffer...", bytes >> 20);
    k_init<<<grid, BLOCK>>>(buf, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    // one timed launch to size the batches (~50 ms each)
    CK(cudaEventRecord(e0));
    k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    double t1 = std::max(1e-3, (double)elapsed(e0, e1));
    int reps = std::max(1, (int)ceil(50.0 / t1));

    // warmup ~0.5 s: lets memory clock reach its max P-state
    int warm = std::max(1, (int)ceil(500.0 / t1));
    progress("  VRAM  warmup...");
    for (int r = 0; r < warm; r++) k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
    CK(cudaDeviceSynchronize());

    double best = 0.0;
    for (int s = 0; s < samples; s++) {
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; r++) k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        double gbs = (double)n * sizeof(int4) * reps / (elapsed(e0, e1) * 1e-3) / 1e9;
        best = std::max(best, gbs);
        progress("  VRAM  sample %d/%d  %7.1f GB/s   best %7.1f GB/s", s + 1, samples, gbs,
                 best);
    }
    progress_clear();
    CK(cudaGetLastError());

    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaFree(sink));
    CK(cudaFree(buf));
    return best;
}

// ---------------------------------------------------------------- PCIe
static double bench_pcie(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, int samples)
{
    const char *dir = kind == cudaMemcpyHostToDevice ? "H2D" : "D2H";
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    progress("  PCIe  %s warmup...", dir);
    for (int w = 0; w < 3; w++) CK(cudaMemcpy(dst, src, bytes, kind));

    double best = 0.0;
    for (int s = 0; s < samples; s++) {
        CK(cudaEventRecord(e0));
        CK(cudaMemcpyAsync(dst, src, bytes, kind, 0));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        double gbs = (double)bytes / (elapsed(e0, e1) * 1e-3) / 1e9;
        best = std::max(best, gbs);
        progress("  PCIe  %s sample %d/%d  %6.2f GB/s   best %6.2f GB/s", dir, s + 1, samples,
                 gbs, best);
    }
    progress_clear();
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return best;
}

// ---------------------------------------------------------------- driver
int gpu_run(const std::vector<int> &devs_in, size_t size_mib, int samples)
{
    int ndev = 0;
    progress("GPU   initializing CUDA...");
    cudaError_t err = cudaGetDeviceCount(&ndev);
    progress_clear();
    if (err != cudaSuccess || ndev == 0) {
        printf("CUDA  no usable CUDA GPU (%s)\n\n",
               err != cudaSuccess ? cudaGetErrorString(err) : "0 devices");
        return 0;
    }
    std::vector<int> devs = devs_in;
    if (devs.empty())
        for (int d = 0; d < ndev; d++) devs.push_back(d);

    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("CUDA driver %d.%d, runtime %d.%d, %d GPU(s)\n\n", drv / 1000, (drv % 1000) / 10,
           rt / 1000, (rt % 1000) / 10, ndev);

    for (int d : devs) {
        if (d < 0 || d >= ndev) {
            fprintf(stderr, "GPU %d does not exist\n", d);
            continue;
        }
        progress("GPU %d creating CUDA context...", d);
        CK(cudaSetDevice(d));
        CK(cudaFree(0)); // create context now (can take seconds: JIT, large VRAM)
        progress_clear();

        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d));
        int sms = 0, clk_khz = 0, bus_bits = 0, l2 = 0;
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d));
        CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, d));
        CK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, d));
        CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, d));
        char pci[32] = "?";
        cudaDeviceGetPCIBusId(pci, sizeof(pci), d);

        size_t freeb = 0, totalb = 0;
        CK(cudaMemGetInfo(&freeb, &totalb));
        size_t bytes = size_mib ? size_mib << 20 : std::min((size_t)1 << 30, freeb / 2);
        bytes &= ~(((size_t)1 << 20) - 1);

        // DDR: 2 transfers per reported clock (valid for GDDR5/5X/6/6X/7, HBM)
        double peak = 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;

        printf("GPU %d  %s  [%s]  sm_%d%d  %.1f GiB\n", d, p.name, pci, p.major, p.minor,
               totalb / 1073741824.0);

        double vram = bench_vram(sms, bytes, samples);
        if (peak > 0)
            printf("  VRAM  theoretical %7.1f GB/s   measured %7.1f GB/s  (%.0f%%)\n", peak, vram,
                   100.0 * vram / peak);
        else
            printf("  VRAM  measured %7.1f GB/s\n", vram);
        if (bytes < (size_t)l2 * 8)
            printf("  warning: buffer %zu MiB < 8x L2 cache, result inflated by the cache\n",
                   bytes >> 20);

        // pinned host memory; shrink on failure, last resort pageable
        progress("  PCIe  allocating host buffer...");
        size_t pbytes = std::min(bytes, (size_t)256 << 20);
        void *h = nullptr, *dbuf = nullptr;
        bool pinned = false;
        for (size_t sz = pbytes; sz >= ((size_t)16 << 20); sz /= 4) {
            if (cudaHostAlloc(&h, sz, cudaHostAllocDefault) == cudaSuccess) {
                pinned = true;
                pbytes = sz;
                break;
            }
            cudaGetLastError(); // clear non-sticky error
        }
        if (!pinned) {
            h = malloc(pbytes);
            if (!h) { fprintf(stderr, "malloc of %zu bytes failed\n", pbytes); exit(1); }
        }
        CK(cudaMalloc(&dbuf, pbytes));
        memset(h, 0x5a, pbytes);
        double h2d = bench_pcie(dbuf, h, pbytes, cudaMemcpyHostToDevice, samples);
        double d2h = bench_pcie(h, dbuf, pbytes, cudaMemcpyDeviceToHost, samples);
        CK(cudaFree(dbuf));
        if (pinned) CK(cudaFreeHost(h)); else free(h);
        printf("  PCIe  H2D %6.2f GB/s   D2H %6.2f GB/s%s\n\n", h2d, d2h,
               pinned ? "" : "   (pageable: pinned memory not available, values underestimated)");

        CK(cudaDeviceReset());
    }
    return ndev;
}

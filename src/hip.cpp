// vrambw - real VRAM and PCIe bandwidth of AMD GPUs (ROCm / HIP)
//
// VRAM: read-only streaming kernel over a buffer much larger than L2.
//       Same access pattern as LLM decode (weights streamed once per token).
// PCIe: hipMemcpy host<->device with pinned memory.
// Same code as gpu.cu (CUDA), with HIP names.
// Units: GB/s = 1e9 bytes/s (same as marketing).

#include "hip.h"
#include "progress.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x)                                                                 \
    do {                                                                      \
        hipError_t e_ = (x);                                                 \
        if (e_ != hipSuccess) {                                              \
            progress_clear();                                                 \
            fprintf(stderr, "HIP error: %s (%s:%d): %s\n", #x, __FILE__,    \
                    __LINE__, hipGetErrorString(e_));                        \
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

static float elapsed(hipEvent_t a, hipEvent_t b)
{
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    return ms;
}

// ---------------------------------------------------------------- VRAM
static double bench_vram(int sms, size_t bytes, int samples)
{
    size_t n = bytes / sizeof(int4);
    int4 *buf = nullptr;
    int *sink = nullptr;
    CK(hipMalloc(&buf, n * sizeof(int4)));
    CK(hipMalloc(&sink, sizeof(int)));

    int per_sm = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_read, BLOCK, 0));
    int grid = sms * (per_sm > 0 ? per_sm : 1);

    progress("  VRAM  preparing %zu MiB buffer...", bytes >> 20);
    k_init<<<grid, BLOCK>>>(buf, n);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));

    // one timed launch to size the batches (~50 ms each)
    CK(hipEventRecord(e0));
    k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    CK(hipGetLastError());
    double t1 = std::max(1e-3, (double)elapsed(e0, e1));
    int reps = std::max(1, (int)ceil(50.0 / t1));

    // warmup ~0.5 s: lets memory clock reach its max P-state
    int warm = std::max(1, (int)ceil(500.0 / t1));
    progress("  VRAM  warmup...");
    for (int r = 0; r < warm; r++) k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
    CK(hipDeviceSynchronize());

    double best = 0.0;
    for (int s = 0; s < samples; s++) {
        CK(hipEventRecord(e0));
        for (int r = 0; r < reps; r++) k_read<<<grid, BLOCK>>>(buf, n, sink, 0x7fffffff);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        double gbs = (double)n * sizeof(int4) * reps / (elapsed(e0, e1) * 1e-3) / 1e9;
        best = std::max(best, gbs);
        progress("  VRAM  sample %d/%d  %7.1f GB/s   best %7.1f GB/s", s + 1, samples, gbs,
                 best);
    }
    progress_clear();
    CK(hipGetLastError());

    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(sink));
    CK(hipFree(buf));
    return best;
}

// ---------------------------------------------------------------- PCIe
static double bench_pcie(void *dst, const void *src, size_t bytes, hipMemcpyKind kind, int samples)
{
    const char *dir = kind == hipMemcpyHostToDevice ? "H2D" : "D2H";
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    progress("  PCIe  %s warmup...", dir);
    for (int w = 0; w < 3; w++) CK(hipMemcpy(dst, src, bytes, kind));

    double best = 0.0;
    for (int s = 0; s < samples; s++) {
        CK(hipEventRecord(e0));
        CK(hipMemcpyAsync(dst, src, bytes, kind, 0));
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        double gbs = (double)bytes / (elapsed(e0, e1) * 1e-3) / 1e9;
        best = std::max(best, gbs);
        progress("  PCIe  %s sample %d/%d  %6.2f GB/s   best %6.2f GB/s", dir, s + 1, samples,
                 gbs, best);
    }
    progress_clear();
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    return best;
}

// ---------------------------------------------------------------- driver
void hip_run(int samples, double ram_peak)
{
    int ndev = 0;
    progress("ROCm  initializing HIP...");
    hipError_t err = hipGetDeviceCount(&ndev);
    progress_clear();
    if (err != hipSuccess || ndev == 0) {
        printf("ROCm  no usable ROCm GPU (%s)\n\n", err != hipSuccess ? hipGetErrorString(err) : "0 devices");
        return;
    }
    int rt = 0;
    (void)hipRuntimeGetVersion(&rt); // major * 10000000 + minor * 100000 + patch
    printf("ROCm  HIP runtime %d.%d, %d GPU(s)\n\n", rt / 10000000, (rt / 100000) % 100, ndev);

    for (int d = 0; d < ndev; d++) {
        progress("ROCm %d creating HIP context...", d);
        CK(hipSetDevice(d));
        CK(hipFree(0));
        progress_clear();

        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, d));
        int cus = 0, clk_khz = 0, bus_bits = 0, l2 = 0, integrated = 0;
        CK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, d));
        CK(hipDeviceGetAttribute(&clk_khz, hipDeviceAttributeMemoryClockRate, d));
        CK(hipDeviceGetAttribute(&bus_bits, hipDeviceAttributeMemoryBusWidth, d));
        CK(hipDeviceGetAttribute(&l2, hipDeviceAttributeL2CacheSize, d));
        CK(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, d));
        char pci[32] = "?";
        (void)hipDeviceGetPCIBusId(pci, sizeof(pci), d);
        char arch[64];
        snprintf(arch, sizeof(arch), "%s", p.gcnArchName);
        arch[strcspn(arch, ":")] = 0; // "gfx90a:sramecc+:xnack-" -> "gfx90a"

        size_t freeb = 0, totalb = 0;
        CK(hipMemGetInfo(&freeb, &totalb));
        size_t bytes = std::min((size_t)1 << 30, freeb / 2);
        bytes &= ~(((size_t)1 << 20) - 1);

        printf("ROCm %d  %s  [%s]  %s  %s  %.1f GiB\n", d, p.name, pci, arch,
               integrated ? "integrated" : "discrete", totalb / 1073741824.0);
        fflush(stdout);

        double vram = bench_vram(cus, bytes, samples);

        // theoretical peak
        double peak = 0;
        if (integrated) {
            peak = ram_peak; // an iGPU reads system RAM
        } else {
            // HBM reports the real clock (2 transfers/clock); GDDR6 reports a clock
            // 8x lower than the data rate (16 transfers/clock). Pick the one that fits.
            double p2 = 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
            if (p2 > 0 && vram <= p2 * 1.05) peak = p2;
            else if (p2 > 0 && vram <= p2 * 8 * 1.05 && vram >= p2 * 8 * 0.3) peak = p2 * 8;
        }
        const char *what = integrated ? "  (iGPU: system RAM)" : "";
        if (peak > 0)
            printf("  VRAM  theoretical %7.1f GB/s   measured %7.1f GB/s  (%.0f%%)%s\n", peak, vram,
                   100.0 * vram / peak, what);
        else
            printf("  VRAM  measured %7.1f GB/s%s\n", vram, what);
        if (bytes < (size_t)l2 * 8)
            printf("  warning: buffer %zu MiB < 8x L2 cache, result inflated by the cache\n", bytes >> 20);

        // pinned host memory; shrink on failure, last resort pageable
        progress("  PCIe  allocating host buffer...");
        size_t pbytes = std::min(bytes, (size_t)256 << 20);
        void *h = nullptr, *dbuf = nullptr;
        bool pinned = false;
        for (size_t sz = pbytes; sz >= ((size_t)16 << 20); sz /= 4) {
            if (hipHostMalloc(&h, sz, hipHostMallocDefault) == hipSuccess) {
                pinned = true;
                pbytes = sz;
                break;
            }
            (void)hipGetLastError(); // clear non-sticky error
        }
        if (!pinned) {
            h = malloc(pbytes);
            if (!h) { fprintf(stderr, "malloc of %zu bytes failed\n", pbytes); exit(1); }
        }
        CK(hipMalloc(&dbuf, pbytes));
        memset(h, 0x5a, pbytes);
        double h2d = bench_pcie(dbuf, h, pbytes, hipMemcpyHostToDevice, samples);
        double d2h = bench_pcie(h, dbuf, pbytes, hipMemcpyDeviceToHost, samples);
        CK(hipFree(dbuf));
        if (pinned) CK(hipHostFree(h)); else free(h);
        printf("  %s  H2D %6.2f GB/s   D2H %6.2f GB/s%s%s\n\n", integrated ? "copy" : "PCIe", h2d, d2h,
               integrated ? "   (iGPU: copies inside system RAM, no PCIe)" : "",
               pinned ? "" : "   (pageable: pinned memory not available, values underestimated)");

        CK(hipDeviceReset());
    }
}

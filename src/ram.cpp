// vrambw - system RAM read bandwidth (pure C++, no CUDA)
#include "ram.h"
#include "progress.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdint>
#include <unistd.h>

// ---------------------------------------------------------------- system RAM
// Same method as VRAM: read-only streaming, buffer >> L3, all hardware
// threads, best sample. Each thread first-touches its own slice (NUMA-local).
static volatile uint64_t g_sink;

static void ram_init(uint64_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = i * 0x9E3779B97F4A7C15ull;
}

static void ram_read(const uint64_t *p, size_t n, int reps, uint64_t *out)
{
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            a0 ^= p[i];     a1 ^= p[i + 1]; a2 ^= p[i + 2]; a3 ^= p[i + 3];
            a4 ^= p[i + 4]; a5 ^= p[i + 5]; a6 ^= p[i + 6]; a7 ^= p[i + 7];
        }
        a0 += (uint64_t)r; // keep passes distinct
    }
    *out = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}

double bench_ram(int samples, int *nthreads_out, size_t *bytes_out)
{
    int nt = (int)std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;

    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    size_t bytes = (size_t)1 << 30;
    if (l3 > 0) bytes = std::max(bytes, (size_t)l3 * 8);
    if (pages > 0 && psz > 0) bytes = std::min(bytes, (size_t)pages * psz / 4);
    size_t per = (bytes / nt / sizeof(uint64_t)) & ~(size_t)7; // words per thread
    bytes = per * nt * sizeof(uint64_t);

    uint64_t *buf = (uint64_t *)malloc(bytes);
    if (!buf) { fprintf(stderr, "malloc of %zu bytes for RAM test failed\n", bytes); exit(1); }
    std::vector<uint64_t> res(nt);
    std::vector<std::thread> th;

    progress("RAM   preparing %zu MiB buffer...", bytes >> 20);
    for (int t = 0; t < nt; t++) th.emplace_back(ram_init, buf + t * per, per);
    for (auto &x : th) x.join();

    // k threads read the whole buffer: thread t reads slices t, t+k, t+2k...
    auto run = [&](int k, int reps) {
        th.clear();
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < k; t++)
            th.emplace_back([&, t]() {
                uint64_t acc = 0, r;
                for (int rep = 0; rep < reps; ++rep) {
                    for (int s = t; s < nt; s += k) {
                        ram_read(buf + s * per, per, 1, &r);
                        acc = (acc * 0x9E3779B97F4A7C15ull) ^ r;
                    }
                }
                res[t] = acc;
            });
        for (auto &x : th) x.join();
        auto t1 = std::chrono::steady_clock::now();
        for (int t = 0; t < k; t++) g_sink ^= res[t];
        return std::chrono::duration<double>(t1 - t0).count();
    };

    // more threads is not always faster (SMT, fabric contention): try nt, nt/2, ... 1
    double best = 0.0;
    int best_k = nt;
    for (int k = nt; k >= 1; k /= 2) {
        progress("RAM   %2d threads  warmup...       best %6.1f GB/s", k, best);
        double t1 = std::max(1e-4, run(k, 1));          // also warmup
        int reps = std::max(1, (int)ceil(0.2 / t1));    // ~200 ms per sample
        for (int s = 0; s < samples; s++) {
            double gbs = (double)bytes * reps / run(k, reps) / 1e9;
            if (gbs > best) { best = gbs; best_k = k; }
            progress("RAM   %2d threads  sample %d/%d  %6.1f GB/s   best %6.1f GB/s", k,
                     s + 1, samples, gbs, best);
        }
    }

    progress_clear();
    free(buf);
    *nthreads_out = best_k;
    *bytes_out = bytes;
    return best;
}


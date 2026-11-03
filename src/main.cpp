// vrambw - real memory bandwidth: system RAM, and (if built with CUDA) VRAM + PCIe.
#include "ram.h"
#ifdef HAVE_CUDA
#include "gpu.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static void usage(const char *p)
{
    printf("usage: %s [-d LIST] [-s MiB] [-n N]\n"
           "  -d LIST   GPUs to test, e.g. 0 or 0,2 (default: all)\n"
           "  -s MiB    VRAM buffer size (default: min(1024, 50%% of free VRAM))\n"
           "  -n N      samples per measurement, best is reported (default 20)\n",
           p);
}

int main(int argc, char **argv)
{
    std::vector<int> devs;
    size_t size_mib = 0;
    int samples = 20;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            char *s = argv[++i];
            for (char *t = strtok(s, ","); t; t = strtok(nullptr, ",")) devs.push_back(atoi(t));
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            size_mib = strtoull(argv[++i], nullptr, 10);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            samples = std::max(1, atoi(argv[++i]));
        } else {
            usage(argv[0]);
            return strcmp(argv[i], "-h") && strcmp(argv[i], "--help") ? 1 : 0;
        }
    }

    int nt = 0;
    size_t rb = 0;
    double ram = bench_ram(std::min(samples, 5), &nt, &rb);
    printf("RAM   measured %7.1f GB/s   (%d threads, %zu MiB)\n\n", ram, nt, rb >> 20);
    fflush(stdout);

#ifdef HAVE_CUDA
    gpu_run(devs, size_mib, samples);
#else
    (void)size_mib;
    printf("GPU   not tested: built without CUDA\n");
#endif
    return 0;
}

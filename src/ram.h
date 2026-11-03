#pragma once
#include <cstddef>

// Best read bandwidth of system RAM in GB/s (1e9). Fills thread count and buffer size used.
double bench_ram(int samples, int *nthreads_out, size_t *bytes_out);

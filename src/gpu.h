#pragma once
#include <cstddef>
#include <vector>

// Tests VRAM + PCIe on the given devices (empty = all). Returns number of CUDA devices.
int gpu_run(const std::vector<int> &devs, size_t size_mib, int samples);

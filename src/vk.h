#pragma once

// Tests every Vulkan GPU (software devices skipped): read bandwidth from
// device-local memory and from host RAM. libvulkan is loaded at run time.
// ram_peak_gbs: theoretical system RAM peak (0 if unknown), used for iGPUs.
void vk_run(int samples, double ram_peak_gbs);

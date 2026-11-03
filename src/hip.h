#pragma once

// Tests every ROCm (HIP) GPU: VRAM read bandwidth and PCIe H2D/D2H.
// ram_peak_gbs: theoretical system RAM peak (0 if unknown), used for iGPUs.
void hip_run(int samples, double ram_peak_gbs);

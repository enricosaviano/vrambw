# vrambw

`vrambw` measures the **real memory bandwidth** of a computer:

| What | Why it matters |
|---|---|
| **System RAM** read bandwidth, type and channels | Speed of LLM decode when layers run on the CPU (offload) |
| **GPU VRAM** read bandwidth (NVIDIA, CUDA) | Speed of LLM decode on the GPU; comparable with the vendor's spec sheet |
| **PCIe** bandwidth host → GPU and GPU → host | Speed of model loading and of CPU↔GPU traffic |
| **Vulkan** read bandwidth of any GPU (AMD, Intel, NVIDIA, also integrated) | Speed of LLM decode with Vulkan backends (llama.cpp Vulkan), e.g. on iGPUs |

It is plain C++, CUDA and Vulkan, like llama.cpp: no Python, no libraries, no runtime dependencies.
It builds with CUDA 11.4, 12.6 and 12.8, and also **without CUDA**. Vulkan is optional too: tests that cannot be built or run are skipped.

---

## Quick start

```sh
./configure
make
./vrambw
```

That's all. `./configure` finds the compiler and CUDA by itself.

---

## Requirements

- Linux, `sh`, `make`
- A C++ compiler: `g++`, `c++` or `clang++`
- Optional: the CUDA toolkit (`nvcc`) and the NVIDIA driver, to test NVIDIA GPUs with CUDA
- Optional: the Vulkan headers (`libvulkan-dev`) to build, and a Vulkan driver to run, the Vulkan test

---

## Building

### What `./configure` does

1. Finds a working C++ compiler.
2. Looks for `nvcc` in this order: `--cuda` / `--nvcc` options, `$CUDA_HOME`, `$CUDA_PATH`, `PATH`, `/usr/local/cuda`, `/opt/cuda`, `/usr/local/cuda-*` (newest first).
3. If `nvcc` is **not found**, it builds without the CUDA tests. No error.
   It also looks for the Vulkan headers; without them the Vulkan test is not built. No error either.
4. If `nvcc` is found, it asks `nvcc` which GPU architectures it supports and builds code for all of them, plus PTX for the newest one, so future GPUs still work.
   - CUDA 11.4: up to sm_86 (Ampere), down to sm_35 (Kepler)
   - CUDA 12.6: sm_50 (Maxwell) to sm_90 (Hopper)
   - CUDA 12.8: sm_50 to sm_120 (Blackwell)
5. Compiles a tiny test program, to catch a bad compiler setup early.
6. Writes `config.mk`. It is a short, readable file: you can edit it by hand.

### Options

| Option | Meaning |
|---|---|
| `--cuda=DIR` | Use the CUDA toolkit in `DIR` (e.g. `/usr/local/cuda-11.4`). CUDA becomes mandatory. |
| `--nvcc=PATH` | Use this exact `nvcc`. CUDA becomes mandatory. |
| `--without-cuda` | Do not build the CUDA tests, even if CUDA is installed. |
| `--without-vulkan` | Do not build the Vulkan test, even if the headers are installed. |
| `--arch=LIST` | Build only for these GPUs, e.g. `--arch=61` (Pascal) or `--arch=61,75,86`. Much faster to compile. |
| `--host-cxx=CXX` | C++ compiler to use, also passed to `nvcc` as host compiler (e.g. `g++-10`). |
| `--allow-unsupported-compiler` | Let `nvcc` accept a newer `g++` than it officially supports. |

Examples:

```sh
./configure --cuda=/usr/local/cuda-11.4 --host-cxx=g++-10   # old CUDA needs an older g++
./configure --arch=61                                       # only Pascal cards (P104/P106, GTX 10xx)
./configure --without-cuda                                  # machine without NVIDIA GPU
```

### Make targets

| Command | Meaning |
|---|---|
| `make` | Build `./vrambw` |
| `make run` | Build and run |
| `make clean` | Delete binary and object files |
| `make shader` | Recompile the Vulkan shader (only after editing `src/vk_read.comp`; needs `glslc`) |
| `make distclean` | Also delete `config.mk` |

---

## Running

```sh
./vrambw                 # RAM + all GPUs
./vrambw -d 0            # RAM + GPU 0 only
./vrambw -d 0,2          # RAM + GPUs 0 and 2
./vrambw -s 2048 -n 30   # 2 GiB VRAM buffer, 30 samples per measurement
./vrambw > result.txt    # save the results to a file
```

| Option | Meaning | Default |
|---|---|---|
| `-d LIST` | CUDA GPUs to test, comma separated (Vulkan always tests all GPUs) | all |
| `-s MiB` | VRAM test buffer size | 1024 MiB, or half of the free VRAM if smaller |
| `-n N` | Samples per measurement; the best one is reported | 20 (5 for RAM) |
| `-h` | Help | |

A full run takes about 5–10 seconds for the RAM and a few seconds per GPU.

### Live progress

While it runs, the bottom line of the terminal shows what is happening, for example:

```
RAM    4 threads  sample 3/5    42.1 GB/s   best   43.6 GB/s
```

The line shows the current phase, the last sample and the best value so far. When a measurement is finished the progress line disappears and only the result stays.
The progress line goes to stderr and only when it is a terminal, so `./vrambw > file` gives a clean file.

---

## Reading the output

Example (RTX 2080 Ti; the PCIe and Vulkan lines are illustrative):

```
RAM   LPDDR5/5X-6400 (guessed), 4 x 32-bit = 128-bit (dual-channel equivalent), 16 GiB
RAM   theoretical   102.4 GB/s   measured    43.6 GB/s  (43%)   (4 threads, 1024 MiB)

CUDA driver 13.0, runtime 12.8, 1 GPU(s)

GPU 0  NVIDIA GeForce RTX 2080 Ti  [0000:02:00.0]  sm_75  10.6 GiB
  VRAM  theoretical   616.0 GB/s   measured   586.5 GB/s  (95%)
  PCIe  H2D  12.20 GB/s   D2H  12.90 GB/s

Vulkan  1 GPU(s) (software devices skipped)

VK 0  NVIDIA GeForce RTX 2080 Ti  discrete GPU  Vulkan 1.4
  device-local  measured   580.0 GB/s   (1024 MiB buffer)
  host RAM      measured    12.5 GB/s   (1024 MiB buffer)
```

| Line | Meaning |
|---|---|
| `RAM LPDDR5/5X-6400 ...` | Detected memory type, speed (MT/s), channels, total bus width and size (see *System RAM type and channels*) |
| `RAM theoretical` | Theoretical RAM peak computed from the detected configuration (missing if it could not be detected) |
| `RAM measured` | Measured system RAM read bandwidth, percentage of the theoretical peak, and how many CPU threads gave the best result |
| `CUDA driver / runtime` | CUDA version supported by the installed driver, and CUDA version the program was built with |
| `GPU 0 ... [0000:02:00.0]` | CUDA device number, name, **PCI bus address** (tells which physical card it is when you have several identical ones), compute capability, total VRAM |
| `VRAM theoretical` | **Theoretical** peak = the number in marketing material |
| `VRAM measured` | **Measured** read bandwidth, and percentage of the theoretical peak |
| `PCIe H2D` | Copy speed from host RAM to GPU (Host to Device) |
| `PCIe D2H` | Copy speed from GPU to host RAM (Device to Host) |
| `VK 0 ...` | Vulkan device number, name, type (integrated / discrete), Vulkan version |
| `device-local` | GPU reading its own memory (VRAM, or the RAM reserved for an iGPU) |
| `host RAM` | GPU reading system RAM directly (over PCIe for a discrete GPU) |

**Units:** GB/s means 10⁹ bytes per second (not 2³⁰). This is the unit used by NVIDIA, AMD, Intel and memory vendors, so the numbers are directly comparable.

---

## How the measurements work

All tests follow the same simple idea: **read a big block of memory many times, as fast as possible, and divide bytes by time**. Every test is repeated several times and the **best** result is reported, because that is the real capability of the hardware (worse samples are caused by other programs, clock changes, etc.).

### VRAM (GPU memory)

- A 1 GiB buffer is allocated on the GPU and filled with data.
- The buffer is much bigger than the GPU's L2 cache (2–96 MB depending on the GPU), so the data really comes from VRAM and not from the cache. If the buffer is too small (`-s`), the program warns you.
- A CUDA kernel reads the whole buffer using every SM (streaming multiprocessor) of the GPU, with wide 16-byte loads and several loads in flight per thread. This is how a well-optimized program saturates the memory bus.
- The data is only **read**, never written. This is exactly what happens during LLM **decode**: for every generated token all model weights are read once from VRAM.
- Before measuring, the GPU runs the kernel for about half a second (warmup), so the memory clock has time to go up to full speed.
- Time is measured on the GPU itself with CUDA events.

**Theoretical peak** is computed from the values the driver reports:

```
peak GB/s = memory clock × 2 × bus width (bits) / 8
```

For example GTX 1080: 5005 MHz × 2 × 256 / 8 = 320 GB/s. The formula is correct for GDDR5, GDDR5X, GDDR6, GDDR6X, GDDR7 and HBM.

### PCIe

- A 256 MiB buffer is allocated in host RAM as **pinned** memory (locked in RAM, so the GPU can copy it directly with DMA, without extra copies by the CPU). This is the fastest way and what CUDA programs use for big transfers.
- The buffer is copied host → GPU (H2D) and GPU → host (D2H) several times; the best speed is reported.
- If the system refuses pinned memory, the program tries 64 MiB and 16 MiB. If that also fails it uses normal (pageable) memory and prints `(pageable ...)`: those numbers are lower than what the hardware can do. See *Troubleshooting*.

Rough reference values (real numbers are a bit below):

| Link | Max per direction |
|---|---|
| PCIe 1.0 x4 | 1 GB/s |
| PCIe 2.0 x4 | 2 GB/s |
| PCIe 3.0 x16 | 15.8 GB/s |
| PCIe 4.0 x16 | 31.5 GB/s |
| PCIe 5.0 x16 | 63 GB/s |

### Vulkan (any GPU: AMD, Intel, NVIDIA, integrated or discrete)

If the system has a Vulkan driver, every Vulkan GPU is tested too. Software devices such as `llvmpipe`, which run on the CPU, are skipped. This works on GPUs that CUDA does not support, for example the AMD and Intel integrated GPUs used by llama.cpp's Vulkan backend.

- The method is the same as the CUDA VRAM test: a compute shader reads a buffer of up to 1 GiB, 16 bytes per load, with all threads of the GPU. Data is only read, there is a warmup, and the best sample is reported.
- Two kinds of memory are measured:
  - **device-local**: memory "close" to the GPU. On a discrete GPU this is VRAM. On an integrated GPU it is the part of system RAM reserved for the GPU (the "UMA frame buffer" set in the BIOS).
  - **host RAM**: normal system memory read directly by the GPU. On a discrete GPU this goes over PCIe. On an integrated GPU it is the same RAM, reached without the reserved area.
- For integrated GPUs the line also shows the percentage of the **RAM theoretical peak**, because their memory *is* the system RAM.
- If a driver exposes only one kind of memory (common on Intel iGPUs), the line says `same memory as device-local (unified memory)`.
- `libvulkan.so.1` is loaded when the program starts (with `dlopen`), not linked: the same binary also runs on machines without Vulkan, and then prints `Vulkan  not available`.

The shader source is `src/vk_read.comp`. Its compiled SPIR-V is stored in `src/vk_read.spv.inc`, so building does not need a shader compiler. After editing the shader, run `make shader` (needs `glslc`).

### System RAM

- A buffer of at least 1 GiB (and at least 8× the CPU L3 cache, at most 1/4 of the installed RAM) is allocated and filled.
- Each thread fills its own part of the buffer, so on multi-socket (NUMA) machines memory sits next to the CPU that uses it.
- The buffer is read by all CPU threads, then by half of them, a quarter, … down to 1 thread. The best result and its thread count are reported. Using all threads is not always fastest: Hyper-Threading/SMT siblings and the CPU's internal interconnect can make fewer threads faster.
- Like the VRAM test it only **reads**, so it matches the pattern of CPU decode.

### System RAM type and channels (heuristic)

Before the RAM test, `vrambw` tries to find out **what memory is installed**: type (DDR3, DDR3L, DDR4, DDR5, LPDDR4, LPDDR4X, LPDDR5, LPDDR5X, …), speed, number of channels and total size. From this it computes the **theoretical RAM peak**, so the RAM line can be compared with the spec sheet exactly like the VRAM line.

Where the information comes from:

1. The **SMBIOS table** of the BIOS/UEFI (`/sys/firmware/dmi/tables/DMI`). It describes every memory module. Only root can read it, so this is used with `sudo ./vrambw`.
2. Without root, the **udev database** (`/run/udev/data/+dmi:id`), where systemd 248+ (Ubuntu 22.04 and newer) keeps a readable copy of the same data.
3. If neither is available, the line says `type/channels unknown` and only the measured value is shown.

Why "heuristic":

- **Type.** The BIOS reports a type code, but SMBIOS has no codes for LPDDR4X, LPDDR5X or DDR3L, and older tools do not know newer codes (LPDDR5 often shows as `<OUT OF SPEC>`). In these cases the type is guessed from voltage and speed, and marked `(guessed)`:

  | Voltage | Type |
  |---|---|
  | 0.5 V | LPDDR5 / LPDDR5X (above 6400 MT/s it must be LPDDR5X; at exactly 6400 it can be both, shown as `LPDDR5/5X`) |
  | 0.6 V | LPDDR4X |
  | 1.1 V | DDR5 (or LPDDR4 if soldered and ≤ 4266 MT/s) |
  | 1.2 V | DDR4 |
  | 1.35 V | DDR3L |
  | 1.5 V | DDR3 |

  Without voltage, the speed decides (e.g. ≥ 4400 MT/s → DDR5, 2133–3200 → DDR4).
- **Speed.** The configured speed is used (the speed the memory really runs at). Old BIOSes (SMBIOS 2.7 and earlier) write the memory **clock** in MHz there, which is half the MT/s: DDR3-1600 appears as 800. When the configured value is about half of the rated speed, or below the minimum speed of that memory type, it is doubled.
- **Channels.** Modules are grouped by socket, memory controller and channel, read from the slot labels written by the board vendor (`P0 CHANNEL A`, `ChannelA-DIMM0`, `Controller1-ChannelA`, `DIMM_B1`, `CPU1_DIMM_C1`, …). Two modules on the same channel (e.g. 4 DIMMs on a dual-channel desktop) count as **one** channel. Some boards also write ECC bits into the data width (72 instead of 64): these are removed, and the line says `ECC`. If no label names a channel, the program guesses and says so: boards with 1 socket and up to 4 slots are assumed **dual-channel** (`channels guessed: 4-slot board`), because desktop and laptop CPUs have 2 channels; bigger boards are assumed to have 1 module per channel (`channels guessed: 1 module per channel`), as is usual on servers.
- **Channel names.** Desktop and server channels are 64 bits wide. LPDDR memory uses narrower channels (16 or 32 bits), so the total bus is also expressed in 64-bit units, with the word `equivalent`: 4 × 32-bit LPDDR5 = 128-bit = **dual-channel equivalent**, the same bandwidth class as two DDR5 DIMMs.

Examples of the description line:

```
RAM   DDR4-3200, 2 x 64-bit = 128-bit (dual-channel), 64 GiB
RAM   DDR5-6000, 2 x 64-bit = 128-bit (dual-channel), 32 GiB
RAM   LPDDR5/5X-6400 (guessed), 4 x 32-bit = 128-bit (dual-channel equivalent), 16 GiB
RAM   LPDDR4X-4267, 8 x 16-bit = 128-bit (dual-channel equivalent), 16 GiB
RAM   DDR3L-1866, 4 x 64-bit = 256-bit (quad-channel), 32 GiB
RAM   DDR4-2933, 6 x 64-bit = 384-bit (hexa-channel), 96 GiB
RAM   DDR4-3200, 16 x 64-bit = 1024-bit (2 sockets x octa-channel), 512 GiB
```

The theoretical peak is:

```
peak GB/s = MT/s × total bus width in bytes
```

| Memory | Typical total bus | Example |
|---|---|---|
| Desktop DDR4/DDR5, 2 DIMMs (dual channel) | 128 bit = 16 bytes | DDR5-6000: 6000 × 16 = 96 GB/s |
| Laptop LPDDR5/5X (soldered) | often 128 bit = 16 bytes | LPDDR5-6400: 102.4 GB/s |
| Server, 8 channels DDR5 | 512 bit = 64 bytes | DDR5-4800: 307 GB/s |

---

## Interpreting the results

### Estimating LLM decode speed

Decode (generating tokens one by one) is limited by memory bandwidth: for every token the whole model is read once. So:

```
max tokens/s ≈ measured bandwidth (GB/s) / model size (GB)
```

Example: 586 GB/s and a 7B model in Q4_K_M (≈ 4.4 GB) → at most ≈ 130 tokens/s.
Real programs reach less than this upper bound (KV cache reads, compute, kernel overheads), but the ratio between two cards is a good predictor of which one decodes faster.

For a model split between GPU and CPU, the CPU part runs at the speed of the **RAM** number, which is usually 10–20× slower.

### What percentage of the theoretical peak is normal?

- **About 80–95%** on a healthy GPU: normal. No real program reaches 100%.
- **Much lower:** typical causes:
  - **P2 power state.** On GeForce and mining cards the driver often keeps the memory clock below the maximum while running CUDA programs. Inference programs see the same lower speed, so the measured number is the one that matters for them.
  - Another program is using the GPU at the same time.
  - The card has modified clocks, or a mining BIOS.
- **CPU RAM far below the theoretical value.** Common on laptop chips and APUs: the CPU cores cannot use the full memory bandwidth because the internal connection between CPU and memory controller is slower (e.g. AMD Ryzen 6000 mobile: ≈ 42 GB/s measured with 102 GB/s LPDDR5). The measured value is what the CPU can really use.

### Integrated GPU vs CPU

On laptops and APUs the integrated GPU often reads the system RAM **faster than the CPU cores can**, because it has a wider path to the memory controller. Example, AMD Ryzen 7 6800HS with LPDDR5-6400 (102.4 GB/s theoretical):

```
RAM   theoretical   102.4 GB/s   measured    43.4 GB/s  (42%)     <- CPU cores
  device-local  measured    84.0 GB/s   (..., 82% of RAM theoretical)   <- Radeon 680M iGPU
```

So on this machine the upper bound for decode with a Vulkan backend on the iGPU is about twice the one on the CPU, with the same RAM.

### Several GPUs

All GPUs are tested one after another, never at the same time, so they do not disturb each other.
GPU numbers follow CUDA's order. To know which physical card is which, use the PCI address in brackets; `nvidia-smi` shows the same address. Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA numbers match the order of `nvidia-smi`.

---

## Troubleshooting

**`RAM   type/channels unknown: SMBIOS not readable (run as root for details)`**
The memory description needs SMBIOS data. Run `sudo ./vrambw`. Without root it only works with systemd 248 or newer.

**Wrong RAM type or channel count**
The detection relies on what the BIOS writes in SMBIOS, which is sometimes incomplete or wrong. Check with `sudo dmidecode -t memory` (fields `Type`, `Configured Memory Speed`, `Data Width`, `Bank Locator`, `Locator`). The measured bandwidth is always correct, because it does not depend on this.
For debugging, `vrambw` can read the memory data from a file instead of the system:

```sh
sudo cat /sys/firmware/dmi/tables/DMI > dmi.bin      # raw SMBIOS table
VRAMBW_DMI=dmi.bin ./vrambw

cp /run/udev/data/+dmi:id udev.txt                 # udev database
VRAMBW_UDEV_DB=udev.txt ./vrambw
```

These two files are all that is needed to reproduce a wrong detection on another machine.

**`vulkan: headers not found -> building without Vulkan`**
Install the Vulkan headers (`sudo apt install libvulkan-dev` on Debian/Ubuntu) and run `./configure` again. If you use the LunarG SDK, `$VULKAN_SDK/include` is searched too.

**`Vulkan  not available (...)`**
The Vulkan loader (`libvulkan1`) or a Vulkan driver is not installed: e.g. `mesa-vulkan-drivers` for AMD/Intel, or the NVIDIA driver. Check with `vulkaninfo --summary`.

**`nvcc: not found -> building without CUDA`**
CUDA is not installed or not where expected. Use `./configure --cuda=/path/to/cuda`.

**Test compile fails / `unsupported GNU version`**
The CUDA version does not accept your `g++`. Install an older one and use `--host-cxx=g++-10` (or another supported version), or try `--allow-unsupported-compiler`.

**`CUDA  no usable CUDA GPU (...)`**
The program was built with CUDA but there is no NVIDIA GPU or driver, or the driver is too old for the CUDA version it was built with. The RAM result is still valid. Check `nvidia-smi`.

**PCIe line shows `(pageable ...)`, and `dmesg` contains `NVRM: Failed to create a DMA mapping!`**
The system cannot give the GPU direct access to host RAM. Most common reason: the IOMMU (Intel VT-d / AMD-Vi) is on in "translated" mode, which recent kernels enable by default. Check with:

```sh
sudo dmesg | grep "Default domain type"
```

If it says `Translated`, add `iommu=pt` to the kernel command line (in `/etc/default/grub`, `GRUB_CMDLINE_LINUX_DEFAULT`), run `sudo update-grub` and reboot. This affects every CUDA program, not only `vrambw`.

**`warning: buffer ... < 8x L2 cache, result inflated by the cache`**
The `-s` buffer is too small, so the GPU cache is measured instead of VRAM and the number is too high. Use a bigger buffer or the default.

---

## Project layout

```
configure        detects compiler and CUDA, writes config.mk
Makefile         build rules (reads config.mk)
src/main.cpp     command line and output
src/ram.cpp      system RAM test (plain C++, always built)
src/meminfo.cpp  RAM type / channel detection from SMBIOS or udev
src/gpu.cu       VRAM and PCIe tests (CUDA, built only when nvcc is found)
src/vk.cpp       Vulkan test (built when the Vulkan headers are found)
src/vk_read.comp Vulkan compute shader (GLSL source)
src/vk_read.spv.inc  compiled shader, embedded in the binary (make shader)
src/progress.h   live progress line
```

---

## License

MIT. See [LICENSE](LICENSE).

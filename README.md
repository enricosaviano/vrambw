# vrambw

`vrambw` measures the **real memory bandwidth** of a computer:

| What | Why it matters |
|---|---|
| **System RAM** read bandwidth | Speed of LLM decode when layers run on the CPU (offload) |
| **GPU VRAM** read bandwidth (NVIDIA, CUDA) | Speed of LLM decode on the GPU; comparable with the vendor's spec sheet |
| **PCIe** bandwidth host → GPU and GPU → host | Speed of model loading and of CPU↔GPU traffic |

It is plain C++ and CUDA, like llama.cpp: no Python, no libraries, no runtime dependencies.
It builds with CUDA 11.4, 12.6 and 12.8, and also **without CUDA** (then only the RAM test is built).

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
- Optional: the CUDA toolkit (`nvcc`) and the NVIDIA driver, to test GPUs

---

## Building

### What `./configure` does

1. Finds a working C++ compiler.
2. Looks for `nvcc` in this order: `--cuda` / `--nvcc` options, `$CUDA_HOME`, `$CUDA_PATH`, `PATH`, `/usr/local/cuda`, `/opt/cuda`, `/usr/local/cuda-*` (newest first).
3. If `nvcc` is **not found**, it prepares a **RAM-only build**. No error.
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
| `--without-cuda` | Build only the RAM test, even if CUDA is installed. |
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
| `-d LIST` | GPUs to test, comma separated | all |
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

Example (RTX 2080 Ti; the PCIe line is illustrative):

```
RAM   measured    43.6 GB/s   (4 threads, 1024 MiB)

CUDA driver 13.0, runtime 12.8, 1 GPU(s)

GPU 0  NVIDIA GeForce RTX 2080 Ti  [0000:02:00.0]  sm_75  10.6 GiB
  VRAM  theoretical   616.0 GB/s   measured   586.5 GB/s  (95%)
  PCIe  H2D  12.20 GB/s   D2H  12.90 GB/s
```

| Line | Meaning |
|---|---|
| `RAM measured` | Measured system RAM read bandwidth, and how many CPU threads gave the best result |
| `CUDA driver / runtime` | CUDA version supported by the installed driver, and CUDA version the program was built with |
| `GPU 0 ... [0000:02:00.0]` | CUDA device number, name, **PCI bus address** (tells which physical card it is when you have several identical ones), compute capability, total VRAM |
| `VRAM theoretical` | **Theoretical** peak = the number in marketing material |
| `VRAM measured` | **Measured** read bandwidth, and percentage of the theoretical peak |
| `PCIe H2D` | Copy speed from host RAM to GPU (Host to Device) |
| `PCIe D2H` | Copy speed from GPU to host RAM (Device to Host) |

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

### System RAM

- A buffer of at least 1 GiB (and at least 8× the CPU L3 cache, at most 1/4 of the installed RAM) is allocated and filled.
- Each thread fills its own part of the buffer, so on multi-socket (NUMA) machines memory sits next to the CPU that uses it.
- The buffer is read by all CPU threads, then by half of them, a quarter, … down to 1 thread. The best result and its thread count are reported. Using all threads is not always fastest: Hyper-Threading/SMT siblings and the CPU's internal interconnect can make fewer threads faster.
- Like the VRAM test it only **reads**, so it matches the pattern of CPU decode.

There is no "theoretical" RAM value, because reading the memory configuration needs root. Compute it yourself:

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

### Several GPUs

All GPUs are tested one after another, never at the same time, so they do not disturb each other.
GPU numbers follow CUDA's order. To know which physical card is which, use the PCI address in brackets; `nvidia-smi` shows the same address. Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA numbers match the order of `nvidia-smi`.

---

## Troubleshooting

**`nvcc: not found -> building without CUDA (RAM test only)`**
CUDA is not installed or not where expected. Use `./configure --cuda=/path/to/cuda`.

**Test compile fails / `unsupported GNU version`**
The CUDA version does not accept your `g++`. Install an older one and use `--host-cxx=g++-10` (or another supported version), or try `--allow-unsupported-compiler`.

**`GPU   no usable CUDA GPU (...)`**
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
src/gpu.cu       VRAM and PCIe tests (CUDA, built only when nvcc is found)
src/progress.h   live progress line
```

---

## License

MIT. See [LICENSE](LICENSE).

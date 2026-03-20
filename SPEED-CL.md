# OpenCL conv2d kernel optimization log

Hardware: Qualcomm Adreno 830 (Snapdragon 8 Gen 3), OpenCL 3.0, 1024 MB max alloc
Remote: Termux on Android, `LD_LIBRARY_PATH=/vendor/lib64`
Model: SD v2.1 Q8_0 (f16:1015, q8_0:291 — 2199.61 MB total)
Benchmark: 512×512, 20 steps, Euler A, seed 42, `--diffusion-conv-direct --vae-conv-direct`

## Results summary (SD v2.1 Q8_0)

| # | Version | Per step | Sampling (20 steps) | VAE decode | Total | Speedup vs baseline |
|---|---------|----------|---------------------|------------|-------|---------------------|
| 0 | im2col + matmul (no `--*-conv-direct`) | 14.74 s/it | 294.88s | crash (OOM) | — | — |
| 0a | im2col + matmul, `--vae-on-cpu` | 16.26 s/it | 326.20s | 60.16s (CPU) | 386.77s | — |
| 1 | **Direct conv2d (baseline, BS_CRS=16)** | **7.07 s/it** | **141.33s** | **10.34s** | **152.20s** | **1.0× (baseline)** |
| 2 | **Optimized (BS_CRS=32, 1×1 fast path, contiguous weights)** | **6.27 s/it** | **125.33s** | **10.49s** | **136.16s** | **1.12×** |

---

## Detailed changelog

### 0. im2col + matmul (default, no conv-direct) — 14.74 s/it

The default convolution path decomposes `CONV_2D` into `IM2COL → MUL_MAT → reshape`.
The VAE crashes because the compute buffer (1664 MB) exceeds the Adreno 830's
1024 MB max single allocation limit. With `--vae-on-cpu`, VAE decode takes 60s.

### 1. Direct conv2d (baseline) — 7.07 s/it (2.1× vs im2col)

Using `--diffusion-conv-direct --vae-conv-direct` activates the OpenCL conv2d kernel.
The implicit GEMM approach eliminates im2col intermediate buffers entirely.
VAE compute buffer drops from 1664 MB to 704 MB — fits within the 1024 MB
max alloc limit, enabling VAE decode on GPU in 10.34s.

### 2. BS_CRS 32, 1×1 fast path, contiguous weights — 6.27 s/it (+12%)

**Three changes applied together:**

1. **BS_CRS 16→32 (larger K-tile).** Doubles the reduction dimension tile from 16
   to 32. Halves the number of K iterations and `barrier(CLK_LOCAL_MEM_FENCE)` calls.
   Shared memory increases from ~4 KB to ~8 KB (well within the 32 KB limit).

   For the largest conv layers (IC=1280, 3×3, CRS=11520): iterations drop from
   720 to 360, saving 360 barrier synchronizations per workgroup.

2. **1×1 conv fast path.** When KH=KW=1 (attention projections, skip connections),
   the kernel skips all index decomposition for both A-tile and B-tile loading.
   For A-tile: `knl_data[crs_g * nb02 + k_g * nb03]` (no division).
   For B-tile: `Cin_idx = crs_g` directly (no `crs_g / KHW` division).

3. **Contiguous weight loading.** When weights are contiguous in memory
   (`nb01 == KW && nb02 == KH*KW`), the A-tile load simplifies to
   `knl_data[k_g * nb03 + crs_g]` — a single add instead of 3 integer divisions
   (`crs_g / KHW`, `rem / KW`, `rem % KW`). This applies to all standard conv
   layers in SD v2.1.

**Result:**
- Sampling: 125.33s vs 141.33s → **11% faster**
- VAE decode: 10.49s (unchanged, expected — same conv2d kernel used in both)
- Total: 136.16s vs 152.20s → **11% faster overall**

```
  |==================================================| 20/20 - 6.27s/it
  sampling completed, taking 125.33s
  decode_first_stage completed, taking 10.49s
  generate_image completed in 136.16s
```

---

## GPU memory usage (512×512)

Measured by polling `/proc/meminfo` → `GpuTotal` every 300ms during inference.
System has 5.4 GB total GPU-addressable memory, idle baseline ~556 MB.

### Peak GPU memory (from `/proc/meminfo GpuTotal`)

| Configuration | Peak GpuTotal | Delta vs idle | Steady-state (sampling) |
|---------------|---------------|---------------|------------------------|
| Idle (no inference) | 556 MB | — | — |
| im2col + matmul, `--vae-on-cpu` | **3193 MB** | **2637 MB** | ~2861 MB |
| Direct conv2d (optimized) | **3287 MB** | **2731 MB** | ~1884 MB (sampling), ~3287 MB (VAE) |

### Compute buffer sizes (from ggml debug output, VRAM)

| Component | im2col + matmul | Direct conv2d | Saving |
|-----------|----------------|---------------|--------|
| U-Net compute buffer | 367.98 MB | 367.70 MB | 0.28 MB |
| VAE compute buffer | **1664 MB (crash on GPU)** | **704.06 MB** | **960 MB (58%)** |
| CLIP compute buffer | 1.89 MB | 1.89 MB | — |
| Model params (fixed) | 2199.61 MB | 2199.61 MB | — |

### Memory profile (direct conv2d, optimized)

GPU memory grows through distinct phases:

| Phase | GpuTotal | What's allocated |
|-------|----------|-----------------|
| Idle | ~556 MB | System/compositor |
| Model loading | ~700 MB | CLIP params (698 MB) loading |
| Params loaded | ~1884 MB | All params: CLIP (698) + U-Net (1407) + VAE (94) |
| Sampling (U-Net) | ~2958 MB | + U-Net compute buffer (368 MB) + CLIP buffers |
| VAE decode (peak) | **~3287 MB** | + VAE compute buffer (704 MB), U-Net buffer freed |

### Why VAE crashes without conv-direct

The Adreno 830 reports `max mem alloc size: 1024 MB`. The im2col path needs a
1152 MiB single buffer for VAE decode — exceeds the hardware limit.
The direct conv2d path eliminates im2col intermediates, reducing the compute
buffer to 704 MB which fits.

### Memory headroom

With 5.4 GB total GPU memory and peak usage of 3.3 GB, there's ~2.1 GB headroom.
However, the 1024 MB max single allocation limit is the binding constraint —
not total memory. The VAE compute buffer at 704 MB leaves only 320 MB under
this limit.

---

## CPU vs OpenCL comparison

| Metric | CPU (4× Cortex-X4) | OpenCL im2col | OpenCL direct (baseline) | OpenCL direct (optimized) |
|--------|--------------------|--------------|--------------------------|----|
| Per step | 27.16 s/it | 14.74 s/it | 7.07 s/it | 6.27 s/it |
| Sampling (20 steps) | ~543s (est.) | 294.88s | 141.33s | 125.33s |
| VAE decode | 74.23s | crash | 10.34s | 10.49s |
| Peak GPU mem | 0 MB | 3193 MB | — | 3287 MB |
| **GPU speedup vs CPU** | — | 1.8× | 3.8× | **4.3×** |

---

## How to measure GPU memory usage

On Qualcomm Adreno (Android/Termux), GPU memory is tracked via the KGSL
(Kernel Graphics Support Layer) driver and exposed through `/proc/meminfo`.

### Reading GPU memory

```bash
grep -E 'GpuTotal|KgslShmemUsage|GpuSwap' /proc/meminfo
```

| Field | Meaning |
|-------|---------|
| `GpuTotal` | Total GPU memory currently allocated (KGSL buffers + GPU-mapped pages) |
| `KgslShmemUsage` | Shared memory allocated through KGSL (subset of GpuTotal) |
| `GpuSwap` | GPU memory currently swapped out |

### Peak measurement during inference

Poll `GpuTotal` at a fixed interval, log to a file, then extract the peak:

```bash
LOG=~/gpu_mem_log.txt
rm -f $LOG

# Start monitor in background
(while true; do
    grep "GpuTotal:" /proc/meminfo >> $LOG
    sleep 0.3
done) &
MONITOR_PID=$!

# Run inference
LD_LIBRARY_PATH=/vendor/lib64:$LD_LIBRARY_PATH build-cl/bin/sd-cli \
    -m models/stable-diffusion-v2-1-Q8_0.gguf \
    -p "prompt" --steps 20 --seed 42 \
    --diffusion-conv-direct --vae-conv-direct \
    -o output.png -v 2>&1 | grep -E "compute buffer|params memory"

# Stop monitor
kill $MONITOR_PID 2>/dev/null; wait $MONITOR_PID 2>/dev/null

# Report peak
peak=$(awk '{print $2}' $LOG | sort -n | tail -1)
echo "Peak GpuTotal: $peak kB ($((peak / 1024)) MB)"

# Distribution (shows memory at each phase)
awk '{print $2}' $LOG | sort -n | uniq -c
```

### Other useful GPU sysfs nodes

```bash
# GPU clock speed (MHz)
cat /sys/kernel/gpu/gpu_clock

# Max GPU clock
cat /sys/kernel/gpu/gpu_max_clock

# GPU model
cat /sys/kernel/gpu/gpu_model

# GPU busy percentage (instantaneous)
cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage

# Available GPU frequencies
cat /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies
```

### OpenCL device limits

```bash
LD_LIBRARY_PATH=/vendor/lib64:$LD_LIBRARY_PATH clinfo | grep -i -E 'global mem|max alloc|local mem|max work'
```

Key limits for Adreno 830:
- Global memory: 5.4 GB
- Max single allocation: 1024 MB
- Local memory: 32 KB (640 KB preferred, includes on-chip global memory)
- Max workgroup size: 1024

### ggml-level compute buffer sizes

Run with `-v` (verbose) to see per-component VRAM allocation:

```bash
sd-cli -m model.gguf -p "test" --steps 1 -v 2>&1 | grep "compute buffer"
```

Output shows the compute buffer allocated for each model component (CLIP, U-Net,
VAE) and whether it's in VRAM or RAM. This is the ggml-level view — the actual
GPU driver allocation (from `/proc/meminfo`) may be higher due to alignment,
page granularity, and driver overhead.

---

## Remaining optimization avenues

- **mul_mat kernel tuning**: The U-Net spends most time in mul_mat (attention, linear
  layers), not conv2d. Tuning the OpenCL mul_mat kernels for Adreno could yield
  larger gains than further conv2d work.

- **Flash attention**: `--fa` flag for fused attention — reduces memory and compute
  for attention layers.

- **CLIP offload**: `--clip-on-cpu` frees ~698 MB VRAM at cost of ~200ms slower conditioning.

- **VAE tiling**: `--vae-tiling` for 768×768+ resolution on GPU.

- **q8_0 conv2d kernel**: Direct convolution on quantized weights using
  `cl_khr_integer_dot_product` / `cl_qcom_dot_product8` — avoids dequantization.

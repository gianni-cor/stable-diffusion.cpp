# Adreno OpenCL utilities

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

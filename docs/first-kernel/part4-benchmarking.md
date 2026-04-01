# Part 4: Benchmarking and Performance Tuning

*This tutorial follows [Part 3](part3-verifying-numerics.md) of this series.*

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 4: Benchmarking vs PyTorch and tuning](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

Welcome to the final part of our Vector Addition kernel tutorial series. In previous tutorials, we implemented the kernel and verified its numerical accuracy against PyTorch. With that solid foundation, this tutorial focuses on benchmarking and tuning the kernel to optimize its performance.

## Step 1: Setting Up the Benchmarking Framework

To comprehensively evaluate performance, we use a benchmarking function that runs the kernel over a range of vector sizes. The sizes are powers of two, starting from 2^10 (1024) up to 2^28, covering a broad spectrum of tensor sizes.

## Step 2: Configuring the Benchmark

```python
providers = [
    {"name": "PyTorch", "color": "blue"},
    {"name": "Triton", "color": "orange"}
]
```

## Step 3: Running the Benchmark

```python
import time

def benchmark_addition(vector_size, provider, block_size=128, num_warps=4):
    A = torch.randn(vector_size, device='cuda', dtype=torch.float32)
    B = torch.randn(vector_size, device='cuda', dtype=torch.float32)
    output = torch.empty_like(A)
    start = time.time()
    if provider == "PyTorch":
        output = A + B
    elif provider == "Triton":
        kernel_vector[(vector_size // block_size,)](A, B, output, vector_size, block_size, num_warps=num_warps)
    torch.cuda.synchronize()
    end = time.time()
    # GB/s: bytes read (2 inputs) + bytes written (1 output) per second
    gbps = (A.numel() * A.element_size() * 3) / (end - start) / 1e9
    return gbps
```

## Step 4: Interpreting Initial Results

Vector addition is a **memory-bandwidth-bound** operation: for each element, we perform one addition but must load two values from HBM and store one result. The kernel's performance is therefore limited by how fast we can move data, not by arithmetic throughput.

A useful reference point is the **peak memory bandwidth** of your GPU (e.g., ~2 TB/s on an H100, ~935 GB/s on an A100, ~900 GB/s on an A10). Comparing your measured GB/s to this peak tells you how efficiently your kernel is using the available bandwidth. Getting close to the hardware peak is the goal for bandwidth-bound kernels.

After running the initial benchmarks, you will observe that Triton's performance largely matches PyTorch's across most tensor sizes. There may be a slight performance gap at mid-range sizes, but at larger tensor sizes, Triton begins to match or exceed PyTorch.

## Step 5: Performance Tuning — Adjusting Block Size

To improve performance, increase the block size from 128 to 1024 and rerun the benchmark. A larger block size allows each kernel instance to process more elements per launch, which can improve GPU utilization and reduce launch overhead.

```python
block_size = 1024
# Rerun benchmark with new block size
```

## Step 6: Performance Tuning — Adjusting Number of Warps

Next, tune the number of warps used by the kernel. The default is 4 warps (128 threads), but increasing to 8 warps allows the GPU to better hide memory latency by switching between warps while one is waiting for data.

```python
num_warps = 8
kernel_vector[(vector_size // block_size,)](A, B, output, vector_size, block_size, num_warps=num_warps)
```

## Step 7: Final Benchmark Results

With the increased block size and number of warps, the benchmark shows that Triton kernel performance matches or slightly exceeds PyTorch's performance across the tested range.

## Conclusion

In this tutorial, you learned how to benchmark and tune a custom Triton vector addition kernel, comparing its performance to PyTorch and optimizing key parameters for maximum throughput. By setting up a flexible benchmarking framework, interpreting performance results in terms of memory bandwidth utilization, and tuning block size and warp count, you gained practical skills for developing high-performance GPU kernels.

---

**Next:** [Softmax →](../softmax/index.md)

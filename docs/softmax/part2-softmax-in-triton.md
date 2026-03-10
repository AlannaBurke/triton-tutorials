# Part 2: Softmax in Triton

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Coding a Triton Kernel for Softmax (fwd pass) Computation](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

In this tutorial, we'll walk through implementing a custom softmax kernel in Triton, including both the kernel itself and the host program that launches it. We'll cover memory management, parallelization, pointer arithmetic, and benchmark our implementation against PyTorch's native softmax and a naive implementation.

## 1. Understanding Triton Kernels and Host Programs

When working with Triton, you typically write two components:

- **The Kernel:** The function that runs on the GPU, processing data in parallel.
- **The Host Program:** The Python code that sets up meta-information (like block size and grid shape), allocates memory, and launches the kernel.

## 2. The Softmax Kernel and Host Program

```python
import torch
import triton
import triton.language as tl

def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

@triton.jit
def softmax_kernel(
    output_ptr, output_stride,
    input_ptr, input_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    input_row_ptr = input_ptr + row_idx * input_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_row_ptr + col_offsets, mask=mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denom = tl.sum(numerator, axis=0)
    softmax = numerator / denom
    output_row_ptr = output_ptr + row_idx * output_stride
    tl.store(output_row_ptr + col_offsets, softmax, mask=mask)

def triton_softmax(x):
    assert x.ndim == 2, "Only 2D tensors are supported in this example."
    n_rows, n_cols = x.shape
    BLOCK_SIZE = next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16
    grid = (n_rows,)
    output = torch.empty_like(x)
    softmax_kernel[grid](
        output, output.stride(0),
        x, x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output
```

## 3. Key Implementation Details

**Memory and Pointer Arithmetic:**
Tensors in memory are 1D arrays; strides are used to navigate between rows. The kernel receives pointers to the input and output data, and uses pointer arithmetic (`row_idx * stride`) to locate the correct row. Masking ensures we only read/write valid data when the number of columns is not a power of two.

**Parallelization:**
We launch one kernel instance per row (`grid = (n_rows,)`). Each instance processes a full row independently, with `BLOCK_SIZE` set to the next power of two greater than or equal to the number of columns.

**Softmax Computation:**
For numerical stability, we subtract the row maximum before exponentiating (the same technique as naive softmax). We then sum the exponentiated values to get the denominator and divide.

## 4. Debugging: A Common Pointer Arithmetic Pitfall

Pointer arithmetic is error-prone. A common mistake is to compute the output row pointer incorrectly — for example, reusing the input row pointer for the output, or forgetting to multiply by `output_stride`. This causes all rows to be written to the same location, producing incorrect results that can be hard to diagnose.

Always verify that your output pointer is computed as `output_ptr + row_idx * output_stride`, and that masking is applied consistently to both `tl.load` and `tl.store`.

## 5. Numerical Verification and Benchmarking

Before benchmarking, always verify correctness:

```python
import torch.nn.functional as F

x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
y_triton = triton_softmax(x)
y_torch = F.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch, atol=1e-6), "Numerical mismatch!"
print("Triton softmax matches PyTorch softmax.")
```

Then benchmark across a range of tensor sizes:

```python
import time

def naive_softmax(x):
    x_max = x.max(dim=1, keepdim=True).values
    x_safe = x - x_max
    numerator = torch.exp(x_safe)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator

sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
for n_cols in sizes:
    x = torch.randn(1024, n_cols, device='cuda', dtype=torch.float32)
    start = time.time(); y_triton = triton_softmax(x); triton_time = time.time() - start
    start = time.time(); y_torch = F.softmax(x, dim=1); torch_time = time.time() - start
    start = time.time(); y_naive = naive_softmax(x); naive_time = time.time() - start
    print(f"Cols: {n_cols} | Triton: {triton_time:.4f}s | PyTorch: {torch_time:.4f}s | Naive: {naive_time:.4f}s")
```

## Conclusion

In this tutorial, you learned how to write and launch a custom softmax kernel in Triton, verify its numerical correctness, and benchmark its performance against PyTorch and a naive implementation.

---

**Next:** [Tiled Matrix Multiplication →](../advanced/tiled-matmul.md)

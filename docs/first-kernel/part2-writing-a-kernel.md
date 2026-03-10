# Part 2: Writing a Vector Addition Kernel

*This tutorial follows [Part 1](part1-parallel-programming.md) of this series.*

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 2: Coding the Triton Kernel](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

In this tutorial, we walk through the process of writing a complete vector addition kernel in Triton. We start by creating a host function that prepares the inputs and output buffers, sets up the execution grid, and calls the Triton kernel. Then, we implement the kernel itself, which performs element-wise addition of two input vectors on the GPU.

## Step 1: Importing Libraries

```python
import triton
import triton.language as tl
import torch
```

## Step 2: Writing the Host Function

The host function is the Python-side interface for running the kernel. It takes two input vectors, A and B, and returns their element-wise sum as a PyTorch tensor. Inside the host program:

- We create an output buffer with the same shape and device as A.
- We assert that both A and B are CUDA tensors and have the same size.
- We determine the number of elements to process.
- We define a block size to chunk the workload.
- We compute the grid size using ceiling division to ensure all elements are covered.
- We call the Triton kernel with the prepared parameters.

```python
def vector_addition(A, B):
    output = torch.empty_like(A)
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.numel() == B.numel(), "Input tensors must have the same number of elements"
    num_elements = A.numel()
    block_size = 128
    grid_size = (triton.cdiv(num_elements, block_size),)
    kernel_vector[grid_size](A, B, output, num_elements, block_size)
    return output
```

## Step 3: Writing the Triton Kernel

The kernel performs the core computation on the GPU. Key points:

- `tl.program_id(0)` retrieves the block ID for the current program instance.
- `block_start` calculates the starting element index for this block.
- `thread_offsets` computes all element indices this block will process.
- `mask` prevents out-of-bounds memory access.
- `tl.load` and `tl.store` read from and write to HBM using the computed offsets and mask.

Marking `block_size` as a `tl.constexpr` allows the compiler to treat it as a compile-time constant, enabling additional optimizations such as loop unrolling.

```python
@triton.jit
def kernel_vector(a_ptr, b_ptr, out_ptr, num_elements, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elements
    a_vals = tl.load(a_ptr + thread_offsets, mask=mask)
    b_vals = tl.load(b_ptr + thread_offsets, mask=mask)
    result = a_vals + b_vals
    tl.store(out_ptr + thread_offsets, result, mask=mask)
```

## Step 4: Debugging Tips

Triton provides `tl.device_print` for debugging inside kernels. It does not support formatted strings, but can print scalar values such as the current program ID:

```python
# Inside the kernel (remove before production use):
tl.device_print("pid: ", pid)
```

## Conclusion

By completing this tutorial, you will have written and executed a complete vector addition kernel in Triton, gaining hands-on experience with both the Python host side and the GPU kernel side of the workflow.

---

**Next:** [Part 3: Verifying Numerics →](part3-verifying-numerics.md)

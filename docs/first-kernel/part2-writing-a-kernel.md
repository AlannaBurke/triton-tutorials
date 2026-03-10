# Part 2: Writing a Vector Addition Kernel

*This tutorial follows Part 1 of this series.*

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 2: Coding the Triton Kernel](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

In this tutorial, we will walk through the process of writing a vector addition kernel using Triton, a language and compiler designed for writing efficient GPU code. This tutorial follows Part One of this Triton series. We will start by creating a host function that prepares the inputs and output buffers, sets up the execution grid, and calls the Triton kernel. Then, we will implement the kernel itself, which performs element-wise addition of two input vectors on the GPU. Finally, we will discuss some optimization considerations and debugging tips.

## Step 1: Importing Libraries and Setting Up

We begin by importing the necessary libraries. Triton is imported along with its language module, and PyTorch is imported for comparison purposes, both for performance and correctness.

```python
import triton
import triton.language as tl
import torch
```

## Step 2: Writing the Host Function

The host function serves as the interface for users to run the kernel. It takes two input vectors, A and B, and returns their element-wise sum as a PyTorch tensor.

Inside the host program:
- We create an output buffer with the same shape and device as A.
- We assert that both A and B are CUDA tensors and have the same size.
- We determine the number of elements to process.
- We define a block size to chunk the workload.
- We compute the grid size using a ceiling division function to ensure all elements are covered.
- We call the Triton kernel with the prepared parameters.

```python
def vector_addition(A, B):
    output = torch.empty_like(A)
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.numel() == B.numel(), "Input tensors must have the same number of elements"
    num_elements = A.numel()
    block_size = 128
    def ceil_div(x, y):
        return (x + y - 1) // y
    grid_size = (ceil_div(num_elements, block_size),)
    kernel_vector[grid_size](A, B, output, num_elements, block_size)
    return output
```

## Step 3: Implementing the Ceiling Division Function

The ceiling division function ensures that the grid size covers all elements, even if the last block has fewer elements than the block size.

```python
def ceil_div(x, y):
    return (x + y - 1) // y
```

## Step 4: Writing the Triton Kernel

The kernel performs the core computation on the GPU. Key points include:
- Retrieving the program ID to identify the current block.
- Calculating the starting offset for this block based on the block size and program ID.
- Computing thread offsets within the block.
- Applying a mask to prevent out-of-bounds memory access.
- Loading elements from input vectors A and B using the computed offsets and mask.
- Performing element-wise addition.
- Storing the result back to the output buffer with masking.

Marking the number of elements and block size as constant expressions allows the compiler to optimize the kernel, which can lead to significant performance improvements.

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

## Step 5: Debugging Tips

Triton provides a simple device print function for debugging purposes. While it does not support formatted strings, it can be used to print values such as the current program ID to the console during kernel execution.

```python
# Inside the kernel (remove before production use):
tl.device_print("pid: ", pid)
```

## Conclusion

By completing this tutorial, you will have written and executed a complete vector addition kernel in Triton, gaining hands-on experience with both the Python and GPU sides of the workflow.

---

**Next:** [Part 3: Verifying Numerics →](part3-verifying-numerics.md)

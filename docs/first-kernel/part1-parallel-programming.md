# Part 1: Making the Shift to Parallel Programming

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 1: Making the Shift to Parallel Programming](https://youtu.be/MEZ7XhzTLEg?si=QYQb9slrDQja8wrp)*

## Introduction

In this tutorial, we will explore how to write a vector addition kernel using Triton, a language and compiler designed for writing efficient GPU code. We begin with the mental model shift required to move from traditional serial programming to parallel programming. Understanding this shift is crucial for effectively leveraging GPU architectures.

## Mental Model Shift: From Serial to Parallel Programming

Consider a simple problem where we have two integer vectors, A and B, each containing seven elements. Our goal is to compute their element-wise sum and store the result in vector C.

### Serial Approach

In a conventional single-threaded programming model (e.g., Python or C++), you would process one element at a time:

```python
# Serial vector addition in Python
A = [1, 2, 3, 4, 5, 6, 7]
B = [7, 6, 5, 4, 3, 2, 1]
C = []
for i in range(len(A)):
    C.append(A[i] + B[i])
print(C)  # Output: [8, 8, 8, 8, 8, 8, 8]
```

### Parallel Approach with Triton

Triton introduces a different paradigm by leveraging parallelism on the GPU. Instead of processing elements sequentially, we divide the workload across many independent **program instances** organized into a grid. Each program instance operates on a contiguous block of the data.

For example, if we choose a block size of 2, each program instance processes two elements. For our problem of seven elements, we create a grid of four program instances (4 × 2 = 8 slots, which covers all 7 elements plus one extra). Each instance runs independently and simultaneously on the GPU.

> **Note:** In Triton, `tl.program_id(axis=0)` returns the **block ID** of the current program instance — it identifies which block of data this instance is responsible for, not an individual thread ID.

## Handling Edge Cases with Masking

A key challenge arises when the total number of elements is not a multiple of the block size. In our example, the last program instance covers indices 6 and 7, but only index 6 has valid data. The thread handling index 7 has no valid data to work on.

To handle this safely, we use **masking**. Each program instance checks whether its assigned indices are within the valid range. Indices outside the valid range are masked out, preventing out-of-bounds memory access. Proper masking is essential — without it, the kernel would read or write to arbitrary memory addresses.

## Complete Example

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)  # Each program instance has a unique block ID
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # Mask to avoid out-of-bounds
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

# The host program to launch the kernel
def triton_vector_add(A, B):
    assert A.shape == B.shape
    N = A.numel()
    BLOCK_SIZE = 128
    C = torch.empty_like(A)
    # Grid must be a tuple; the trailing comma makes (expr,) a 1-element tuple
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)
    return C

# Example usage
A = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device='cuda')
B = torch.tensor([7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
C = triton_vector_add(A, B)
print(C.cpu().tolist())  # Output: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
```

Key points in the code:

- `tl.program_id(0)` returns the unique block ID for this program instance.
- `offsets` computes the element indices this instance will process.
- `mask` ensures no out-of-bounds memory access occurs.
- The kernel loads, adds, and stores elements in parallel across all instances.

## Conclusion

By completing this tutorial, you have developed a strong conceptual understanding of the shift from serial to parallel programming, which is fundamental for writing efficient GPU kernels with Triton.

---

**Next:** [Part 2: Writing a Kernel →](part2-writing-a-kernel.md)

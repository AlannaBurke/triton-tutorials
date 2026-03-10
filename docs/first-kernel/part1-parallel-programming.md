# Part 1: Making the Shift to Parallel Programming

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 1: Making the Shift to Parallel Programming](https://youtu.be/MEZ7XhzTLEg?si=QYQb9slrDQja8wrp)*

## Introduction

In this tutorial, we will explore how to write a vector addition kernel using Triton, a language and compiler designed for writing efficient GPU code. The tutorial is divided into two parts. First, we will cover the mental model shift required to move from traditional serial programming to parallel programming. Understanding this shift is crucial for effectively leveraging GPU architectures. Then, we will dive into the actual coding of the Triton kernel, illustrating how to implement vector addition in a parallelized manner.

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

Triton introduces a different paradigm by leveraging parallelism on the GPU. Instead of processing elements sequentially, we divide the workload across many threads organized into blocks and grids. Each thread operates independently on a subset of the data.

For example, if we choose a block size of 2, each block will have two threads. For our problem of seven elements, we create a grid of four blocks (since 4 blocks × 2 threads = 8 threads, which covers all 7 elements plus one extra). Each thread in a block processes one element of the vectors simultaneously.

## Handling Edge Cases with Masking

A key challenge arises when the total number of elements is not a multiple of the block size. In our example, the last block has two threads, but only one element remains to be processed. The second thread in the last block has no valid data to work on.

To handle this safely, we use masking. Each thread checks if its assigned index is within the valid range of elements. Threads with indices outside the valid range become inert and do not perform any load, compute, or store operations. Proper masking prevents out-of-bounds memory access, which would otherwise cause program crashes.

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
    pid = tl.program_id(0)  # Each block has a unique program ID
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
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)
    return C

# Example usage
A = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device='cuda')
B = torch.tensor([7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
C = triton_vector_add(A, B)
print(C.cpu().tolist())  # Output: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
```

Key points in the code:
- `tl.program_id(0)` gets a unique ID in the grid.
- `offsets` computes the indices this block's threads will process.
- `mask` ensures threads do not access out-of-bounds memory.
- The kernel loads, adds, and stores elements in parallel.

## Conclusion

By completing this tutorial, you have developed a strong conceptual understanding of the shift from serial to parallel programming, which is fundamental for writing efficient GPU kernels with Triton.

---

**Next:** [Part 2: Writing a Kernel →](part2-writing-a-kernel.md)

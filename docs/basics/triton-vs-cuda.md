# Triton vs. CUDA

Triton is an open-source, Python-based programming language and compiler designed for writing high-performance custom Deep Neural Networks (DNNs) compute kernels, particularly targeting NVIDIA GPUs. Compute Device Unified Architecture (CUDA), on the other hand, is a parallel computing platform and programming model developed by NVIDIA, which allows developers to leverage the power of NVIDIA GPUs for general-purpose computing. Triton is often preferred over CUDA because it simplifies GPU programming for machine learning, particularly for Python developers, while still allowing for deep optimization.

This guide compares the key differences between Triton and CUDA.

## Key Differences

| Feature | Triton | CUDA |
|---------|--------|------|
| Programming language | Python language | C or C++ |
| Target audience | Ideal for deep learning researchers and developers who want to write custom kernels for deep learning models without needing extensive CUDA expertise. | A more general-purpose platform for GPU programming, used in various fields, including deep learning, scientific computing, and game development. |
| Performance | While Triton aims to simplify GPU programming, it may come with a slight performance penalty compared to highly optimized CUDA kernels. | Offers the potential for very high performance, especially when developers are deeply familiar with GPU architecture. |
| Programming model | Higher-level DSL focused on deep learning. Uses a single-level decomposition (blocks). | Low-level, general-purpose. Uses a two-level decomposition (threads and blocks). |
| Learning curve | Lower: Employs a higher-level, more Pythonic approach. | Higher: Requires a deeper understanding of GPU architecture. |

## Compiler Optimizations: Triton vs. CUDA

Triton automates many compiler optimizations commonly done manually in CUDA, including memory coalescing, shared memory allocation, and instruction scheduling.

| Feature | Triton | CUDA |
|---------|--------|------|
| Memory coalescing | Automatic | Manual |
| Shared memory management | Automatic | Manual |
| Scheduling (within SMs) | Automatic | Manual |
| Scheduling (across SMs) | Manual | Manual |

## Example: Vector Addition — Triton vs. CUDA

**Triton:**

```python
# triton_vector_addition.py
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x, y):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output

x = torch.arange(1024, dtype=torch.float32, device="cuda")
y = torch.arange(1024, dtype=torch.float32, device="cuda") * 2
result = vector_add(x, y)
print(result[:10])
```

Output: `tensor([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24., 27.], device='cuda:0')`

**CUDA:**

```cpp
// cuda_vector_add.cu
#include <cuda.h>
#include <iostream>

__global__ void add_kernel(const float* a, const float* b, float* c, int vector_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vector_size) { c[i] = a[i] + b[i]; }
}

int main() {
  int n = 1024;
  // ... (allocate, copy, launch, print, free)
  return 0;
}
```

Output: `0 3 6 9 12 15 18 21 24 27`

## Conclusion

By completing this tutorial, you will have a clear understanding of how Triton and CUDA differ in their approach to GPU programming, especially for deep learning and scientific computing.

---

**Next:** [GPU Fundamentals →](gpu-fundamentals.md)

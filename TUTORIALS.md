Triton Tutorial Series — Outline

How to Use This Series

This series of tutorials is designed to be a hands-on guide to learning Triton, from fundamental concepts to writing and optimizing your own GPU kernels. Each tutorial is self-contained and can be run end-to-end in a Jupyter notebook environment with a GPU runtime. The expected time to complete the full series is approximately 8–12 hours, depending on your pace and how much you experiment with the code.

Target Audience and Prerequisites

This series is intended for developers and researchers who are familiar with Python and PyTorch and want to learn how to write custom GPU kernels to accelerate their workloads. No prior experience with GPU programming or CUDA is required.

Prerequisites:
- Python 3.x
- PyTorch (with CUDA support)
- A CUDA-enabled NVIDIA GPU. All tutorials require a GPU runtime and will not work in a CPU-only environment (e.g., if using Google Colab, select a GPU runtime under Runtime > Change runtime type).
- Jupyter Notebook or JupyterLab (Google Colab works well)

Basics
- Intro to Triton
- What is Triton?
- When to use Triton vs. PyTorch eager mode
- How Triton fits into the PyTorch ecosystem
- Getting Started with Triton
- Triton vs. CUDA
- Introduction to GPU Fundamentals for Kernel Programming
- First Kernel - Vector Addition
- Making the shift to parallel programming
- Coding a vector addition kernel.
- Verifying numerics.
- Benchmarking vs PyTorch.
- Softmax
- Basic softmax in PyTorch
- Online softmax in PyTorch
- Softmax in Triton
- Tiled Matrix Multiplication in Triton
- Part 1: Flash Attention 2: Overview
- Part 2: Flash Attention 2: Forward Kernel
- Part 3: Flash Attention 2: Backward Kernel
- Part 4: Benchmarking / Tuning
Advanced Topics
- TMA
- Warp Specialization
- Coop Matmul in Triton


---

1. Basics
---


Intro to Triton

Introduction to Triton

1. What is Triton?

Triton is an open-source, Python-based programming language and compiler for writing highly efficient custom GPU kernels. It transforms high-level Python code into GPU machine instructions, enabling developers and researchers—even those with little or no experience in GPU hardware or specialized languages like CUDA—to write expressive, parallel programs for accelerators such as NVIDIA GPUs. Triton's Pythonic syntax is familiar to users of scientific computing libraries like NumPy and PyTorch, making it accessible for a wide range of scientific and engineering applications.

Key characteristics of Triton:
- Pythonic API: Write GPU code in a style similar to NumPy or PyTorch, making it accessible to Python developers.
- Custom GPU Kernels: Allows you to write your own GPU operations (kernels) for tasks where built-in libraries may not be optimal.
- Automatic Compilation: Triton compiles your Python-like code into highly optimized GPU code, abstracting away much of the complexity of CUDA or other low-level languages.
- Focus on Performance: Designed to achieve performance close to hand-written CUDA, but with much less boilerplate and complexity.

2. When Should You Use Triton Instead of PyTorch Eager Mode?

PyTorch eager mode executes operations immediately as they are called, which is excellent for rapid prototyping and flexibility. However, it relies on pre-built GPU kernels from libraries such as cuBLAS and cuDNN for most operations. This means you are limited to what those libraries provide.

Triton is the right choice when:
- You need a custom GPU operation that PyTorch does not provide out of the box.
- You want to fuse multiple operations into a single kernel to reduce memory bandwidth usage.
- You need to optimize a specific computation for your workload beyond what generic PyTorch kernels offer.

In short: use PyTorch eager mode for standard operations and prototyping; reach for Triton when you need custom, high-performance GPU code.

Triton, PyTorch Eager, and Helion compared

Feature              | PyTorch Eager            | Triton                               | Helion
---------------------|--------------------------|--------------------------------------|--------------------------------------
Custom GPU Kernels   | Limited                  | Full control                         | Partial (automated tiling)
Performance Tuning   | Limited to built-ins     | Fine-grained, user-optimized         | Automatic search
API Style            | Pythonic, high-level     | Pythonic, kernel-focused             | Pythonic, high-level with tiling abstraction
Integration          | Native                   | Seamless with PyTorch                | Seamless with PyTorch
Learning Curve       | Low                      | Moderate (easier than CUDA)          | Moderate

Helion is a Python-embedded DSL for writing high-performance ML kernels that compiles to Triton. It offers a higher-level abstraction, with features like automatic grid size calculation and autotuning of data tile mappings, making it even easier to get started with custom kernels.

3. How Triton Fits into the PyTorch Ecosystem

The most important thing to understand is that Triton is not a replacement for PyTorch—it is a powerful extension. You keep all of the productivity and ecosystem of PyTorch while gaining the ability to write custom, high-performance GPU operations when you need them.

Triton integrates with PyTorch in two key ways:
- Direct use: You can write a Triton kernel and call it directly from your PyTorch code, passing PyTorch tensors as arguments.
- TorchInductor: PyTorch's compiler backend (torch.compile) uses Triton as a code generator for GPU operations. When you call torch.compile on a PyTorch model, Inductor may automatically generate Triton kernels to accelerate it—so even if you never write a Triton kernel yourself, Triton is likely running under the hood.

4. How does Triton work?

A Triton program has two parts: a kernel that runs on the GPU (the device), and a host program that runs on the CPU and is responsible for preparing data, configuring the launch, and calling the kernel.

A Triton kernel is a regular Python function decorated with @triton.jit, placed in a .py file or notebook cell:

@triton.jit
def kernel_name(...):
# GPU code goes here

The host program launches the kernel by indexing it with a grid:

result = kernel_name[grid](...)

The grid specifies how many independent instances of the kernel to launch, organized as a 1D, 2D, or 3D tuple:

grid = (X,)        # 1D grid: X kernel instances
grid = (X, Y)      # 2D grid: X*Y kernel instances
grid = (X, Y, Z)   # 3D grid: X*Y*Z kernel instances

Each kernel instance is called a program. Inside the kernel, tl.program_id(axis=0) returns the unique block ID of the current program instance along the given axis. This is how each instance knows which slice of the data it should process.

The @triton.jit decorator tells the Triton compiler to JIT-compile this function into a GPU kernel the first time it is called.

The Compilation Process

The Triton compilation process involves several stages:

1. Kernel declaration and compilation:
- The @triton.jit decorator marks the function for JIT compilation.
- On first call, the Triton compiler walks the Python AST to generate Triton-IR.

2. Intermediate representation (IR) generation:
- Triton-IR is a machine-independent, unoptimized representation of the kernel.

3. Optimization and lowering:
- Triton-IR is optimized and lowered to Triton-GPU IR (TTGIR), which encodes hardware-specific information such as how tensors are distributed across warps and compute capability.
- TTGIR is then lowered to LLVM-IR.

4. GPU code generation:
- The LLVM-IR is passed to vendor toolchains to produce executable GPU code.
- For NVIDIA GPUs: LLVM-IR → PTX (Parallel Thread Execution assembly) → CUBIN (CUDA binary), compiled by NVIDIA's ptxas assembler.
- For AMD GPUs: LLVM-IR → AMDGCN code → HSACO binary, compiled by AMD's toolchain.

5. Execution on GPU:
- The compiled binary is launched on the GPU according to the grid you specified.

Try It

Get your notebook ready

Before running the code in this tutorial, make sure you have a Jupyter Notebook environment with a GPU runtime. If you are using Google Colab, go to Runtime > Change runtime type and select a GPU accelerator. All tutorials in this series require GPU access and will fail on a CPU-only runtime with an error such as "torch compiled without CUDA enabled."

Install Triton (if needed)
Run this cell to install Triton. (You only need to do this once per environment.)

!pip install triton

Import Libraries

```python
import torch
import triton
import triton.language as tl

Define the Triton Kernel

@triton.jit
def vector_add_kernel(
x_ptr,           # Pointer to input vector x
y_ptr,           # Pointer to input vector y
output_ptr,      # Pointer to output vector
n_elements,      # Number of elements in the vectors
BLOCK_SIZE: tl.constexpr   # Number of elements processed per block (compile-time constant)
):
pid = tl.program_id(axis=0)  # Block ID for this kernel instance
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements  # Mask to handle out-of-bounds
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
result = x + y
tl.store(output_ptr + offsets, result, mask=mask)

Prepare Data and Launch the Kernel

# Set up input data on the GPU
data_x = torch.arange(10, dtype=torch.float32, device='cuda')
data_y = torch.arange(10, 0, -1, dtype=torch.float32, device='cuda')
output = torch.empty_like(data_x)
n_elements = data_x.numel()
BLOCK_SIZE = 1024
# Define grid: one kernel instance per block of BLOCK_SIZE elements
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
# Launch the kernel
vector_add_kernel[grid](data_x, data_y, output, n_elements, BLOCK_SIZE)

View the Results

print("Input x:", data_x)
```
print("Input y:", data_y)
print("Output (x + y):", output)

What You Should See

Input x: tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda:0')
Input y: tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.], device='cuda:0')
Output (x + y): tensor([10., 10., 10., 10., 10., 10., 10., 10., 10., 10.], device='cuda:0')

Conclusion

By completing this tutorial, you should now understand what Triton is and when to use it, how it fits into the PyTorch ecosystem, and the basics of how a Triton kernel is structured and compiled. With this foundation, you are ready to start writing your own kernels in the tutorials that follow.

---

Getting Started with Triton
---


Getting started with Triton

Triton is a Python-based language and compiler that transforms high-level Python code into GPU machine instructions (i.e., Parallel Thread Execution (PTX) code). It enables programmers with little or no experience of GPU hardware and GPU-specific programming languages, such as CUDA, to write very efficient parallel programs.

A typical Triton program consists of Triton kernels (functions) and a host program that calls the kernels. Triton kernels are executed in parallel by many threads on the GPU.

Before running the code in this tutorial, make sure you have a Jupyter Notebook environment with a GPU runtime (e.g., Google Colab with a GPU accelerator selected). All code in this series requires CUDA-enabled PyTorch and will fail on a CPU-only runtime.

Vector addition in GPU

The only way to learn Triton is by writing programs in it and testing them. The structure of a Triton program consists of two parts: a host part and a device part:
- The device part consists of Triton kernels that look like Python functions that are compiled and executed on the GPU.
- The host part handles tasks like loading data, launching kernels, and collecting results from the GPU.

For example, the following Triton kernel adds a scalar value to every element of a vector on the GPU:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
x_ptr,
increment_value,
output_ptr,
n_elements: tl.constexpr,
BLOCK_SIZE: tl.constexpr
):
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
output = x + increment_value
tl.store(output_ptr + offsets, output, mask=mask)

Note that this kernel adds a scalar (increment_value) to each element of the input vector x, rather than adding two vectors together. The following host code launches it to add 10 to each element:

BLOCK_SIZE = 1
data = [1, 2, 3, 4, 5]
x = torch.tensor(data, device='cuda')
output = torch.empty_like(x)
N = len(data)
# The grid lambda computes the number of blocks needed at launch time.
# triton.cdiv(N, BLOCK_SIZE) is ceiling division: (N + BLOCK_SIZE - 1) // BLOCK_SIZE
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
add_kernel[grid](x, 10, output, N, BLOCK_SIZE)

print(f"x = {x}")
```
print(f"x + 10 = {output}")

The output looks like this:

x = tensor([1, 2, 3, 4, 5], device='cuda:0')
x + 10 = tensor([11, 12, 13, 14, 15], device='cuda:0')

Testing Triton tutorials

Before testing the Triton tutorials, download them from the official Triton documentation page:
- Open the tutorials page at https://triton-lang.org/main/getting-started/tutorials/
- Scroll to the bottom of the page.
- Click Download all examples in Python source code: tutorials_python.zip.
- Extract the tutorials:

$ cd ~
$ mkdir tutorials_python/
$ unzip ~/Downloads/tutorials_python.zip -d tutorials_python/
$ cd tutorials_python/
$ ls
01-vector-add.py  06-fused-attention.py
02-fused-softmax.py  07-extern-functions.py
...

After downloading the Triton tutorials, you can run them in a Jupyter notebook or as standalone Python scripts. For example, to test 01-vector-add.py:
- Open an existing notebook or create a new one with a GPU runtime.
- Copy the contents of ~/tutorials_python/01-vector-add.py and paste it into a cell.
- Run the cell. If you encounter a CUDA error, verify that your environment has a GPU runtime enabled.

Conclusion

By completing this tutorial, you will have taken your first practical steps in GPU programming with Triton. You will understand how Triton kernels work, how to structure and execute a basic program, and how to test your code interactively in a Jupyter Notebook. With these foundational skills and access to further examples, you are well-prepared to explore more advanced GPU programming concepts and leverage Triton for high-performance computing tasks in your own projects.

---

Triton vs. CUDA
---


Triton vs. CUDA

Triton is an open-source, Python-based programming language and compiler designed for writing high-performance custom GPU kernels, particularly for deep learning workloads. CUDA, on the other hand, is a parallel computing platform and programming model developed by NVIDIA that gives developers direct, low-level access to GPU hardware using C or C++.

This guide compares the key differences between Triton and CUDA. Note that the GPU programming landscape is evolving rapidly: NVIDIA has recently introduced new Python-based DSLs such as CuTeDSL and CuTile (available in CUDA 13.1+) that bring a more Pythonic interface to CUDA-level programming. These are worth watching as the ecosystem matures.

Key differences between Triton and CUDA

Feature              | Triton                                      | CUDA
---------------------|---------------------------------------------|------------------------------------------
Programming language | Python                                      | C or C++
Target audience      | Deep learning researchers and developers who want to write custom kernels without extensive low-level GPU expertise. | General-purpose GPU programming across deep learning, scientific computing, game development, and more.
Performance          | Approaches hand-written CUDA performance for most deep learning workloads. The compiler handles many optimizations automatically. | Highest possible performance when the developer has deep knowledge of GPU architecture and can optimize manually.
Programming model    | Higher-level DSL. Operates on blocks of data; the compiler manages thread-level details within each block. | Low-level, general-purpose. Requires explicit management of threads, blocks, shared memory, and synchronization.
Learning curve       | Lower: Pythonic syntax, automatic memory coalescing and shared memory management. | Higher: Requires understanding of GPU architecture, thread management, memory access patterns, and synchronization.

Compiler optimizations in Triton vs. CUDA

One of Triton's key advantages is that it automates many of the low-level optimizations that must be done manually in CUDA. This includes memory coalescing (combining memory requests from multiple threads into a single transaction to maximize bandwidth), shared memory allocation, and instruction scheduling within a Streaming Multiprocessor (SM).

Feature                                    | Triton    | CUDA
-------------------------------------------|-----------|--------
Memory coalescing                          | Automatic | Manual
Shared memory management                   | Automatic | Manual
Instruction scheduling (within SMs)        | Automatic | Manual
Scheduling (across SMs)                    | Manual    | Manual

Example: Vector addition—Triton vs. CUDA

Triton:

# triton_vector_addition.py
import triton
import triton.language as tl
```python
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

Output: tensor([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24., 27.], device='cuda:0')

CUDA:

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

Output: 0 3 6 9 12 15 18 21 24 27

Conclusion

By completing this tutorial, you will have a clear understanding of how Triton and CUDA differ in their approach to GPU programming, especially for deep learning and scientific computing. Triton's Pythonic, high-level API lowers the barrier to writing custom GPU kernels, while CUDA offers maximum control and performance for those willing to manage low-level details.

---

Introduction to GPU Fundamentals for Kernel Programming
---


Introduction

This tutorial provides an introduction to the GPU hardware concepts that are most relevant to writing Triton kernels. Understanding the GPU memory hierarchy and execution model will help you write kernels that are both correct and performant.

This tutorial is derived from a SOTA Deep Learning Tutorials video:
Intro to Triton: A Parallel Programming Compiler and Language, esp for AI acceleration (updated)

What is Triton?

Triton is a programming language and compiler designed specifically for writing high-performance GPU kernels. It provides a higher-level, Pythonic interface that abstracts away many of the low-level details of GPU programming—such as explicit thread management and manual shared memory allocation—while still allowing you to control how data moves through the memory hierarchy. The Triton compiler handles optimizations like memory coalescing and shared memory usage automatically, enabling you to write custom kernels that approach the performance of hand-written CUDA code.

GPU Fundamentals Relevant to Triton

A modern GPU is composed of several key components:

- Streaming Multiprocessors (SMs): The core compute units of the GPU where actual processing happens. For example, the NVIDIA A100 GPU has 108 SMs.
- Global memory (HBM): A large memory pool (e.g., 40GB or 80GB on the A100) where tensors reside. When you call tensor.cuda() in PyTorch, data moves here.
- Shared memory (SRAM): Fast, on-chip memory shared by all threads within an SM. It has much lower latency than HBM—roughly 15x faster on the A100—but is much smaller (192 KB per SM on the A100).
- Registers: Each thread has its own register file, enabling thousands of threads to run concurrently.
- Warps: Threads are grouped into warps of 32 threads. Each warp is scheduled and executed together on the SM.

The memory bandwidth bottleneck is the primary performance limiter for most deep learning kernels, not raw compute throughput. Writing efficient Triton kernels is largely about minimizing unnecessary data movement between HBM and the compute cores.

Example: PyTorch vs Triton Data Movement

# PyTorch: Each operation fetches from global memory separately
A = torch.randn(1024, device='cuda')
B = torch.randn(1024, device='cuda')
C = A + B  # Each operand fetched from global memory

# Triton: A fused kernel can load both operands in a single pass,
# reducing the number of round-trips to HBM.
# See vector_add_kernel above for an example.

Key Takeaways on GPU Programming with Triton

- The memory bandwidth bottleneck is the primary performance limiter on GPUs, not compute FLOPS.
- Triton enables you to write kernels that minimize data movement and maximize throughput.
- Memory coalescing—accessing contiguous memory locations—is critical for performance and is handled automatically by the Triton compiler.
- High thread counts are encouraged to keep SMs fully utilized and to hide memory latency through warp switching.

Example: Memory Coalescing in Triton

offsets = block_start + tl.arange(0, BLOCK_SIZE)
a = tl.load(A_ptr + offsets, mask=mask)  # Coalesced: threads access contiguous memory

Conclusion

By completing this tutorial, you have gained a foundational understanding of the GPU hardware that Triton targets. You now appreciate how memory hierarchy and bandwidth shape kernel performance, and how Triton empowers you to optimize data movement and computation. This knowledge prepares you to write efficient Triton kernels and sets the stage for hands-on tutorials covering practical implementations like softmax, vector addition, and matrix multiplication.

---

2. First Kernel
---


First Kernel - Vector Addition
- Making the shift to parallel programming
- Coding a vector addition kernel.
- Verifying numerics.
- Benchmarking vs PyTorch.

---

Part 1 - Parallel Programming
---


Writing a Vector Addition Kernel in Triton
Making the Shift to Parallel Programming

Introduction

In this tutorial, we will explore the mental model shift required to move from traditional serial programming to parallel programming on the GPU. Understanding this shift is crucial for effectively leveraging GPU architectures. We will use vector addition as our running example—a simple operation that clearly illustrates the parallel programming model.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Triton Vector Addition Kernel, part 1: Making the Shift to Parallel Programming
https://youtu.be/MEZ7XhzTLEg?si=QYQb9slrDQja8wrp

Mental Model Shift: From Serial to Parallel Programming

Consider a simple problem where we have two integer vectors, A and B, each containing seven elements. Our goal is to compute their element-wise sum and store the result in vector C.

Serial Approach

In a conventional single-threaded programming model (e.g., Python or C++), you would process one element at a time:

# Serial vector addition in Python
A = [1, 2, 3, 4, 5, 6, 7]
B = [7, 6, 5, 4, 3, 2, 1]
C = []
for i in range(len(A)):
C.append(A[i] + B[i])
print(C)  # Output: [8, 8, 8, 8, 8, 8, 8]

Parallel Approach with Triton

Triton introduces a different paradigm by leveraging parallelism on the GPU. Instead of processing elements sequentially, we divide the workload across many independent kernel instances, each responsible for a block of elements. All blocks execute simultaneously on the GPU.

For example, if we choose a block size of 2, each block will process two elements. For our problem of seven elements, we need four blocks (ceil(7/2) = 4), which gives us 8 thread slots—enough to cover all 7 elements, with one extra that we will mask out.

Handling Edge Cases with Masking

A key challenge arises when the total number of elements is not a multiple of the block size. In our example, the last block covers indices 6 and 7, but index 7 is out of bounds. We handle this with a mask: each thread checks whether its assigned index is within the valid range. Threads with out-of-bounds indices are masked out and do not perform any load, compute, or store operations. This prevents memory access violations.

Next Steps: Writing the Triton Kernel

With the mental model established, the next step is to write the actual Triton kernel code. The kernel will:
- Use tl.program_id(0) to get the block ID of the current kernel instance.
- Calculate the starting offset for this block: block_start = pid * BLOCK_SIZE.
- Compute per-element offsets within the block: offsets = block_start + tl.arange(0, BLOCK_SIZE).
- Apply a mask to avoid out-of-bounds memory access: mask = offsets < N.
- Load elements from input vectors A and B.
- Perform element-wise addition.
- Store the results back to the output vector.

Below is a complete example of a Triton vector addition kernel and its host program:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
A_ptr, B_ptr, C_ptr, N,
BLOCK_SIZE: tl.constexpr
):
pid = tl.program_id(0)  # Block ID for this kernel instance
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
grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)
return C

# Example usage
A = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device='cuda')
B = torch.tensor([7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
C = triton_vector_add(A, B)
print(C.cpu().tolist())  # Output: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
```

Key points in the code:
- tl.program_id(0) returns the block ID of the current kernel instance in the grid.
- offsets computes the indices this block will process.
- mask ensures threads do not access out-of-bounds memory.
- The kernel loads, adds, and stores elements in parallel across all blocks.

Conclusion

By completing this tutorial, you have developed a strong conceptual understanding of the shift from serial to parallel programming, which is fundamental for writing efficient GPU kernels with Triton. You now know how to divide work among blocks and how to use masking to handle edge cases safely. This foundation enables you to move confidently into the next stage: implementing and testing Triton kernels for vector addition and beyond.

---

Part 2 - Writing a Kernel
---


Writing a Vector Addition Kernel in Triton

Introduction

In this tutorial, we will walk through writing a complete vector addition kernel in Triton. This tutorial follows Part 1 of this series. Rather than just presenting the finished code, we will build it up step by step—explaining the purpose of each component as we go—so you understand not just what the code does, but why it is written this way.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Triton Vector Addition Kernel, part 2: Coding the Triton Kernel

Step 1: Imports

import triton
import triton.language as tl
```python
import torch

Step 2: Writing the Host Function

The goal of the host function is to give callers an easy, PyTorch-native way to run our kernel. It takes two input tensors and returns their element-wise sum, handling all of the kernel setup internally.

Here is our plan:
1. Allocate an output buffer with the same shape and device as the inputs.
2. Validate that both inputs are CUDA tensors and have the same number of elements.
3. Decide on a block size (how many elements each kernel instance will process).
4. Compute the grid size: how many kernel instances we need to cover all elements.
5. Launch the kernel.

def vector_addition(A, B):
output = torch.empty_like(A)
assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
assert A.numel() == B.numel(), "Input tensors must have the same number of elements"
num_elements = A.numel()
block_size = 128
grid_size = (triton.cdiv(num_elements, block_size),)
kernel_vector[grid_size](A, B, output, num_elements, block_size)
return output

Step 3: Computing the Grid Size with triton.cdiv

The grid size must be large enough that every element is covered by at least one kernel instance. If num_elements is not a multiple of block_size, simple integer division would leave the last few elements unprocessed. Ceiling division solves this:

triton.cdiv(num_elements, block_size)
# equivalent to: (num_elements + block_size - 1) // block_size

This is why we use triton.cdiv rather than the // operator. The last block may have fewer than block_size valid elements—we handle those with masking in the kernel itself.

Step 4: Writing the Triton Kernel

Now let's write the kernel. The kernel receives pointers to the input and output data, the total number of elements, and the block size. It computes which elements it is responsible for, loads them, adds them, and stores the result.

@triton.jit
def kernel_vector(a_ptr, b_ptr, out_ptr, num_elements, block_size: tl.constexpr):
pid = tl.program_id(0)                              # Which block am I?
block_start = pid * block_size                      # Where does my block start?
thread_offsets = block_start + tl.arange(0, block_size)  # My element indices
mask = thread_offsets < num_elements                # Don't go out of bounds

a_vals = tl.load(a_ptr + thread_offsets, mask=mask)
b_vals = tl.load(b_ptr + thread_offsets, mask=mask)
result = a_vals + b_vals
tl.store(out_ptr + thread_offsets, result, mask=mask)

Marking block_size as tl.constexpr tells the compiler that this value is known at compile time, which enables additional optimizations such as loop unrolling.

Step 5: Debugging Tips and Kernel Hygiene

Triton provides tl.device_print for printing values from inside a kernel during development:

# Inside the kernel (remove before production use):
tl.device_print("pid: ", pid)

A few good practices to follow from the start:

- Always apply your mask to both tl.load and tl.store. Forgetting the mask on a store can silently corrupt memory.
- Use tl.constexpr for block sizes and other compile-time constants to enable compiler optimizations.
- Verify numerical correctness (see Part 3) before benchmarking. A fast but incorrect kernel is not useful.
- Keep block sizes as powers of 2 (e.g., 64, 128, 256, 512, 1024). This aligns well with GPU warp sizes and memory access patterns.

Conclusion

By completing this tutorial, you will have written and executed a complete vector addition kernel in Triton, gaining hands-on experience with both the Python and GPU sides of the workflow. You understand how to partition work across blocks, use masking to handle edge cases safely, and leverage Triton's compiler optimizations for efficient execution.

---

Part 3 - Verifying Numerical Fidelity
---


Verifying Numerical Fidelity of a Triton Vector Addition Kernel

Introduction

After implementing a Triton kernel, the next critical step is to verify that it produces correct results. This tutorial demonstrates how to validate that the Triton kernel produces results consistent with PyTorch's native operations. This tutorial follows Part 2 of this series. We will use PyTorch's torch.allclose API to compare outputs and ensure correctness. We will also set up a reproducible testing environment and discuss considerations for numerical tolerance.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Triton Vector Addition Kernel, part 3: Verifying Numerical Accuracy

Step 1: Setting Up the Test Environment

import torch
torch.manual_seed(0)  # Seed CPU and GPU RNGs for reproducibility

Step 2: Creating Test Vectors

vector_size = 8192
A = torch.rand(vector_size, device='cuda')
B = torch.rand_like(A)

Step 3: Computing Reference Result with PyTorch

torch_result = A + B

Step 4: Running the Triton Kernel

triton_result = vector_addition(A, B)

Step 5: Comparing Results with torch.allclose

result_correct = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-4)
print("Numerical fidelity correct:", result_correct)
```

Notes on Tolerances:
- The absolute tolerance (atol) and relative tolerance (rtol) may need adjustment depending on the operation and data.
- For more complex operations like matrix multiplication, rounding errors from chunking and parallelism can cause slight numerical differences.
- For vector addition in float32, results should match very closely. If they do not, the most likely cause is a bug in the kernel logic or pointer arithmetic.

Step 6: Wrapping the Verification in a Function

def verify_numerics():
torch.manual_seed(0)
vector_size = 8192
A = torch.rand(vector_size, device='cuda')
B = torch.rand_like(A)
torch_result = A + B
triton_result = vector_addition(A, B)
result_correct = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-4)
print("Numerical fidelity correct:", result_correct)
return result_correct

Step 7: Running the Verification

if __name__ == "__main__":
verify_numerics()

Troubleshooting Common Errors
- Ensure that input tensors are on CUDA devices (.is_cuda property).
- Use torch.manual_seed to seed both CPU and GPU RNGs for reproducibility.
- When calling .is_cuda, use it as a property, not a method (i.e., no parentheses).
- If results are incorrect, check that your mask is applied consistently to both tl.load and tl.store.

Conclusion

By completing this tutorial, you will have learned how to validate the numerical accuracy of your Triton vector addition kernel, ensuring it matches PyTorch's results within reasonable tolerances. This verification step is essential for building confidence in your custom GPU code and prepares you for the next stage: benchmarking and performance tuning.

---

Part 4 - Benchmarking
---


Triton Vector Addition Kernel: Benchmarking vs PyTorch and tuning

Introduction

Welcome to the final part of our Vector Addition kernel tutorial series. This tutorial follows Part 3 of this series. In previous tutorials, we implemented the kernel and verified its numerical accuracy against PyTorch. With that solid foundation, this tutorial focuses on benchmarking and tuning the kernel to optimize its performance.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Triton Vector Addition Kernel, part 4: Benchmarking vs PyTorch and tuning

Step 1: Setting Up the Benchmarking Framework

To comprehensively evaluate performance, we use a benchmarking function that runs the kernel over a range of vector sizes. The sizes are powers of two, starting from 2^10 (1024) up to 2^28, covering a broad spectrum of tensor sizes.

Step 2: Configuring the Benchmark

providers = [
{"name": "PyTorch", "color": "blue"},
{"name": "Triton", "color": "orange"}
]

Step 3: Running the Benchmark

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
gbps = (A.numel() * A.element_size() * 2) / (end - start) / 1e9
return gbps

Step 4: Interpreting Initial Results

After running the initial benchmarks, you will observe that Triton's performance largely matches PyTorch's across most tensor sizes. There may be a slight performance gap in the mid-range sizes, but at larger tensor sizes, Triton begins to match or exceed PyTorch.

Vector addition is a memory-bandwidth-bound operation: for each element, we perform one addition but must load two values from HBM and store one. The kernel's performance is therefore limited by how fast we can move data, not by arithmetic throughput. A useful reference point is the peak memory bandwidth of your GPU (e.g., ~2 TB/s on an H100 or ~935 GB/s on an A100). Comparing your measured GB/s to this peak tells you how efficiently your kernel is using the available bandwidth.

Step 5: Performance Tuning — Adjusting Block Size

To improve performance, increase the block size from 128 to 1024 and rerun the benchmark. A larger block size allows each kernel instance to process more elements per launch, which can improve GPU utilization and reduce launch overhead.

block_size = 1024
# Rerun benchmark with new block size

Step 6: Performance Tuning — Adjusting Number of Warps

Next, tune the number of warps used by the kernel. The default is 4 warps (128 threads), but increasing to 8 warps allows the GPU to better hide memory latency by switching between warps while one is waiting for data.

num_warps = 8
kernel_vector[(vector_size // block_size,)](A, B, output, vector_size, block_size, num_warps=num_warps)

Step 7: Final Benchmark Results

With the increased block size and number of warps, the benchmark shows that Triton kernel performance matches or slightly exceeds PyTorch's performance across the tested range.

Conclusion

In this tutorial, you learned how to benchmark and tune a custom Triton vector addition kernel, comparing its performance to PyTorch and optimizing key parameters for maximum throughput. By setting up a flexible benchmarking framework, interpreting performance results, and tuning block size and warp count, you gained practical skills for developing high-performance GPU kernels.

---

3. Softmax
---


Softmax
- Basic softmax in PyTorch
- Online softmax in PyTorch
- Softmax in Triton

---

Part 1 - Basic Softmax and Online Softmax in PyTorch
---


Softmax in PyTorch: From Naive to Online

In this tutorial, we will implement the softmax function in PyTorch, starting with a naive implementation and then improving it with the online softmax algorithm. Understanding both approaches is essential before implementing softmax in Triton, and the online softmax algorithm is particularly important because it is the key algorithmic insight behind FlashAttention.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Intro to Triton: Coding Softmax in PyTorch

1. Environment Setup

pip install torch triton

2. Creating a Sample Tensor

```python
import torch
import torch.nn.functional as F

sample = torch.tensor(
[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
dtype=torch.float32,
device='cuda'
)

3. Reference: PyTorch Softmax

softmax_ref = F.softmax(sample, dim=1)
print("PyTorch Softmax Output:\n", softmax_ref)
```

4. Naive Softmax: Implementation and Inefficiencies

Let's implement our own numerically stable softmax function in PyTorch:

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
# Step 1: Subtract the row-wise max for numerical stability
x_max = x.max(dim=1, keepdim=True).values
x_safe = x - x_max
# Step 2: Exponentiate
numerator = torch.exp(x_safe)
# Step 3: Sum across rows for denominator
denominator = numerator.sum(dim=1, keepdim=True)
# Step 4: Divide to get softmax
return numerator / denominator

softmax_naive = naive_softmax(sample)
assert torch.allclose(softmax_naive, softmax_ref, atol=1e-6), "Mismatch!"

This implementation requires multiple passes over each row of data:
1. Find the maximum value (for numerical stability) — 1st pass.
2. Subtract the max and exponentiate — 2nd pass.
3. Sum the exponentiated values — 3rd pass.
4. Divide to get the final softmax — 4th pass.

Each pass reads the entire row from HBM, which is costly for large tensors.

5. Online Softmax: Reducing Memory Passes

Online softmax reduces the number of passes to just two by maintaining running statistics—the current row maximum and the running sum of exponentials—as it processes each element. When the running maximum is updated, the accumulated sum is rescaled to remain consistent.

def online_softmax(x: torch.Tensor) -> torch.Tensor:
n_rows, n_cols = x.shape
output = torch.zeros_like(x)
for r in range(n_rows):
row_max = float('-inf')
normalizer = 0.0
# First pass: compute row_max and normalizer together
for c in range(n_cols):
val = x[r, c].item()
prev_row_max = row_max
row_max = max(row_max, val)
if row_max != prev_row_max:
# Rescale the running sum for the new maximum
normalizer = normalizer * torch.exp(torch.tensor(prev_row_max - row_max))
normalizer += torch.exp(torch.tensor(val - row_max))
# Second pass: compute final softmax values
for c in range(n_cols):
output[r, c] = torch.exp(x[r, c] - row_max) / normalizer
return output

# Verify correctness
softmax_online = online_softmax(sample.cpu()).to(sample.device)
assert torch.allclose(softmax_online, softmax_ref.cpu(), atol=1e-6), "Mismatch!"
print("Online softmax matches PyTorch softmax.")

6. Why Online Softmax Matters: The Connection to FlashAttention

Online softmax is not just a micro-optimization—it is the key algorithmic step that makes FlashAttention possible. FlashAttention computes the attention mechanism in tiles, processing small blocks of the query-key-value matrices at a time rather than materializing the full attention matrix. To do this correctly, it needs to compute softmax over a sequence of partial results, which is exactly what the online softmax algorithm enables. Without it, tiled attention computation would require a separate pass to find the global maximum before computing the softmax, negating the memory savings.

7. Performance Comparison

In practice, the online softmax is significantly faster than the naive version for large tensors, because it reduces the number of passes over each row from four to two. The performance gain grows with the size of the input.

Conclusion

In this tutorial, you implemented both naive and online softmax in PyTorch and verified their correctness. You also learned why online softmax is a foundational algorithm for high-performance deep learning, particularly as the basis for FlashAttention. With this understanding, you are ready to implement softmax as a Triton kernel in the next tutorial.

---

Part 2 - Softmax in Triton
---


Softmax in Triton: Writing and Benchmarking a Custom Softmax Kernel

Introduction

In this tutorial, we'll walk through implementing a custom softmax kernel in Triton, including both the kernel itself and the host program that launches it. We'll cover memory management, parallelization, pointer arithmetic, and benchmark our implementation against PyTorch's native softmax and a naive implementation.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Coding a Triton Kernel for Softmax (fwd pass) Computation

1. Understanding Triton Kernels and Host Programs

When working with Triton, you typically write two components:
- The Kernel: The function that runs on the GPU, processing data in parallel.
- The Host Program: The Python code that sets up meta-information (like block size and grid shape), allocates memory, and launches the kernel.

2. Setting Up the Host Program

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

3. Key Implementation Details

Memory and Pointer Arithmetic:
Tensors in memory are 1D arrays; strides are used to move between rows. The kernel receives pointers to the input and output data, and uses pointer arithmetic to access the correct row and column. Masking ensures we only read/write valid data when the number of columns is not a power of two.

Parallelization:
We launch one kernel instance per row (grid = (n_rows,)). Each kernel instance processes a full row, with the BLOCK_SIZE set to the next power of two greater than or equal to the number of columns.

Softmax Computation:
For numerical stability, we subtract the row max before exponentiating. We sum the exponentiated values to get the denominator, and write the result back to the output buffer using the same mask.

4. Debugging and Common Pitfalls

Pointer arithmetic is error-prone. A common mistake is to incorrectly calculate the output pointer, leading to overwriting the same row multiple times. Always ensure that:
- The output pointer is offset by both the row index (via output_stride) and the column offset.
- Masking is applied consistently when both reading and writing.

If you see duplicated or inverted rows in your output, double-check your pointer calculations.

5. Numerical Verification and Benchmarking

Before benchmarking, always verify correctness:

import torch.nn.functional as F

x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
y_triton = triton_softmax(x)
y_torch = F.softmax(x, dim=1)
assert torch.allclose(y_triton, y_torch, atol=1e-6), "Numerical mismatch!"
print("Triton softmax matches PyTorch softmax.")
```

Then benchmark across a range of tensor sizes:

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

6. Visualizing Results

Plotting the throughput (e.g., GB/s) for each implementation across tensor sizes can help you see where Triton shines. Typically, you'll see:
- Triton: Outperforms naive and matches or exceeds PyTorch for large tensors.
- PyTorch: Fast for small tensors, but may decline as size increases.
- Naive: Significantly slower than both.

7. Conclusion

In this tutorial, you learned how to write and launch a custom softmax kernel in Triton, verify its numerical correctness, and benchmark its performance. Writing custom Triton kernels gives you fine-grained control over GPU computation and can yield significant performance gains, especially for large-scale deep learning workloads.

---

WIP -- 4. Tiled Matrix Multiplication
---


Tiled Matrix Multiplication in Triton
- Part 1: Flash Attention 2: Overview
- Part 2: Flash Attention 2: Forward Kernel
- Part 3: Flash Attention 2: Backward Kernel
- Part 4: Benchmarking / Tuning

---

WIP -- 5. Advanced Topics
---


Advanced Topics
- TMA
- Warp Specialization
- Coop Matmul in Triton

Note: The Advanced Topics section should be positioned before the Flash Attention tutorials, as it introduces the advanced GPU programming concepts (TMA, warp specialization) that Flash Attention relies on.

---

Additional Resources
---


Additional Resources

Here are some other excellent resources for strengthening your Triton skills.

Triton-Viz

Triton-Viz is a visualization and profiling toolkit designed to make GPU programming with Triton more intuitive and accessible. It provides real-time visualizations of tensor operations and memory usage, helping developers debug, analyze performance, and better understand how their Triton code interacts with accelerator hardware. Notably, Triton-Viz can be used without a GPU, allowing users to explore and optimize Triton programs on any system.

GitHub - Deep-Learning-Profiling-Tools/triton-viz

Triton Puzzles

Triton Puzzles is an interactive, hands-on resource designed to teach Triton programming from first principles. Through a series of progressively challenging puzzles, you'll learn key concepts such as memory loading, storage, and efficient GPU programming—starting with simple tasks and advancing to real-world algorithms like Flash Attention and quantized neural networks. The puzzles run in a Triton interpreter, so you can experiment and learn without needing a GPU.

GitHub - srush/Triton-Puzzles: Puzzles for learning Triton

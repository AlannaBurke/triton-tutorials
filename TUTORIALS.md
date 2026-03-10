Triton Tutorial Series — Outline

Initial Project Docs & Resources:
- Triton Tutorials 2025
- PyTorch Docs + Partner Sync Meeting Notes
- Triton Module
- https://www.internalfb.com/wiki/Triton_Tutorials/
- SOTA Deep Learning Tutorials - YouTube
- Triton documentation
- https://github.com/gpu-mode/triton-tutorials/

Meeting Agenda: Triton Series Agenda H1 2026

Basics
- Intro to Triton
- What is Triton?
- What does it provide over PyTorch Eager?
- Why is it important relative to PyTorch?
- Getting Started with Triton
- Triton vs. CUDA
- Introduction to Triton and GPU Fundamentals for Kernel Programming
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

2. What Does Triton Provide Over PyTorch Eager?

PyTorch eager mode is an execution setting in which operations execute immediately and return results as soon as they're invoked in Python code, achieving part of the performance gain of the fully compiled mode. While PyTorch eager mode is excellent for rapid prototyping and flexibility, it relies on pre-built GPU kernels, e.g., cuBLAS, cuDNN library implementations, for most operations. This means:
- Limited Customization: If you need a custom operation or want to optimize a specific computation, you are limited by what PyTorch provides out of the box.
- Performance Ceiling: PyTorch's built-in kernels are highly optimized for general use, but may not be optimal for every specialized workload or new research idea.

Triton provides:
- Custom Kernel Development: You can write your own GPU kernels for any operation, enabling fine-grained control over memory access, parallelism, and computation.
- Performance Tuning: Triton lets you optimize for your specific use case, often achieving better performance than generic PyTorch kernels, especially for novel or non-standard operations.
- Seamless Integration: Triton kernels can be called directly from Python and interoperate with PyTorch tensors, making it easy to integrate into existing PyTorch workflows.
- Higher-Level Abstraction: Compared to CUDA, Triton's API is much simpler and more concise, reducing the learning curve and development time for custom GPU programming.

Triton and Eager, compared

Feature         | Eager                    | Triton
----------------|--------------------------|---------------------------
Custom GPU Kernels | Limited               | Full control
Performance Tuning | Limited to built-ins  | Fine-grained, user-optimized
API Style       | Pythonic, high-level     | Pythonic, kernel-focused
Integration     | Native                   | Seamless with PyTorch
Learning Curve  | Low                      | Moderate (but easier than CUDA)

3. How Triton Adds to PyTorch?

Triton is important in the PyTorch ecosystem for several reasons:
- Unlocks Custom GPU Programming for Researchers: Many cutting-edge research ideas require custom GPU operations that are not available in standard libraries. Triton makes it feasible for researchers to implement and experiment with these ideas without deep expertise in CUDA or GPU hardware.
- Bridges the Gap Between Flexibility and Performance: PyTorch eager mode is flexible but can be limited in performance for custom operations. Triton allows users to maintain flexibility while achieving near hand-tuned performance for their specific workloads.
- Accelerates Innovation: By lowering the barrier to writing high-performance GPU code, Triton enables faster prototyping and deployment of new algorithms, especially in areas like deep learning, scientific computing, and large-scale data processing.
- Complementary to PyTorch: Triton is not a replacement for PyTorch, but a powerful extension. It allows users to keep the productivity and ecosystem of PyTorch while extending its capabilities to custom, high-performance GPU operations.
- Works hand-in-hand with TorchInductor:
TorchInductor is PyTorch's compiler backend that automatically lowers and optimizes PyTorch programs into efficient kernels, often using Triton as a code generator for GPU. Together, Inductor + Triton provide an end-to-end path from high-level PyTorch code to optimized GPU execution with minimal manual kernel writing.

4. How does Triton work?

A typical Triton program consists of device kernels and a host program that calls the kernels.

A Triton kernel has this form:

@triton.jit
def kernel_name(...):
# ...

And a host program calls the kernel as follows:

result = kernel_name[grid](...)

where grid is a tuple defining the number of thread blocks in each dimension:

An 1D grid:  grid = (X,)
A 2D grid:   grid = (X, Y)
A 3D grid:   grid = (X, Y, Z)

For Nvidia GPUs, when a kernel is called, the CUDA runtime launches a grid of threads that execute the kernel code.

The @triton.jit decorator is used to define a Triton kernel for just-in-time (JIT) compiling. This enables developers to write concise, high-performance code for GPU kernels while leveraging the familiar Python syntax.

The following diagram illustrates the compilation stages of a Triton kernel:

Here's a breakdown of how it works:

1. Kernel declaration and compilation:
- A kernel is a Python function that is decorated with the @triton.jit decorator, which tells the Triton compiler to compile that kernel.
- This kernel will be executed on the GPU when it is called.

2. Intermediate representation (IR) generation:
- The Triton compiler analyzes the kernel's abstract syntax tree (AST) to generate a Triton intermediate representation (Triton-IR).
- Triton-IR is a machine-independent, unoptimized representation of the kernel, capturing the high-level structure of the computation.

3. Optimization:
- The Triton compiler performs optimizations on the generated Triton-IR.
- The optimized Triton-IR is then lowered (transformed) to Triton-GPU IR (Triton-TTGIR) and subsequently to low-level virtual machine IR (LLVM-IR).

4. GPU code generation:
- The LLVM-IR is used to generate CUDA code or AMDGCN code.
- For NVIDIA GPUs, from LLVM-IR the Triton compiler generates Parallel Thread Execution (PTX) code which is then JIT-compiled by Nvidia ptxas into a CUDA binary (CUBIN). Nvidia ptxas is an assembler tool used in the NVIDIA CUDA environment to convert Parallel Thread eXecution (PTX) assembly code into executable binary code for NVIDIA GPUs.
- For AMD GPUs, from LLVM-IR the Triton compiler generates AMDGCN code which is then compiled by AMD JIT into AMD hsaco binary.

5. Execution on GPU:
- The JIT-compiled GPU code is executed in parallel on the GPU, leveraging its processing power for accelerated computations.

Try It

Get your notebook ready

We'll be testing some code in this tutorial, so make sure you have a Jupyter Notebook such as Google Colab.

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
n_elements: tl.constexpr,  # Number of elements in the vectors
BLOCK_SIZE: tl.constexpr   # Number of elements processed per block
):
pid = tl.program_id(axis=0)  # Unique program/thread block ID
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
# Define grid: number of blocks needed
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

Tips for Jupyter Notebook Users
- Run each code block in a separate cell for clarity.
- If you encounter errors about CUDA or Triton, make sure your notebook is running on a GPU-enabled environment (e.g., Google Colab with GPU runtime).
- You can modify the input data or kernel logic to experiment further.

Conclusion

By completing this tutorial, you should now understand the fundamentals of Triton, including its purpose, advantages over PyTorch eager mode, and its significance in the PyTorch ecosystem. You will be able to describe how Triton works, how it enables custom GPU kernel development, and why it is a valuable tool for researchers and engineers seeking high-performance, flexible GPU programming. With this foundation, you are ready to start experimenting with Triton and explore its potential for accelerating your own deep learning and scientific computing projects.

---

Getting Started with Triton
---


Getting started with Triton

Triton is a Python-based language and compiler that transforms high-level Python code into GPU machine instructions (i.e., Parallel Thread Execution (PTX) code). It enables programmers with little or no experience of GPU hardware and GPU-specific programming languages, such as CUDA, to write very efficient parallel programs.

A typical Triton program consists of Triton kernels (functions) and a host program that calls the kernels. Triton kernels are executed in parallel by many threads on the GPU.

We'll be testing some code in this tutorial, so make sure you have a Jupyter Notebook such as Google Colab.

Vector addition in GPU

The only way to learn Triton is by writing programs in it and testing them. The structure of a Triton program consists of two parts: a host part and a device part:
- The device part consists of Triton kernels that look like Python functions that are compiled and executed on the GPU.
- The host part handles tasks like loading data, launching kernels, and collecting results from the GPU.

For example, the following Triton kernel executes vector addition on the GPU:

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

And the following code does the following vector addition:

BLOCK_SIZE = 1
data = [1, 2, 3, 4, 5]
x = torch.tensor(data, device='cuda')
output = torch.empty_like(x)
N = len(data)
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
add_kernel[grid](x, 10, output, N, BLOCK_SIZE)

print(f"x = {x}")
```
print(f"x + 10 = {output}")

The output looks like this:

x = tensor([1, 2, 3, 4, 5], device='cuda:0')
x + 10 = tensor([11, 12, 13, 14, 15], device='cuda:0')

Testing Triton tutorials

Before testing the Triton tutorials, download them:
- Open the tutorials page.
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

After downloading the Triton tutorials, test them in your notebook. For example, to test 01-vector-add.py, do the following:
- Open an existing notebook or create a new notebook.
- If it is not connected to a server, connect it to an on-demand GPU devserver.
- Make a copy of ~/tutorials_python/01-vector-add.py and paste it to a cell in your notebook.
- Run 01-vector-add.py. If you get this error (see the caveats):

Make this change in 01-vector-add.py:

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Re-run 01-vector-add.py.

Conclusion

By completing this tutorial, you will have taken your first practical steps in GPU programming with Triton. You will understand how Triton kernels work, how to structure and execute a basic program, and how to test your code interactively in a Jupyter Notebook. With these foundational skills and access to further examples, you are well-prepared to explore more advanced GPU programming concepts and leverage Triton for high-performance computing tasks in your own projects.

---

Triton vs. CUDA
---


Triton vs. CUDA

Triton is an open-source, Python-based programming language and compiler designed for writing high-performance custom Deep Neural Networks (DNNs) compute kernels, particularly targeting NVIDIA GPUs. Compute Device Unified Architecture (CUDA), on the other hand, is a parallel computing platform and programming model developed by NVIDIA, which allows developers to leverage the power of NVIDIA GPUs for general-purpose computing. Triton is often preferred over CUDA because it simplifies GPU programming for machine learning, particularly for Python developers, while still allowing for deep optimization.

This guide compares the key differences between Triton and CUDA.

Key differences between Triton and CUDA

Feature              | Triton                                      | CUDA
---------------------|---------------------------------------------|------------------------------------------
Programming language | Python language                             | C or C++
Target audience      | Ideal for deep learning researchers and developers who want to write custom kernels for deep learning models without needing extensive CUDA expertise. | A more general-purpose platform for GPU programming, used in various fields, including deep learning, scientific computing, and game development.
Performance          | While Triton aims to simplify GPU programming, it may come with a slight performance penalty compared to highly optimized CUDA kernels. | Offers the potential for very high performance, especially when developers are deeply familiar with GPU architecture.
Programming model    | Higher-level DSL focused on deep learning. Uses a single-level decomposition (blocks). | Low-level, general-purpose. Uses a two-level decomposition (threads and blocks).
Learning curve       | Lower: Employs a higher-level, more Pythonic approach. | Higher: Requires a deeper understanding of GPU architecture.

Compiler optimizations in Triton vs. CUDA

Triton automates many compiler optimizations commonly done manually in CUDA, including memory coalescing, shared memory allocation, and instruction scheduling.

Feature                                    | Triton    | CUDA
-------------------------------------------|-----------|--------
Memory coalescing                          | Automatic | Manual
Shared memory management                   | Automatic | Manual
Scheduling (within SMs)                    | Automatic | Manual
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

By completing this tutorial, you will have a clear understanding of how Triton and CUDA differ in their approach to GPU programming, especially for deep learning and scientific computing.

---

Introduction to Triton and GPU Fundamentals for Kernel Programming
---


Introduction

This tutorial provides an updated introduction to Triton, a powerful language and compiler for programming GPU kernels. Triton offers direct control over GPU memory hierarchies, enabling developers to write highly performant kernels by bridging the gap between GPU hardware and algorithm design.

What is Triton?

Triton is both a programming language designed specifically for writing GPU kernels and a compiler that translates Triton code into efficient GPU machine instructions. The main advantage of Triton is that it gives you direct control over two key types of GPU memory: SRAM (shared memory or on-chip memory), which is very fast and low-latency and is shared among threads within a streaming multiprocessor (SM), and HBM (high bandwidth memory or global GPU memory), which is larger but slower and accessible by all SMs. This control allows you to optimize data movement and computation, which is essential for achieving high performance on GPUs.

Triton is both a programming language and a compiler designed specifically for writing high-performance GPU kernels. It provides a higher-level, Pythonic interface that abstracts away many of the low-level details of GPU programming (such as explicit thread/block management and manual shared memory allocation) while still allowing users to optimize data movement and computation. Triton kernels operate on global GPU memory (HBM), and the language/compiler are designed to generate efficient memory access patterns. While Triton does not expose explicit control over shared memory (SRAM) in the same way as CUDA, it enables users to write custom GPU kernels that can approach the performance of hand-written CUDA code, making it easier to optimize for modern deep learning workloads.

GPU Fundamentals Relevant to Triton

A modern GPU is composed of several key components. Streaming Multiprocessors (SMs) are the core compute units of the GPU where actual processing happens. For example, the NVIDIA A100 GPU has 108 SMs. Global memory (HBM) is a large memory pool (such as 40GB or 80GB) where data and models reside when moved to the GPU, for example via PyTorch's .cuda() method. Shared memory (SRAM) is fast, on-chip memory shared by all threads within an SM, and is used for efficient data sharing and computation. Triton kernels execute on the SMs, giving you direct control over shared memory and compute cores, whereas PyTorch primarily operates on global memory.

Key Takeaways on GPU Programming with Triton

Triton operates on-chip, controlling SMs and shared memory directly, while PyTorch primarily operates on global memory. The memory bandwidth bottleneck is the primary performance limiter on GPUs, not compute FLOPS. Triton enables intelligent data movement and kernel design to hide latency and maximize throughput.

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

In this tutorial, we will explore how to write a vector addition kernel using Triton, a language and compiler designed for writing efficient GPU code. The tutorial is divided into two parts. First, we will cover the mental model shift required to move from traditional serial programming to parallel programming. Understanding this shift is crucial for effectively leveraging GPU architectures. Then, we will dive into the actual coding of the Triton kernel, illustrating how to implement vector addition in a parallelized manner.

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

Triton introduces a different paradigm by leveraging parallelism on the GPU. Instead of processing elements sequentially, we divide the workload across many threads organized into blocks and grids. Each thread operates independently on a subset of the data.

For example, if we choose a block size of 2, each block will have two threads. For our problem of seven elements, we create a grid of four blocks (since 4 blocks × 2 threads = 8 threads, which covers all 7 elements plus one extra). Each thread in a block processes one element of the vectors simultaneously.

Handling Edge Cases with Masking

A key challenge arises when the total number of elements is not a multiple of the block size. In our example, the last block has two threads, but only one element remains to be processed. The second thread in the last block has no valid data to work on.

To handle this safely, we use masking. Each thread checks if its assigned index is within the valid range of elements. Threads with indices outside the valid range become inert and do not perform any load, compute, or store operations. Proper masking prevents out-of-bounds memory access, which would otherwise cause program crashes.

Next Steps: Writing the Triton Kernel

With the mental model established, the next step is to write the actual Triton kernel code that implements this parallel vector addition. The kernel will:
- Use the program ID to identify the current block.
- Calculate the starting offset for the block based on the block size.
- Compute thread offsets within the block.
- Apply masking to avoid out-of-bounds memory access.
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
- tl.program_id(0) gets a unique ID in the grid.
- offsets computes the indices this block's threads will process.
- mask ensures threads do not access out-of-bounds memory.
- The kernel loads, adds, and stores elements in parallel.

Conclusion

By completing this tutorial, you have developed a strong conceptual understanding of the shift from serial to parallel programming, which is fundamental for writing efficient GPU kernels with Triton.

---

Part 2 - Writing a Kernel
---


Writing a Vector Addition Kernel in Triton

Introduction

In this tutorial, we will walk through the process of writing a vector addition kernel using Triton, a language and compiler designed for writing efficient GPU code. This tutorial follows Part One of this Triton series. We will start by creating a host function that prepares the inputs and output buffers, sets up the execution grid, and calls the Triton kernel. Then, we will implement the kernel itself, which performs element-wise addition of two input vectors on the GPU. Finally, we will discuss some optimization considerations and debugging tips. This tutorial aims to provide a clear, step-by-step guide from conceptual understanding to practical implementation.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Triton Vector Addition Kernel, part 2: Coding the Triton Kernel

Step 1: Importing Libraries and Setting Up

We begin by importing the necessary libraries. Triton is imported along with its language module, and PyTorch is imported for comparison purposes, both for performance and correctness.

import triton
import triton.language as tl
```python
import torch

Step 2: Writing the Host Function

The host function serves as the interface for users to run the kernel. It takes two input vectors, A and B, and returns their element-wise sum as a PyTorch tensor.

Inside the host program:
- We create an output buffer with the same shape and device as A.
- We assert that both A and B are CUDA tensors and have the same size.
- We determine the number of elements to process.
- We define a block size to chunk the workload.
- We compute the grid size using a ceiling division function to ensure all elements are covered.
- We call the Triton kernel with the prepared parameters.

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

Step 3: Implementing the Ceiling Division Function

The ceiling division function ensures that the grid size covers all elements, even if the last block has fewer elements than the block size.

def ceil_div(x, y):
return (x + y - 1) // y

Step 4: Writing the Triton Kernel

The kernel performs the core computation on the GPU. Key points include:
- Retrieving the program ID to identify the current block.
- Calculating the starting offset for this block based on the block size and program ID.
- Computing thread offsets within the block.
- Applying a mask to prevent out-of-bounds memory access.
- Loading elements from input vectors A and B using the computed offsets and mask.
- Performing element-wise addition.
- Storing the result back to the output buffer with masking.

Marking the number of elements and block size as constant expressions allows the compiler to optimize the kernel, which can lead to significant performance improvements.

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

Step 5: Debugging Tips

Triton provides a simple device print function for debugging purposes. While it does not support formatted strings, it can be used to print values such as the current program ID to the console during kernel execution.

# Example usage inside kernel (not used here)
# tl.print("Program ID:", pid)

Conclusion

By completing this tutorial, you will have written and executed a complete vector addition kernel in Triton, gaining hands-on experience with both the Python and GPU sides of the workflow.

---

Part 3 - Verifying Numerical Fidelity
---


Verifying Numerical Fidelity of a Triton Vector Addition Kernel

Introduction

After implementing a Triton kernel for vector addition, the next critical step is to verify its numerical fidelity. This tutorial demonstrates how to validate that the Triton kernel produces results consistent with PyTorch's native operations. This tutorial follows Part Two of this Triton series. We will use PyTorch's torch.allclose API to compare outputs and ensure correctness. Additionally, we will set up a reproducible testing environment and discuss considerations for numerical tolerance.

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
- Use torch.manual_seed to seed both CPU and GPU RNGs.
- When calling .is_cuda, use it as a property, not a method (i.e., no parentheses).

Conclusion

By completing this tutorial, you will have learned how to validate the numerical accuracy of your Triton vector addition kernel, ensuring it matches PyTorch's results within reasonable tolerances.

---

Part 4 - Benchmarking
---


Triton Vector Addition Kernel: Benchmarking vs PyTorch and tuning

Introduction

Welcome to the final part of our Vector Addition kernel tutorial series. This tutorial follows Part Three of the Triton series. In previous tutorials, we implemented the kernel and verified its numerical accuracy against PyTorch. With that solid foundation, this tutorial focuses on benchmarking and tuning the kernel to optimize its performance.

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

After running the initial benchmarks, you will observe that Triton's performance largely matches PyTorch's across most tensor sizes. There may be a slight performance lag in the mid-range sizes, but at larger tensor sizes, Triton begins to outperform PyTorch. These results provide a baseline and highlight areas where tuning can improve performance.

Step 5: Performance Tuning — Adjusting Block Size

To improve performance, increase the block size from 128 to 1024 and rerun the benchmark.

Step 6: Performance Tuning — Adjusting Number of Warps

Next, tune the number of warps used by the kernel. The default is 4 warps (128 threads), but increasing to 8 warps allows better utilization of GPU resources.

Step 7: Final Benchmark Results

With the increased number of warps, the benchmark shows that Triton kernel performance matches or slightly exceeds PyTorch's performance across the tested range.

Example: Triton Vector Addition Kernel Implementation

@triton.jit
def kernel_vector(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr):
pid = tl.program_id(0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < N
a = tl.load(A_ptr + offsets, mask=mask)
b = tl.load(B_ptr + offsets, mask=mask)
tl.store(output_ptr + offsets, a + b, mask=mask)

Conclusion

In this tutorial, you learned how to benchmark and tune a custom Triton vector addition kernel, comparing its performance to PyTorch and optimizing key parameters for maximum throughput.

---

3. Softmax
---


Softmax
- Basic softmax in PyTorch
- Online softmax in PyTorch
- Softmax in Triton

---

Part 1 - Basic Softmax
---


Basic Softmax: Implementing Softmax in PyTorch and Triton

In this tutorial, we will walk through the process of implementing the softmax function—first in PyTorch, and then preparing for a Triton implementation.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Intro to Triton: Coding Softmax in PyTorch

1. Environment Setup

pip install torch triton

2. Creating a Sample Tensor

sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')

3. Reference: PyTorch Softmax

```python
import torch.nn.functional as F
softmax_ref = F.softmax(sample, dim=1)

4. Implementing a Numerically Stable Softmax in PyTorch

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
x_max = x.max(dim=1, keepdim=True).values
x_safe = x - x_max
numerator = torch.exp(x_safe)
denominator = numerator.sum(dim=1, keepdim=True)
softmax_out = numerator / denominator
return softmax_out

5. Verifying Correctness

assert torch.allclose(softmax_naive, softmax_ref, atol=1e-6), "Mismatch with PyTorch softmax!"

6. Preparing for Triton Implementation

With a working, numerically stable softmax in PyTorch, you are ready to port this logic to Triton.

Conclusion

In this tutorial, you set up a PyTorch and Triton development environment, implemented a numerically stable softmax function in PyTorch, and verified its correctness.

---

Part 2 - Online Softmax
---


Online Softmax: Algorithmic Improvements for Performance

In this tutorial, we explore the online softmax algorithm—a more efficient approach to computing the softmax function, widely used in deep learning.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Coding Online Softmax in PyTorch - a faster Softmax via reduced memory access

1. Background: Why Online Softmax?

The softmax function is a key operation in many machine learning models, especially for converting logits into probabilities. The naive softmax implementation typically requires multiple passes over the data, which can be inefficient for large-scale or streaming applications.

Online softmax is an algorithmic improvement that reduces the number of passes over each row of data, leading to significant performance gains. Originally motivated by the need to process streaming data efficiently, online softmax is now also used in high-performance implementations such as FlashAttention.

2. Naive Softmax: A Review

For each row in your input tensor, the naive softmax typically performs:
1. Find the maximum value (for numerical stability) — 1st pass.
2. Subtract the max from each element — 2nd pass.
3. Exponentiate each element — 3rd pass.
4. Sum the exponentiated values (denominator) — 4th pass.
5. Divide each exponentiated value by the sum (final softmax) — 5th pass.
Total: 5 passes over each row.

3. Online Softmax: The Algorithm

Online softmax reduces the number of passes to just two, by maintaining running statistics as it processes each row.

4. Step-by-Step Implementation in PyTorch

def online_softmax(x: torch.Tensor) -> torch.Tensor:
n_rows, n_cols = x.shape
output = torch.zeros_like(x)
for r in range(n_rows):
row_max = float('-inf')
normalizer = 0.0
for c in range(n_cols):
val = x[r, c].item()
prev_row_max = row_max
row_max = max(row_max, val)
if row_max != prev_row_max:
normalizer = normalizer * torch.exp(torch.tensor(prev_row_max - row_max))
normalizer += torch.exp(torch.tensor(val - row_max))
for c in range(n_cols):
output[r, c] = torch.exp(x[r, c] - row_max) / normalizer
return output

5. Results and Discussion

Online softmax produces results identical to both the naive implementation and PyTorch's built-in softmax. In practice, the online softmax was nearly 4x faster than the naive version, thanks to the reduced number of passes over each row.

6. Conclusion

Online softmax is a powerful technique for anyone looking to optimize their deep learning workloads. In modern deep learning, especially in attention mechanisms like those used in Transformers, online softmax enables efficient computation by processing data in a streaming fashion, reducing memory overhead and improving numerical stability.

---

Part 3 - Softmax in Triton
---


Softmax in Triton: Writing and Benchmarking a Custom Softmax Kernel in Triton

Introduction

In this tutorial, we'll walk through implementing a custom softmax kernel in Triton, including both the kernel itself and the host program that launches it. We'll cover memory management, parallelization, pointer arithmetic, and benchmark our implementation against PyTorch's native softmax and a naive implementation.

This tutorial is derived from this video by SOTA Deep Learning Tutorials:
Coding a Triton Kernel for Softmax (fwd pass) Computation

1. Understanding Triton Kernels and Host Programs

When working with Triton, you typically write two components:
- The Kernel: The function that runs on the GPU, processing data in parallel.
- The Host Program: The Python code that sets up meta-information, allocates memory, and launches the kernel.

2. Setting Up the Host Program

3. Key Implementation Details

4. Debugging and Common Pitfalls

Pointer arithmetic is error-prone. A common mistake is to incorrectly calculate the output pointer, leading to overwriting the same row multiple times.

5. Benchmarking the Triton Kernel

To evaluate performance, benchmark your Triton kernel against both PyTorch's native softmax and a naive implementation across a range of tensor sizes.

6. Visualizing Results

7. Conclusion

Writing custom Triton kernels gives you fine-grained control over GPU computation and can yield significant performance gains, especially for large-scale deep learning workloads.

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

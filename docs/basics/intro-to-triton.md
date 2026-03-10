# Introduction to Triton

## 1. What is Triton?

Triton is an open-source, Python-based programming language and compiler for writing highly efficient custom GPU kernels. It transforms high-level Python code into GPU machine instructions, enabling developers and researchers—even those with little or no experience in GPU hardware or specialized languages like CUDA—to write expressive, parallel programs for accelerators such as NVIDIA GPUs. Triton's Pythonic syntax is familiar to users of scientific computing libraries like NumPy and PyTorch, making it accessible for a wide range of scientific and engineering applications.

Key characteristics of Triton:
- Pythonic API: Write GPU code in a style similar to NumPy or PyTorch, making it accessible to Python developers.
- Custom GPU Kernels: Allows you to write your own GPU operations (kernels) for tasks where built-in libraries may not be optimal.
- Automatic Compilation: Triton compiles your Python-like code into highly optimized GPU code, abstracting away much of the complexity of CUDA or other low-level languages.
- Focus on Performance: Designed to achieve performance close to hand-written CUDA, but with much less boilerplate and complexity.

## 2. What Does Triton Provide Over PyTorch Eager?

PyTorch eager mode is an execution setting in which operations execute immediately and return results as soon as they're invoked in Python code, achieving part of the performance gain of the fully compiled mode. While PyTorch eager mode is excellent for rapid prototyping and flexibility, it relies on pre-built GPU kernels, e.g., cuBLAS, cuDNN library implementations, for most operations. This means:
- Limited Customization: If you need a custom operation or want to optimize a specific computation, you are limited by what PyTorch provides out of the box.
- Performance Ceiling: PyTorch's built-in kernels are highly optimized for general use, but may not be optimal for every specialized workload or new research idea.

Triton provides:
- Custom Kernel Development: You can write your own GPU kernels for any operation, enabling fine-grained control over memory access, parallelism, and computation.
- Performance Tuning: Triton lets you optimize for your specific use case, often achieving better performance than generic PyTorch kernels, especially for novel or non-standard operations.
- Seamless Integration: Triton kernels can be called directly from Python and interoperate with PyTorch tensors, making it easy to integrate into existing PyTorch workflows.
- Higher-Level Abstraction: Compared to CUDA, Triton's API is much simpler and more concise, reducing the learning curve and development time for custom GPU programming.

### Triton and Eager, compared

| Feature | Eager | Triton |
|---------|-------|--------|
| Custom GPU Kernels | Limited | Full control |
| Performance Tuning | Limited to built-ins | Fine-grained, user-optimized |
| API Style | Pythonic, high-level | Pythonic, kernel-focused |
| Integration | Native | Seamless with PyTorch |
| Learning Curve | Low | Moderate (but easier than CUDA) |

## 3. How Triton Adds to PyTorch?

Triton is important in the PyTorch ecosystem for several reasons:
- Unlocks Custom GPU Programming for Researchers: Many cutting-edge research ideas require custom GPU operations that are not available in standard libraries. Triton makes it feasible for researchers to implement and experiment with these ideas without deep expertise in CUDA or GPU hardware.
- Bridges the Gap Between Flexibility and Performance: PyTorch eager mode is flexible but can be limited in performance for custom operations. Triton allows users to maintain flexibility while achieving near hand-tuned performance for their specific workloads.
- Accelerates Innovation: By lowering the barrier to writing high-performance GPU code, Triton enables faster prototyping and deployment of new algorithms, especially in areas like deep learning, scientific computing, and large-scale data processing.
- Complementary to PyTorch: Triton is not a replacement for PyTorch, but a powerful extension. It allows users to keep the productivity and ecosystem of PyTorch while extending its capabilities to custom, high-performance GPU operations.
- Works hand-in-hand with TorchInductor: TorchInductor is PyTorch's compiler backend that automatically lowers and optimizes PyTorch programs into efficient kernels, often using Triton as a code generator for GPU. Together, Inductor + Triton provide an end-to-end path from high-level PyTorch code to optimized GPU execution with minimal manual kernel writing.

## 4. How does Triton work?

A typical Triton program consists of device kernels and a host program that calls the kernels.

A Triton kernel has this form:

```python
@triton.jit
def kernel_name(...):
    # ...
```

And a host program calls the kernel as follows:

```python
result = kernel_name[grid](...)
```

where grid is a tuple defining the number of thread blocks in each dimension:

```
An 1D grid:  grid = (X,)
A 2D grid:   grid = (X, Y)
A 3D grid:   grid = (X, Y, Z)
```

For Nvidia GPUs, when a kernel is called, the CUDA runtime launches a grid of threads that execute the kernel code.

The `@triton.jit` decorator is used to define a Triton kernel for just-in-time (JIT) compiling. This enables developers to write concise, high-performance code for GPU kernels while leveraging the familiar Python syntax.

### The Compilation Process

Here's a breakdown of how it works:

1. **Kernel declaration and compilation:**
   - A kernel is a Python function that is decorated with the `@triton.jit` decorator, which tells the Triton compiler to compile that kernel.
   - This kernel will be executed on the GPU when it is called.

2. **Intermediate representation (IR) generation:**
   - The Triton compiler analyzes the kernel's abstract syntax tree (AST) to generate a Triton intermediate representation (Triton-IR).
   - Triton-IR is a machine-independent, unoptimized representation of the kernel, capturing the high-level structure of the computation.

3. **Optimization:**
   - The Triton compiler performs optimizations on the generated Triton-IR.
   - The optimized Triton-IR is then lowered (transformed) to Triton-GPU IR (Triton-TTGIR) and subsequently to low-level virtual machine IR (LLVM-IR).

4. **GPU code generation:**
   - The LLVM-IR is used to generate CUDA code or AMDGCN code.
   - For NVIDIA GPUs, from LLVM-IR the Triton compiler generates Parallel Thread Execution (PTX) code which is then JIT-compiled by Nvidia ptxas into a CUDA binary (CUBIN).
   - For AMD GPUs, from LLVM-IR the Triton compiler generates AMDGCN code which is then compiled by AMD JIT into AMD hsaco binary.

5. **Execution on GPU:**
   - The JIT-compiled GPU code is executed in parallel on the GPU, leveraging its processing power for accelerated computations.

## Try It

### Get your notebook ready

We'll be testing some code in this tutorial, so make sure you have a Jupyter Notebook such as Google Colab.

### Install Triton (if needed)

Run this cell to install Triton. (You only need to do this once per environment.)

```bash
!pip install triton
```

### Import Libraries

```python
import torch
import triton
import triton.language as tl
```

### Define the Triton Kernel

```python
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
```

### Prepare Data and Launch the Kernel

```python
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
```

### View the Results

```python
print("Input x:", data_x)
print("Input y:", data_y)
print("Output (x + y):", output)
```

### What You Should See

```
Input x: tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda:0')
Input y: tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.], device='cuda:0')
Output (x + y): tensor([10., 10., 10., 10., 10., 10., 10., 10., 10., 10.], device='cuda:0')
```

**Tips for Jupyter Notebook Users:**
- Run each code block in a separate cell for clarity.
- If you encounter errors about CUDA or Triton, make sure your notebook is running on a GPU-enabled environment (e.g., Google Colab with GPU runtime).
- You can modify the input data or kernel logic to experiment further.

## Conclusion

By completing this tutorial, you should now understand the fundamentals of Triton, including its purpose, advantages over PyTorch eager mode, and its significance in the PyTorch ecosystem. With this foundation, you are ready to start experimenting with Triton and explore its potential for accelerating your own deep learning and scientific computing projects.

---

**Next:** [Getting Started with Triton →](getting-started.md)

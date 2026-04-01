# Introduction to Triton

## 1. What is Triton?

Triton is an open-source, Python-based programming language and compiler for writing highly efficient custom GPU kernels. It transforms high-level Python code into GPU machine instructions, enabling developers and researchers—even those with little or no experience in GPU hardware or specialized languages like CUDA—to write expressive, parallel programs for accelerators such as NVIDIA GPUs. Triton's Pythonic syntax is familiar to users of scientific computing libraries like NumPy and PyTorch, making it accessible for a wide range of scientific and engineering applications.

Key characteristics of Triton:

- **Pythonic API:** Write GPU code in a style similar to NumPy or PyTorch, making it accessible to Python developers.
- **Custom GPU Kernels:** Allows you to write your own GPU operations (kernels) for tasks where built-in libraries may not be optimal.
- **Automatic Compilation:** Triton compiles your Python-like code into highly optimized GPU code, abstracting away much of the complexity of CUDA or other low-level languages.
- **Focus on Performance:** Designed to achieve performance close to hand-written CUDA, but with much less boilerplate and complexity.

## 2. When Should You Use Triton Instead of PyTorch Eager Mode?

PyTorch eager mode executes operations immediately as they are called, which is excellent for rapid prototyping and flexibility. However, it relies on pre-built GPU kernels from libraries such as cuBLAS and cuDNN for most operations. This means you are limited to what those libraries provide.

Triton is the right choice when:

- You need a custom GPU operation that PyTorch does not provide out of the box.
- You want to fuse multiple operations into a single kernel to reduce memory bandwidth usage.
- You need to optimize a specific computation for your workload beyond what generic PyTorch kernels offer.

In short: use PyTorch eager mode for standard operations and prototyping; reach for Triton when you need custom, high-performance GPU code.

### Triton, PyTorch Eager, and Helion Compared

| Feature | PyTorch Eager | Triton | Helion |
|---------|--------------|--------|--------|
| Custom GPU Kernels | Limited | Full control | Partial (automated tiling) |
| Performance Tuning | Limited to built-ins | Fine-grained, user-optimized | Automatic search |
| API Style | Pythonic, high-level | Pythonic, kernel-focused | Pythonic, high-level with tiling abstraction |
| Integration | Native | Seamless with PyTorch | Seamless with PyTorch |
| Learning Curve | Low | Moderate (easier than CUDA) | Moderate |

[Helion](https://github.com/pytorch-labs/helion) is a Python-embedded DSL for writing high-performance ML kernels that compiles to Triton. It offers a higher-level abstraction, with features like automatic grid size calculation and autotuning of data tile mappings, making it even easier to get started with custom kernels.

## 3. How Triton Fits into the PyTorch Ecosystem

The most important thing to understand is that Triton is not a replacement for PyTorch—it is a powerful extension. You keep all of the productivity and ecosystem of PyTorch while gaining the ability to write custom, high-performance GPU operations when you need them.

Triton integrates with PyTorch in two key ways:

- **Direct use:** You can write a Triton kernel and call it directly from your PyTorch code, passing PyTorch tensors as arguments.
- **TorchInductor:** PyTorch's compiler backend (`torch.compile`) uses Triton as a code generator for GPU operations. When you call `torch.compile` on a PyTorch model, Inductor may automatically generate Triton kernels to accelerate it—so even if you never write a Triton kernel yourself, Triton is likely running under the hood.

## 4. How Does Triton Work?

A Triton program has two parts: a kernel that runs on the GPU (the device), and a host program that runs on the CPU and is responsible for preparing data, configuring the launch, and calling the kernel.

A Triton kernel is a regular Python function decorated with `@triton.jit`:

```python
@triton.jit
def kernel_name(...):
    # GPU code goes here
```

The host program launches the kernel by indexing it with a grid:

```python
result = kernel_name[grid](...)
```

The grid specifies how many independent instances of the kernel to launch, organized as a 1D, 2D, or 3D tuple:

```python
grid = (X,)        # 1D grid: X kernel instances
grid = (X, Y)      # 2D grid: X*Y kernel instances
grid = (X, Y, Z)   # 3D grid: X*Y*Z kernel instances
```

Each kernel instance is called a **program**. Inside the kernel, `tl.program_id(axis=0)` returns the unique block ID of the current program instance along the given axis. This is how each instance knows which slice of the data it should process.

The `@triton.jit` decorator tells the Triton compiler to JIT-compile this function into a GPU kernel the first time it is called.

### The Compilation Process

The Triton compilation process involves several stages:

1. **Kernel declaration and compilation:** The `@triton.jit` decorator marks the function for JIT compilation. On first call, the Triton compiler walks the Python AST to generate Triton-IR.

2. **Intermediate representation (IR) generation:** Triton-IR is a machine-independent, unoptimized representation of the kernel.

3. **Optimization and lowering:** Triton-IR is optimized and lowered to Triton-GPU IR (TTGIR), which encodes hardware-specific information such as how tensors are distributed across warps and compute capability. TTGIR is then lowered to LLVM-IR.

4. **GPU code generation:** The LLVM-IR is passed to vendor toolchains to produce executable GPU code.
   - For NVIDIA GPUs: LLVM-IR → PTX (Parallel Thread Execution assembly) → CUBIN (CUDA binary), compiled by NVIDIA's `ptxas` assembler.
   - For AMD GPUs: LLVM-IR → AMDGCN code → HSACO binary, compiled by AMD's toolchain.

5. **Execution on GPU:** The compiled binary is launched on the GPU according to the grid you specified.

## Try It

### Get Your Notebook Ready

Before running the code in this tutorial, make sure you have a Jupyter Notebook environment with a GPU runtime. If you are using Google Colab, go to **Runtime > Change runtime type** and select a GPU accelerator. All tutorials in this series require GPU access and will fail on a CPU-only runtime with an error such as `"torch compiled without CUDA enabled."`

### Install Triton (if needed)

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
```

### Prepare Data and Launch the Kernel

```python
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

## Conclusion

By completing this tutorial, you should now understand what Triton is and when to use it, how it fits into the PyTorch ecosystem, and the basics of how a Triton kernel is structured and compiled. With this foundation, you are ready to start writing your own kernels in the tutorials that follow.

---

**Next:** [Getting Started with Triton →](getting-started.md)

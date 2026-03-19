# Triton Tutorial Series

## Resources

- [SOTA Deep Learning Tutorials — YouTube](https://www.youtube.com/@SOTADeepLearningTutorials)
- [Triton documentation](https://triton-lang.org/)
- [GitHub — gpu-mode/triton-tutorials](https://github.com/gpu-mode/triton-tutorials/)

## How to Use This Series

This series of tutorials is designed to be a hands-on guide to learning Triton, from fundamental concepts to writing and optimizing your own GPU kernels. Each tutorial is self-contained and can be run end-to-end in a Jupyter notebook environment with a GPU runtime. The expected time to complete the full series is approximately 8–12 hours, depending on your pace and how much you experiment with the code.

### Target Audience and Prerequisites

This series is intended for developers and researchers who are familiar with Python and PyTorch and want to learn how to write custom GPU kernels to accelerate their workloads. No prior experience with GPU programming or CUDA is required.

**Prerequisites:**

- Python 3.x
- PyTorch (with CUDA support)
- A CUDA-enabled NVIDIA GPU. All tutorials require a GPU runtime and will not work in a CPU-only environment. If using Google Colab, select a GPU runtime under **Runtime > Change runtime type**.
- Jupyter Notebook or JupyterLab (Google Colab works well)

## Series Outline

### Basics
- Intro to Triton
  - What is Triton?
  - When to use Triton vs. PyTorch eager mode
  - How Triton fits into the PyTorch ecosystem
  - Getting Started with Triton
  - Triton vs. CUDA
  - Introduction to GPU Fundamentals for Kernel Programming
- First Kernel — Vector Addition
  - Making the shift to parallel programming
  - Coding a vector addition kernel
  - Verifying numerics
  - Benchmarking vs PyTorch
- Softmax
  - Basic softmax in PyTorch
  - Online softmax in PyTorch
  - Softmax in Triton
- Tiled Matrix Multiplication in Triton
- Part 1: Flash Attention 2: Overview
- Part 2: Flash Attention 2: Forward Kernel
- Part 3: Flash Attention 2: Backward Kernel
- Part 4: Benchmarking / Tuning

### Advanced Topics
- TMA
- Warp Specialization
- Coop Matmul in Triton

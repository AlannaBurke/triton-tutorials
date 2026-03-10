# Triton Tutorial Series

## Initial Project Docs & Resources

- [Triton Tutorials 2025](https://www.internalfb.com/wiki/Triton_Tutorials/)
- [SOTA Deep Learning Tutorials - YouTube](https://www.youtube.com/@SOTADeepLearningTutorials)
- [Triton documentation](https://triton-lang.org/)
- [GitHub - gpu-mode/triton-tutorials](https://github.com/gpu-mode/triton-tutorials/)

## Series Outline

### Basics
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

### Advanced Topics
- TMA
- Warp Specialization
- Coop Matmul in Triton

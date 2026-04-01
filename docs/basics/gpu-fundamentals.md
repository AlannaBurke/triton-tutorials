# Introduction to GPU Fundamentals for Kernel Programming

## Introduction

This tutorial provides an introduction to GPU architecture concepts that are directly relevant to writing Triton kernels. Understanding how a GPU is organized — and where data lives at each stage of computation — is essential for writing kernels that are both correct and performant.

## GPU Memory Hierarchy

A modern GPU has two main types of memory that Triton programmers need to understand:

**HBM (High Bandwidth Memory)**, also called global memory, is the large off-chip memory pool on the GPU. When you call `.cuda()` on a PyTorch tensor, the data is stored in HBM. HBM is accessible by all streaming multiprocessors (SMs) on the GPU, but accessing it is relatively slow compared to on-chip memory.

**SRAM (Static RAM)**, also called shared memory or on-chip memory, is a small, very fast memory that lives on each SM. It is shared by all threads running on that SM. Because it is on-chip, reads and writes to SRAM are much faster than to HBM. Efficient GPU kernels are often designed to load data from HBM into SRAM, operate on it there, and write results back — minimizing slow HBM traffic.

## Streaming Multiprocessors (SMs)

SMs are the core compute units of the GPU where actual processing happens. For example, the NVIDIA A100 GPU has 108 SMs. When you launch a Triton kernel, the GPU scheduler distributes your kernel instances (programs) across the available SMs.

## How Triton Relates to This

Triton kernels operate on data in HBM (global memory). Triton does not expose explicit shared memory management in the same way CUDA does — instead, the Triton compiler is responsible for automatically managing on-chip memory to optimize data movement. This is one of Triton's key advantages: you write the algorithm, and the compiler handles the low-level memory placement.

PyTorch, by contrast, primarily orchestrates operations at the HBM level, calling into pre-built library kernels (cuBLAS, cuDNN, etc.) that internally manage shared memory. Triton gives you the ability to write those inner kernels yourself.

## Key Takeaways

The memory bandwidth bottleneck — how fast data can be moved between HBM and the SMs — is typically the primary performance limiter on GPUs for operations like softmax and attention, not raw arithmetic throughput. Writing efficient Triton kernels means designing your data access patterns to maximize bandwidth utilization and minimize redundant HBM reads and writes.

---

**Next:** [First Kernel: Vector Addition →](../first-kernel/index.md)

# Part 3: Verifying Numerical Fidelity

*This tutorial follows [Part 2](part2-writing-a-kernel.md) of this series.*

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 3: Verifying Numerical Accuracy](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

After implementing a Triton kernel, the next critical step is to verify that it produces correct results. This tutorial demonstrates how to validate that the Triton kernel produces results consistent with PyTorch's native operations, using `torch.allclose` to compare outputs and ensure correctness within a specified numerical tolerance.

## Step 1: Setting Up the Test Environment

```python
import torch
torch.manual_seed(0)  # Seeds both CPU and GPU RNGs for reproducibility
```

> **Note:** `torch.manual_seed(0)` seeds both the CPU and CUDA random number generators when a CUDA device is available, ensuring reproducible results across runs.

## Step 2: Creating Test Vectors

```python
vector_size = 8192
A = torch.rand(vector_size, device='cuda')
B = torch.rand_like(A)
```

## Step 3: Computing Reference Result with PyTorch

```python
torch_result = A + B
```

## Step 4: Running the Triton Kernel

```python
triton_result = vector_addition(A, B)
```

## Step 5: Comparing Results with `torch.allclose`

```python
result_correct = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-4)
print("Numerical fidelity correct:", result_correct)
```

The `atol` (absolute tolerance) and `rtol` (relative tolerance) parameters control how much deviation is acceptable. For simple floating-point addition, `atol=1e-6` is a reasonable threshold. For more complex operations involving many accumulations, you may need to loosen tolerances slightly.

## Step 6: Wrapping the Verification in a Function

```python
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

if __name__ == "__main__":
    verify_numerics()
```

## Troubleshooting Common Errors

- **`AttributeError: 'Tensor' object has no attribute 'is_cuda'`** — Use `tensor.is_cuda` as a property (no parentheses), not as a method call.
- **Results don't match** — Check that your mask is applied consistently to both `tl.load` and `tl.store`. A missing mask on the store can cause garbage values to be written.
- **Non-reproducible results** — Ensure `torch.manual_seed` is called before creating the test tensors.

## Conclusion

By completing this tutorial, you will have learned how to validate the numerical accuracy of your Triton vector addition kernel, ensuring it matches PyTorch's results within reasonable tolerances.

---

**Next:** [Part 4: Benchmarking →](part4-benchmarking.md)

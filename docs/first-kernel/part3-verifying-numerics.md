# Part 3: Verifying Numerical Fidelity

*This tutorial follows Part 2 of this series.*

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Triton Vector Addition Kernel, part 3: Verifying Numerical Accuracy](https://www.youtube.com/watch?v=your-link-here)*

## Introduction

After implementing a Triton kernel, the next critical step is to verify that it produces correct results. This tutorial demonstrates how to validate that the Triton kernel produces results consistent with PyTorch's native operations. We will use PyTorch's `torch.allclose` API to compare outputs and ensure correctness. We will also set up a reproducible testing environment and discuss considerations for numerical tolerance.

## Step 1: Setting Up the Test Environment

```python
import torch
torch.manual_seed(0)  # Seed CPU and GPU RNGs for reproducibility
```

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

Notes on Tolerances:
- The absolute tolerance (`atol`) and relative tolerance (`rtol`) may need adjustment depending on the operation and data.
- For more complex operations, tighter tolerances may be needed to ensure correctness.

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

- Ensure that input tensors are on CUDA devices (`.is_cuda` property).
- Use `torch.manual_seed` to seed both CPU and GPU RNGs for reproducibility.
- When calling `.is_cuda`, use it as a property, not a method (i.e., no parentheses).
- If results are incorrect, check that your mask is applied consistently to both `tl.load` and `tl.store`.

## Conclusion

By completing this tutorial, you will have learned how to validate the numerical accuracy of your Triton vector addition kernel, ensuring it matches PyTorch's results within reasonable tolerances.

---

**Next:** [Part 4: Benchmarking →](part4-benchmarking.md)

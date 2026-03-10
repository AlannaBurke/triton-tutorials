# Part 1: Basic Softmax and Online Softmax in PyTorch

*This tutorial is derived from this video by SOTA Deep Learning Tutorials:*
*[Intro to Triton: Coding Softmax in PyTorch](https://www.youtube.com/watch?v=your-link-here)*

## 1. Environment Setup

```bash
pip install torch triton
```

## 2. Creating a Sample Tensor

```python
import torch
import torch.nn.functional as F

sample = torch.tensor(
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    dtype=torch.float32,
    device='cuda'
)
```

## 3. Reference: PyTorch Softmax

```python
softmax_ref = F.softmax(sample, dim=1)
print("PyTorch Softmax Output:\n", softmax_ref)
```

## 4. Naive Softmax: Implementation and Inefficiencies

Let's implement our own numerically stable softmax function in PyTorch:

```python
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
```

This implementation requires multiple passes over each row of data:
1. Find the maximum value (for numerical stability) — 1st pass.
2. Subtract the max and exponentiate — 2nd pass.
3. Sum the exponentiated values — 3rd pass.
4. Divide to get the final softmax — 4th pass.

Each pass reads the entire row from HBM, which is costly for large tensors.

## 5. Online Softmax: Reducing Memory Passes

Online softmax reduces the number of passes to just two by maintaining running statistics—the current row maximum and the running sum of exponentials—as it processes each element. When the running maximum is updated, the accumulated sum is rescaled to remain consistent.

```python
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
```

## 6. Performance Comparison

In practice, the online softmax is significantly faster than the naive version for large tensors, because it reduces the number of passes over each row from four to two. The performance gain grows with the size of the input.

## Conclusion

In this tutorial, you implemented both naive and online softmax in PyTorch and verified their correctness.

---

**Next:** [Part 2: Softmax in Triton →](part2-softmax-in-triton.md)

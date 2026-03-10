# Getting Started with Triton

Triton is a Python-based language and compiler that transforms high-level Python code into GPU machine instructions (i.e., Parallel Thread Execution (PTX) code). It enables programmers with little or no experience of GPU hardware and GPU-specific programming languages, such as CUDA, to write very efficient parallel programs.

A typical Triton program consists of Triton kernels (functions) and a host program that calls the kernels. Triton kernels are executed in parallel by many threads on the GPU.

We'll be testing some code in this tutorial, so make sure you have a Jupyter Notebook such as Google Colab.

## Vector Addition on the GPU

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
```

And the following code does the following vector addition:

```python
BLOCK_SIZE = 1
data = [1, 2, 3, 4, 5]
x = torch.tensor(data, device='cuda')
output = torch.empty_like(x)
N = len(data)
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
add_kernel[grid](x, 10, output, N, BLOCK_SIZE)

print(f"x = {x}")
print(f"x + 10 = {output}")
```

The output looks like this:

```
x = tensor([1, 2, 3, 4, 5], device='cuda:0')
x + 10 = tensor([11, 12, 13, 14, 15], device='cuda:0')
```

## Testing Triton Tutorials

Before testing the Triton tutorials, download them:
- Open the tutorials page.
- Scroll to the bottom of the page.
- Click Download all examples in Python source code: tutorials_python.zip.
- Extract the tutorials:

```bash
$ cd ~
$ mkdir tutorials_python/
$ unzip ~/Downloads/tutorials_python.zip -d tutorials_python/
$ cd tutorials_python/
$ ls
01-vector-add.py  06-fused-attention.py
02-fused-softmax.py  07-extern-functions.py
...
```

After downloading the Triton tutorials, test them in your notebook. For example, to test 01-vector-add.py, do the following:
- Open an existing notebook or create a new notebook.
- If it is not connected to a server, connect it to an on-demand GPU devserver.
- Make a copy of ~/tutorials_python/01-vector-add.py and paste it to a cell in your notebook.
- Run 01-vector-add.py. If you get this error (see the caveats):

Make this change in 01-vector-add.py:

```python
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Re-run 01-vector-add.py.

## Conclusion

By completing this tutorial, you will have taken your first practical steps in GPU programming with Triton. You will understand how Triton kernels work, how to structure and execute a basic program, and how to test your code interactively in a Jupyter Notebook.

---

**Next:** [Triton vs. CUDA →](triton-vs-cuda.md)


| **Function/Method**       | **Description**                                                                                                                                                          |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **is_tensor**             | Returns `True` if `obj` is a PyTorch tensor.                                                                                                                             |
| **is_storage**            | Returns `True` if `obj` is a PyTorch storage object.                                                                                                                     |
| **is_complex**            | Returns `True` if the data type of `input` is a complex data type (i.e., `torch.complex64`, `torch.complex128`).                                                          |
| **is_conj**               | Returns `True` if the `input` is a conjugated tensor, i.e., its conjugate bit is set to `True`.                                                                           |
| **is_floating_point**     | Returns `True` if the data type of `input` is a floating point data type (i.e., `torch.float64`, `torch.float32`, `torch.float16`, `torch.bfloat16`).                    |
| **is_nonzero**            | Returns `True` if the `input` is a single element tensor which is not equal to zero after type conversions.                                                              |
| **set_default_dtype**     | Sets the default floating point `dtype` to `d`.                                                                                                                          |
| **get_default_dtype**     | Gets the current default floating point `torch.dtype`.                                                                                                                   |
| **set_default_device**    | Sets the default `torch.Tensor` to be allocated on `device`.                                                                                                             |
| **get_default_device**    | Gets the default `torch.Tensor` to be allocated on `device`.                                                                                                             |
| **set_default_tensor_type**| Sets the default `torch.Tensor` type.                                                                                                                                   |
| **numel**                 | Returns the total number of elements in the `input` tensor.                                                                                                              |
| **set_printoptions**      | Sets options for printing.                                                                                                                                               |
| **set_flush_denormal**    | Disables denormal floating numbers on CPU.                                                                                                                               |

## Detailed Explanation

### `is_tensor`

**Description**: Returns `True` if `obj` is a PyTorch tensor.

**Usage**:
```python
import torch
x = torch.tensor([1, 2, 3])
print(torch.is_tensor(x))  # True
```

### `is_storage`

**Description**: Returns `True` if `obj` is a PyTorch storage object.

**Usage**:
```python
storage = torch.FloatStorage(10)
print(torch.is_storage(storage))  # True
```

### `is_complex`

**Description**: Returns `True` if the data type of `input` is a complex data type (i.e., `torch.complex64`, `torch.complex128`).

**Usage**:
```python
x = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
print(torch.is_complex(x))  # True
```

### `is_conj`

**Description**: Returns `True` if the `input` is a conjugated tensor, i.e., its conjugate bit is set to `True`.

**Usage**:
```python
x = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
conj_x = x.conj()
print(torch.is_conj(conj_x))  # True
```

### `is_floating_point`

**Description**: Returns `True` if the data type of `input` is a floating point data type (i.e., `torch.float64`, `torch.float32`, `torch.float16`, `torch.bfloat16`).

**Usage**:
```python
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
print(torch.is_floating_point(x))  # True
```

### `is_nonzero`

**Description**: Returns `True` if the `input` is a single element tensor which is not equal to zero after type conversions.

**Usage**:
```python
x = torch.tensor(3)
print(torch.is_nonzero(x))  # True
```

### `set_default_dtype`

**Description**: Sets the default floating point `dtype` to `d`.

**Usage**:
```python
torch.set_default_dtype(torch.float64)
print(torch.tensor([1.0, 2.0]).dtype)  # torch.float64
```

### `get_default_dtype`

**Description**: Gets the current default floating point `torch.dtype`.

**Usage**:
```python
print(torch.get_default_dtype())  # torch.float32 (default)
```

### `set_default_device`

**Description**: Sets the default `torch.Tensor` to be allocated on `device`.

**Usage**:
```python
torch.set_default_device('cuda')
print(torch.tensor([1.0, 2.0]).device)  # cuda:0
```

### `get_default_device`

**Description**: Gets the default `torch.Tensor` to be allocated on `device`.

**Usage**:
```python
print(torch.get_default_device())  # cpu (default)
```

### `set_default_tensor_type`

**Description**: Sets the default `torch.Tensor` type.

**Usage**:
```python
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.0, 2.0]).dtype)  # torch.float64
```

### `numel`

**Description**: Returns the total number of elements in the `input` tensor.

**Usage**:
```python
x = torch.tensor([[1, 2], [3, 4]])
print(x.numel())  # 4
```

### `set_printoptions`

**Description**: Sets options for printing.

**Usage**:
```python
torch.set_printoptions(precision=2)
x = torch.tensor([1.12345, 2.12345])
print(x)  # tensor([1.12, 2.12])
```

### `set_flush_denormal`

**Description**: Disables denormal floating numbers on CPU.

**Usage**:
```python
torch.set_flush_denormal(True)
```
Here is a detailed tabular representation of the data types for `torch.Tensor` in PyTorch, including their respective `dtype` values.

| **Data Type**                                      | **`dtype`**                         | **Description**                                         |
|----------------------------------------------------|-------------------------------------|---------------------------------------------------------|
| **32-bit floating point**                          | `torch.float32` or `torch.float`    | Standard 32-bit floating point                           |
| **64-bit floating point**                          | `torch.float64` or `torch.double`   | Standard 64-bit floating point                           |
| **16-bit floating point**                          | `torch.float16` or `torch.half`     | Standard 16-bit floating point                           |
| **16-bit floating point (bfloat16)**               | `torch.bfloat16`                    | Brain floating point format                              |
| **32-bit complex**                                 | `torch.complex32` or `torch.chalf`  | Standard 32-bit complex number                           |
| **64-bit complex**                                 | `torch.complex64` or `torch.cfloat` | Standard 64-bit complex number                           |
| **128-bit complex**                                | `torch.complex128` or `torch.cdouble`| Standard 128-bit complex number                          |
| **8-bit integer (unsigned)**                       | `torch.uint8`                       | Unsigned 8-bit integer                                   |
| **16-bit integer (unsigned)**                      | `torch.uint16`                      | Unsigned 16-bit integer (limited support)                |
| **32-bit integer (unsigned)**                      | `torch.uint32`                      | Unsigned 32-bit integer (limited support)                |
| **64-bit integer (unsigned)**                      | `torch.uint64`                      | Unsigned 64-bit integer (limited support)                |
| **8-bit integer (signed)**                         | `torch.int8`                        | Signed 8-bit integer                                     |
| **16-bit integer (signed)**                        | `torch.int16` or `torch.short`      | Signed 16-bit integer                                    |
| **32-bit integer (signed)**                        | `torch.int32` or `torch.int`        | Signed 32-bit integer                                    |
| **64-bit integer (signed)**                        | `torch.int64` or `torch.long`       | Signed 64-bit integer                                    |
| **Boolean**                                        | `torch.bool`                        | Boolean type                                             |
| **Quantized 8-bit integer (unsigned)**             | `torch.quint8`                      | Quantized unsigned 8-bit integer                         |
| **Quantized 8-bit integer (signed)**               | `torch.qint8`                       | Quantized signed 8-bit integer                           |
| **Quantized 32-bit integer (signed)**              | `torch.qint32`                      | Quantized signed 32-bit integer                          |
| **Quantized 4-bit integer (unsigned)**             | `torch.quint4x2`                    | Quantized unsigned 4-bit integer                         |
| **8-bit floating point, e4m3**                     | `torch.float8_e4m3fn`               | 8-bit floating point, e4m3 format (limited support)      |
| **8-bit floating point, e5m2**                     | `torch.float8_e5m2`                 | 8-bit floating point, e5m2 format (limited support)      |

### Explanation

- **32-bit Floating Point**: The standard single-precision floating-point format.
- **64-bit Floating Point**: The standard double-precision floating-point format.
- **16-bit Floating Point**: A half-precision floating-point format often used for reduced memory consumption.
- **16-bit Floating Point (bfloat16)**: A variant of the 16-bit floating-point format that offers a larger range by having fewer mantissa bits and more exponent bits.
- **Complex Numbers**: Complex numbers are represented in 32-bit, 64-bit, and 128-bit formats, combining two floating-point values for the real and imaginary parts.
- **8-bit and Higher Integer Types**: Both unsigned and signed integers are supported in 8-bit, 16-bit, 32-bit, and 64-bit formats.
- **Boolean**: Represents binary values (True/False).
- **Quantized Types**: Used for models that require quantization for reduced memory and computation, available in 8-bit, 4-bit, and special floating-point formats.

These `dtype` values allow users to define tensors with the desired precision and memory requirements for various applications in deep learning and numerical computations.


| **Function**                  | **Description**                                                                                                                                                                                                                             |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tensor`                      | Constructs a tensor with no autograd history by copying data.                                                                                                                                                                               |
| `sparse_coo_tensor`           | Constructs a sparse tensor in COO (Coordinate) format with specified values at the given indices.                                                                                                                                           |
| `sparse_csr_tensor`           | Constructs a sparse tensor in CSR (Compressed Sparse Row) format with specified values at the given `crow_indices` and `col_indices`.                                                                                                        |
| `sparse_csc_tensor`           | Constructs a sparse tensor in CSC (Compressed Sparse Column) format with specified values at the given `ccol_indices` and `row_indices`.                                                                                                      |
| `sparse_bsr_tensor`           | Constructs a sparse tensor in BSR (Block Compressed Sparse Row) format with specified 2-dimensional blocks at the given `crow_indices` and `col_indices`.                                                                                     |
| `sparse_bsc_tensor`           | Constructs a sparse tensor in BSC (Block Compressed Sparse Column) format with specified 2-dimensional blocks at the given `ccol_indices` and `row_indices`.                                                                                  |
| `asarray`                     | Converts `obj` to a tensor.                                                                                                                                                                                                                  |
| `as_tensor`                   | Converts data into a tensor, sharing data and preserving autograd history if possible.                                                                                                                                                       |
| `as_strided`                  | Creates a view of an existing `torch.Tensor` input with specified `size`, `stride` and `storage_offset`.                                                                                                                                      |
| `from_file`                   | Creates a CPU tensor with storage backed by a memory-mapped file.                                                                                                                                                                            |
| `from_numpy`                  | Creates a tensor from a `numpy.ndarray`.                                                                                                                                                                                                     |
| `from_dlpack`                 | Converts a tensor from an external library into a `torch.Tensor`.                                                                                                                                                                            |
| `frombuffer`                  | Creates a 1-dimensional tensor from an object that implements the Python buffer protocol.                                                                                                                                                     |
| `zeros`                       | Returns a tensor filled with the scalar value `0`, with the shape defined by the variable argument `size`.                                                                                                                                    |
| `zeros_like`                  | Returns a tensor filled with the scalar value `0`, with the same size as `input`.                                                                                                                                                            |
| `ones`                        | Returns a tensor filled with the scalar value `1`, with the shape defined by the variable argument `size`.                                                                                                                                    |
| `ones_like`                   | Returns a tensor filled with the scalar value `1`, with the same size as `input`.                                                                                                                                                            |
| `arange`                      | Returns a 1-D tensor of size ⌈(end−start)/step⌉ with values from the interval `[start, end)` taken with a common difference `step` beginning from `start`.                                                                                   |
| `range`                       | Returns a 1-D tensor of size ⌊(end−start)/step⌋ + 1 with values from `start` to `end` with step `step`.                                                                                                                                       |
| `linspace`                    | Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `end`, inclusive.                                                                                                                            |
| `logspace`                    | Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from `base^start` to `base^end`, inclusive, on a logarithmic scale with base `base`.                                                                          |
| `eye`                         | Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.                                                                                                                                                                          |
| `empty`                       | Returns a tensor filled with uninitialized data.                                                                                                                                                                                             |
| `empty_like`                  | Returns an uninitialized tensor with the same size as `input`.                                                                                                                                                                               |
| `empty_strided`               | Creates a tensor with the specified `size` and `stride` and filled with undefined data.                                                                                                                                                      |
| `full`                        | Creates a tensor of size `size` filled with `fill_value`.                                                                                                                                                                                    |
| `full_like`                   | Returns a tensor with the same size as `input` filled with `fill_value`.                                                                                                                                                                     |
| `quantize_per_tensor`         | Converts a float tensor to a quantized tensor with given `scale` and `zero point`.                                                                                                                                                           |
| `quantize_per_channel`        | Converts a float tensor to a per-channel quantized tensor with given `scales` and `zero points`.                                                                                                                                              |
| `dequantize`                  | Returns an `fp32` tensor by dequantizing a quantized tensor.                                                                                                                                                                                 |
| `complex`                     | Constructs a complex tensor with its real part equal to `real` and its imaginary part equal to `imag`.                                                                                                                                       |
| `polar`                       | Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value `abs` and angle `angle`.                                                                                      |
| `heaviside`                   | Computes the Heaviside step function for each element in `input`.                                                                                                                                                                            |

### Detailed Descriptions and Mathematical Notations

| **Function**        | **Description and Mathematical Notation**                                                                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tensor`            | Constructs a tensor with no autograd history by copying data.                                                                                                                                                                                                                                                                                           |
| `sparse_coo_tensor` | Constructs a sparse tensor in COO (Coordinate) format. For given indices \(\mathbf{i}\) and values \(\mathbf{v}\), the tensor \(\mathbf{T}\) is formed such that \(T[i_k] = v_k\).                                                                                                                                                                       |
| `sparse_csr_tensor` | Constructs a sparse tensor in CSR (Compressed Sparse Row) format. For given row indices \(\mathbf{r}\), column indices \(\mathbf{c}\), and values \(\mathbf{v}\), the tensor \(\mathbf{T}\) is formed.                                                                                                                                                   |
| `sparse_csc_tensor` | Constructs a sparse tensor in CSC (Compressed Sparse Column) format. For given column indices \(\mathbf{c}\), row indices \(\mathbf{r}\), and values \(\mathbf{v}\), the tensor \(\mathbf{T}\) is formed.                                                                                                                                               |
| `sparse_bsr_tensor` | Constructs a sparse tensor in BSR (Block Compressed Sparse Row) format. For given block row indices \(\mathbf{br}\), block column indices \(\mathbf{bc}\), and block values \(\mathbf{bv}\), the tensor \(\mathbf{T}\) is formed.                                                                                                                         |
| `sparse_bsc_tensor` | Constructs a sparse tensor in BSC (Block Compressed Sparse Column) format. For given block column indices \(\mathbf{bc}\), block row indices \(\mathbf{br}\), and block values \(\mathbf{bv}\), the tensor \(\mathbf{T}\) is formed.                                                                                                                      |
| `asarray`           | Converts `obj` to a tensor.                                                                                                                                                                                                                                                                                                                             |
| `as_tensor`         | Converts data into a tensor, sharing data and preserving autograd history if possible.                                                                                                                                                                                                                                                                 |
| `as_strided`        | Creates a view of an existing `torch.Tensor` with specified `size`, `stride`, and `storage_offset`.                                                                                                                                                                                                                                                     |
| `from_file`         | Creates a CPU tensor with storage backed by a memory-mapped file.                                                                                                                                                                                                                                                                                        |
| `from_numpy`        | Creates a tensor from a `numpy.ndarray`.                                                                                                                                                                                                                                                                                                                |
| `from_dlpack`       | Converts a tensor from an external library into a `torch.Tensor`.                                                                                                                                                                                                                                                                                       |
| `frombuffer`        | Creates a 1-dimensional tensor from an object that implements the Python buffer protocol.                                                                                                                                                                                                                                                                |
| `zeros`             | Returns a tensor filled with the scalar value \(0\), with the shape defined by the variable argument `size`. If \(\mathbf{s} = (s_1, s_2, \ldots, s_n)\), then \(\mathbf{T}\) such that \(T[i_1, i_2, \ldots, i_n] = 0\) for all valid \(i_1, i_2, \ldots, i_n\).                                                                                        |
| `zeros_like`        | Returns a tensor filled with the scalar value \(0\), with the same size as `input`.                                                                                                                                                                                                                                                                      |
| `ones`              | Returns a tensor filled with the scalar value \(1\), with the shape defined by the variable argument `size`. If \(\mathbf{s} = (s_1, s_2, \ldots, s_n)\), then \(\mathbf{T}\) such that \(T[i_1, i_2, \ldots, i_n] = 1\) for all valid \(i_1, i_2, \ldots, i_n\).                                                                                        |
| `ones_like`         | Returns a tensor filled with the scalar value \(1\), with the same size as `input`.                                                                                                                                                                                                                                                                      |
| `arange`            | Returns a 1-D tensor of size \(\left\lceil \frac{end - start}{step} \right\rceil\) with values from the interval \([start, end)\) taken with a common difference `step`.                                                                                                                                                                                |
| `range`             | Returns a 1-D tensor of size \(\left\lfloor \frac{end - start}{step} \right\rfloor + 1\) with values from `start` to `end` with step `step`.|
| `linspace`          | Creates aone-dimensional tensor of size `steps` whose values are evenly spaced from `start` to `end`, inclusive.|
| `logspace`          | Creates a one-dimensional tensor of size `steps` whose values are evenly spaced from \(\text{base}^{start}\) to \(\text{base}^{end}\), inclusive, on a logarithmic scale with base `base`.                                                                                     |
| `eye`               | Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. If `n` is the number of rows and `m` is the number of columns, then \(\mathbf{T}\) such that \(T[i, j] = 1\) if \(i = j\) and \(0\) otherwise.                                                                                                                                        |
| `empty`             | Returns a tensor filled with uninitialized data.                                                                                                                                                                                                                                                                                                        |
| `empty_like`        | Returns an uninitialized tensor with the same size as `input`.                                                                                                                                                                                                                                                                                          |
| `empty_strided`     | Creates a tensor with the specified `size` and `stride` and filled with undefined data.                                                                                                                                                                                                                                                                 |
| `full`              | Creates a tensor of size `size` filled with `fill_value`.                                                                                                                                                                                                                                                                                               |
| `full_like`         | Returns a tensor with the same size as `input` filled with `fill_value`.                                                                                                                                                                                                                                                                                |
| `quantize_per_tensor` | Converts a float tensor to a quantized tensor with given `scale` and `zero point`.                                                                                                                                                                                                                                                                   |
| `quantize_per_channel`| Converts a float tensor to a per-channel quantized tensor with given `scales` and `zero points`.                                                                                                                                                                                                                                                      |
| `dequantize`        | Returns an `fp32` tensor by dequantizing a quantized tensor.                                                                                                                                                                                                                                                                                            |
| `complex`           | Constructs a complex tensor with its real part equal to `real` and its imaginary part equal to `imag`.                                                                                                                                                                                                                                                  |
| `polar`             | Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value `abs` and angle `angle`.                                                                                                                                                                                                |
| `heaviside`         | Computes the Heaviside step function for each element in `input`. If \(\mathbf{T} = [t_1, t_2, \ldots, t_n]\), the Heaviside step function \(H(x)\) is defined as: \(H(x) = 0\) for \(x < 0\) and \(H(x) = 1\) for \(x \geq 0\). Therefore, \(H(T) = [H(t_1), H(t_2), \ldots, H(t_n)]\).                                                                    |

This table captures the essence of each function and provides students with a clear understanding of the variety of tensor creation and manipulation options available in PyTorch.
### PyTorch Tensor Operations and Random Sampling Functions

This section provides an overview of various PyTorch tensor operations and random sampling functions. Each function is briefly described to highlight its purpose and behavior. 

#### Tensor Operations

1. **adjoint**: Returns a view of the tensor conjugated and with the last two dimensions transposed.
2. **argwhere**: Returns a tensor containing the indices of all non-zero elements of the input.
3. **cat**: Concatenates the given sequence of tensors along the specified dimension.
4. **concat**: Alias of `torch.cat()`.
5. **conj**: Returns a view of the input with a flipped conjugate bit.
6. **chunk**: Attempts to split a tensor into the specified number of chunks.
7. **dsplit**: Splits a 3D tensor or higher into multiple tensors depthwise according to indices or sections.
8. **column_stack**: Horizontally stacks the tensors.
9. **dstack**: Stacks tensors depthwise (along the third axis).
10. **gather**: Gathers values along an axis specified by dimension.
11. **hsplit**: Splits a tensor horizontally according to indices or sections.
12. **hstack**: Stacks tensors horizontally (column-wise).
13. **index_add**: Adds tensor values at specified indices along a dimension.
14. **index_copy**: Copies tensor values at specified indices along a dimension.
15. **index_reduce**: Reduces tensor values at specified indices along a dimension using a specified reduction operation.
16. **index_select**: Returns a new tensor which indexes the input tensor along a specified dimension using given indices.
17. **masked_select**: Returns a 1-D tensor with elements selected according to a boolean mask.
18. **movedim**: Moves dimensions of a tensor from source to destination.
19. **moveaxis**: Alias for `torch.movedim()`.
20. **narrow**: Returns a new tensor that is a narrowed version of the input tensor.
21. **narrow_copy**: Same as `narrow()`, but returns a copy rather than shared storage.
22. **nonzero**: Returns a tensor of indices of non-zero elements.
23. **permute**: Returns a view of the tensor with its dimensions permuted.
24. **reshape**: Returns a tensor with the same data and number of elements but in a new shape.
25. **row_stack**: Alias of `torch.vstack()`.
26. **select**: Slices the tensor along the selected dimension at the given index.
27. **scatter**: Out-of-place version of `torch.Tensor.scatter_()`.
28. **diagonal_scatter**: Embeds the values of the source tensor along the diagonal elements of the input tensor.
29. **select_scatter**: Embeds the values of the source tensor at the given index.
30. **slice_scatter**: Embeds the values of the source tensor at the specified dimension.
31. **scatter_add**: Out-of-place version of `torch.Tensor.scatter_add_()`.
32. **scatter_reduce**: Out-of-place version of `torch.Tensor.scatter_reduce_()`.
33. **split**: Splits the tensor into chunks.
34. **squeeze**: Removes dimensions of size 1 from the tensor.
35. **stack**: Concatenates a sequence of tensors along a new dimension.
36. **swapaxes**: Alias for `torch.transpose()`.
37. **swapdims**: Alias for `torch.transpose()`.
38. **t**: Transposes dimensions 0 and 1 for 2-D tensors.
39. **take**: Returns a new tensor with elements from input at given indices.
40. **take_along_dim**: Selects values from input at 1-D indices along a specified dimension.
41. **tensor_split**: Splits a tensor into multiple sub-tensors along a dimension.
42. **tile**: Constructs a tensor by repeating the elements of the input tensor.
43. **transpose**: Returns a transposed version of the input tensor.
44. **unbind**: Removes a tensor dimension.
45. **unravel_index**: Converts a tensor of flat indices into coordinate tensors for a specified shape.
46. **unsqueeze**: Inserts a dimension of size 1 at a specified position.
47. **vsplit**: Splits a tensor vertically according to indices or sections.
48. **vstack**: Stacks tensors vertically (row-wise).
49. **where**: Returns a tensor of elements selected from input or other tensors based on a condition.

#### Generators

1. **Generator**: Creates and returns a generator object that manages the state of the pseudo-random number generator.

#### Random Sampling

1. **seed**: Sets the seed for generating random numbers to a non-deterministic random number on all devices.
2. **manual_seed**: Sets the seed for generating random numbers on all devices.
3. **initial_seed**: Returns the initial seed for generating random numbers as a Python long.
4. **get_rng_state**: Returns the random number generator state as a `torch.ByteTensor`.
5. **set_rng_state**: Sets the random number generator state.
6. **torch.default_generator**: Returns the default CPU `torch.Generator`.

##### Random Sampling Functions

1. **bernoulli**: Draws binary random numbers (0 or 1) from a Bernoulli distribution.
2. **multinomial**: Samples indices from a multinomial probability distribution.
3. **normal**: Returns a tensor of random numbers drawn from normal distributions with specified mean and standard deviation.
4. **poisson**: Samples each element from a Poisson distribution with a given rate parameter.
5. **rand**: Returns a tensor of random numbers from a uniform distribution on the interval \([0, 1)\).
6. **rand_like**: Returns a tensor with the same size as the input, filled with random numbers from a uniform distribution on the interval \([0, 1)\).
7. **randint**: Returns a tensor of random integers generated uniformly between specified low (inclusive) and high (exclusive) values.
8. **randint_like**: Returns a tensor with the same shape as the input, filled with random integers generated uniformly between low (inclusive) and high (exclusive).
9. **randn**: Returns a tensor filled with random numbers from a standard normal distribution (mean 0, variance 1).
10. **randn_like**: Returns a tensor with the same size as the input, filled with random numbers from a standard normal distribution.
11. **randperm**: Returns a random permutation of integers from 0 to \(n-1\).

##### In-Place Random Sampling Functions

1. **torch.Tensor.bernoulli_()**: In-place version of `torch.bernoulli()`.
2. **torch.Tensor.cauchy_()**: Samples from the Cauchy distribution.
3. **torch.Tensor.exponential_()**: Samples from the exponential distribution.
4. **torch.Tensor.geometric_()**: Samples from the geometric distribution.
5. **torch.Tensor.log_normal_()**: Samples from the log-normal distribution.
6. **torch.Tensor.normal_()**: In-place version of `torch.normal()`.
7. **torch.Tensor.random_()**: Samples from the discrete uniform distribution.
8. **torch.Tensor.uniform_()**: Samples from the continuous uniform distribution.

##### Quasi-Random Sampling

1. **quasirandom.SobolEngine**: An engine for generating (scrambled) Sobol sequences.

#### Serialization

1. **save**: Saves an object to a disk file.
2. **load**: Loads an object saved with `torch.save()` from a file.

#### Parallelism

1. **get_num_threads**: Returns the number of threads used for parallelizing CPU operations.
2. **set_num_threads**: Sets the number of threads used for intraop parallelism on CPU.
3. **get_num_interop_threads**: Returns the number of threads used for inter-op parallelism on CPU.
4. **set_num_interop_threads**: Sets the number of threads used for interop parallelism on CPU.

#### Locally Disabling Gradient Computation

1. **no_grad**: Context-manager that disables gradient calculation.
2. **enable_grad**: Context-manager that enables gradient calculation.
3. **autograd.grad_mode.set_grad_enabled**: Context-manager that sets gradient calculation on or off.
4. **is_grad_enabled**: Returns `True` if grad mode is currently enabled.
5. **autograd.grad_mode.inference_mode**: Context-manager that enables or disables inference mode.
6. **is_inference_mode_enabled**: Returns `True` if inference mode is currently enabled.

#### Math Operations

##### Pointwise Ops

1. **abs**: Computes the absolute value of each element.
2. **absolute**: Alias for `torch.abs()`.
3. **acos**: Computes the inverse cosine of each element.
4. **arccos**: Alias for `torch.acos()`.
5. **acosh**: Computes the inverse hyperbolic cosine of each element.
6. **arccosh**: Alias for `torch.acosh()`.
7. **add**: Adds other, scaled by alpha, to the input.
8. **addcdiv**: Element-wise division of tensor1 by tensor2, multiplies the result by a scalar, and adds it to the input.
9. **addcmul**: Element-wise multiplication of tensor1 by tensor2, multiplies the result by a scalar, and adds it to the input.
10. **angle**: Computes the element-wise angle (or phase) of a complex tensor.
11. **asin**: Computes the inverse sine of each element.
12. **arcsin**: Alias for `torch.asin()`.
13. **asinh**: Computes the inverse hyperbolic sine of each element.
14. **arcsinh**: Alias for `torch.asinh()`.
15. **atan**: Computes the inverse tangent of each element.
16. **arctan**: Alias for `torch.atan()`.
17. **atan2**: Computes the element-wise inverse tangent of input divided by other.
18. **arctan2**: Alias for `torch.atan2()`.
19. **atanh**: Computes the inverse hyperbolic tangent of each element.
20. **arctanh**: Alias for `torch.atanh()`.
21. **bitwise_and**: Computes the bitwise AND of the input tensors.
22. **bitwise_not**: Computes the bitwise NOT of the input tensor.
23. **bitwise_or**: Computes the bitwise OR of the input tensors.
24. **bitwise_xor**: Computes the bitwise XOR of the input tensors.
25. **ceil**: Returns a new tensor with the ceiling of the elements of input, element-wise.
26. **conj_physical**: Returns a view of the tensor with a flipped conjugate bit.
27. **copysign**: Creates a new tensor with the magnitude of input and the sign of other, element-wise.
28. **cos**: Returns a new tensor with the cosine of the elements of input.
29. **cosh**: Returns a new tensor with the hyperbolic cosine of the elements of input.
30. **count_nonzero**: Counts the number of non-zero elements in the input tensor.
31. **deg2rad**: Converts each element from degrees to radians.
32. **dequantize**: Given a quantized Tensor, returns an FP32 Tensor with the dequantized values.
33. **digamma**: Computes the logarithmic derivative of the gamma function on each element.
34. **div**: Divides each element of the input tensor by the corresponding element of other.
35. **divide**: Alias for `torch.div()`.
36. **dot**: Computes the dot product of two 1-D tensors.
37. **einstein_sum**: Computes the Einstein summation convention.
38. **eq**: Computes the element-wise equality of two tensors.
39. **equal**: Checks if two tensors have the same size and elements.
40. **exp**: Computes the exponential of each element.
41. **expm1**: Returns a new tensor with the exponential of the elements minus 1.
42. **float_power**: Raises each element of the input tensor to the power of the exponent.
43. **floor**: Returns a new tensor with the floor of the elements of input.
44. **floor_divide**: Computes the quotient of element-wise division of input by other, with floor division.
45. **fmod**: Computes the element-wise remainder of division.
46. **frac**: Computes the fractional portion of each element in the input tensor.
47. **frexp**: Decomposes each element of input into mantissa and exponent.
48. **gcd**: Computes the greatest common divisor of each element.
49. **greater**: Alias for `torch.gt()`.
50. **greater_equal**: Alias for `torch.ge()`.
51. **gt**: Computes the element-wise greater than comparison.
52. **ge**: Computes the element-wise greater than or equal comparison.
53. **hypot**: Computes the hypotenuse of a right-angled triangle.
54. **igamma**: Computes the lower incomplete gamma function of each element.
55. **igammac**: Computes the upper incomplete gamma function of each element.
56. **i0**: Computes the zeroth order modified Bessel function of the first kind of each element.
57. **imag**: Returns the imaginary part of the complex tensor.
58. **isclose**: Returns a tensor of boolean values indicating whether each element of input is close to the corresponding element of other.
59. **isfinite**: Returns a tensor of boolean values indicating whether each element of input is finite.
60. **isinf**: Returns a tensor of boolean values indicating whether each element of input is infinite.
61. **isnan**: Returns a tensor of boolean values indicating whether each element of input is NaN.
62. **isneginf**: Returns a tensor of boolean values indicating whether each element of input is negative infinity.
63. **isposinf**: Returns a tensor of boolean values indicating whether each element of input is positive infinity.
64. **isreal**: Returns a tensor of boolean values indicating whether each element of input is real.
65. **kron**: Computes the Kronecker product of two tensors.
66. **lcm**: Computes the least common multiple of each element.
67. **ldexp**: Multiplies input by 2 raised to the power of the exponent, element-wise.
68. **lerp**: Performs a linear interpolation of two tensors based on a weight.
69. **less**: Alias for `torch.lt()`.
70. **less_equal**: Alias for `torch.le()`.
71. **lgamma**: Computes the logarithm of the gamma function.
72. **log**: Computes the natural logarithm of each element.
73. **log10**: Computes the base 10 logarithm of each element.
74. **log1p**: Computes the natural logarithm of one plus each element.
75. **log2**: Computes the base 2 logarithm of each element.
76. **logaddexp**: Computes the logarithm of the sum of exponentiations of the inputs.
77. **logaddexp2**: Computes the logarithm of the sum of exponentiations of the inputs, base-2.
78. **logical_and**: Computes the element-wise logical AND.
79. **logical_not**: Computes the element-wise logical NOT.
80. **logical_or**: Computes the element-wise logical OR.
81. **logical_xor**: Computes the element-wise logical XOR.
82. **lt**: Computes the element-wise less than comparison.
83. **le**: Computes the element-wise less than or equal comparison.
84. **masked_fill**: Replaces elements of the input tensor with a value where a mask is True.
85. **masked_scatter**: Copies elements from the source tensor into the input tensor at positions where the mask is True.
86. **matmul**: Performs matrix multiplication of two tensors.
87. **max**: Computes the maximum value of the input tensor.
88. **maximum**: Computes the element-wise maximum of the input tensors.
89. **mean**: Computes the mean of the elements of the input tensor.
90. **median**: Computes the median of the elements of the input tensor.
91. **min**: Computes the minimum value of the input tensor.
92. **minimum**: Computes the element-wise minimum of the input tensors.
93. **mul**: Multiplies each element of the input tensor by the corresponding element of other.
94. **multiply**: Alias for `torch.mul()`.
95. **mvlgamma**: Computes the multivariate logarithm of the gamma function.
96. **nan_to_num**: Replaces NaN, positive infinity, and negative infinity with numerical values.
97. **neg**: Computes the negative of each element.
98. **negative**: Alias for `torch.neg()`.
99. **nextafter**: Returns the next floating-point value after the elements of input towards other.
100. **not_equal**: Alias for `torch.ne()`.
101. **ne**: Computes the element-wise inequality of two tensors.
102. **positive**: Alias for `torch.positive()`.
103. **pow**: Computes the element-wise power of input by exponent.
104. **quantile**: Computes the quantile of the elements of the input tensor.
105. **rad2deg**: Converts each element from radians to degrees.
106. **real**: Returns the real part of the complex tensor.
107. **reciprocal**: Computes the reciprocal of each element.
108. **remainder**: Computes the element-wise remainder of division.
109. **renorm**: Renormalizes a tensor along a specified dimension.
110. **round**: Rounds each element to the nearest integer.
111. **rsqrt**: Computes the reciprocal of the square root of each element.
112. **sgn**: Computes the sign of each element.
113. **sign**: Alias for `torch.sgn()`.
114. **signbit**: Returns a tensor of boolean values indicating whether each element of input has its sign bit set.
115. **sin**: Computes the sine of each element.
116. **sinh**: Computes the hyperbolic sine of each element.
117. **sqrt**: Computes the square root of each element.
118. **square**: Computes the square of each element.
119. **sub**: Subtracts each element of the other tensor from the input tensor.
120. **subtract**: Alias for `torch.sub()`.
121. **tan**: Computes the tangent of each element.
122. **tanh**: Computes the hyperbolic tangent of each element.
123. **true_divide**: Computes the true element-wise division.
124. **trunc**: Returns a tensor with truncated integer values of the elements.




| Operation            | Description                                                                                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **allclose**         | This function checks if input and other satisfy the condition.                                                                                               |
| **argsort**          | Returns the indices that sort a tensor along a given dimension in ascending order by value.                                                                  |
| **eq**               | Computes element-wise equality.                                                                                                                              |
| **equal**            | True if two tensors have the same size and elements, False otherwise.                                                                                        |
| **ge**               | Computes `input >= other` element-wise.                                                                                                                      |
| **greater_equal**    | Alias for `torch.ge()`.                                                                                                                                      |
| **gt**               | Computes `input > other` element-wise.                                                                                                                       |
| **greater**          | Alias for `torch.gt()`.                                                                                                                                      |
| **isclose**          | Returns a new tensor with boolean elements representing if each element of `input` is "close" to the corresponding element of `other`.                       |
| **isfinite**         | Returns a new tensor with boolean elements representing if each element is finite or not.                                                                    |
| **isin**             | Tests if each element of `elements` is in `test_elements`.                                                                                                   |
| **isinf**            | Tests if each element of `input` is infinite (positive or negative infinity) or not.                                                                         |
| **isposinf**         | Tests if each element of `input` is positive infinity or not.                                                                                                |
| **isneginf**         | Tests if each element of `input` is negative infinity or not.                                                                                                |
| **isnan**            | Returns a new tensor with boolean elements representing if each element of `input` is NaN or not.                                                            |
| **isreal**           | Returns a new tensor with boolean elements representing if each element of `input` is real-valued or not.                                                    |
| **kthvalue**         | Returns a namedtuple (values, indices) where values is the k-th smallest element of each row of the input tensor in the given dimension `dim`.                |
| **le**               | Computes `input <= other` element-wise.                                                                                                                      |
| **less_equal**       | Alias for `torch.le()`.                                                                                                                                      |
| **lt**               | Computes `input < other` element-wise.                                                                                                                       |
| **less**             | Alias for `torch.lt()`.                                                                                                                                      |
| **maximum**          | Computes the element-wise maximum of `input` and `other`.                                                                                                    |
| **minimum**          | Computes the element-wise minimum of `input` and `other`.                                                                                                    |
| **fmax**             | Computes the element-wise maximum of `input` and `other`.                                                                                                    |
| **fmin**             | Computes the element-wise minimum of `input` and `other`.                                                                                                    |
| **ne**               | Computes `input != other` element-wise.                                                                                                                      |
| **not_equal**        | Alias for `torch.ne()`.                                                                                                                                      |
| **sort**             | Sorts the elements of the input tensor along a given dimension in ascending order by value.                                                                  |
| **topk**             | Returns the k largest elements of the given input tensor along a given dimension.                                                                            |
| **msort**            | Sorts the elements of the input tensor along its first dimension in ascending order by value.                                                                |
| **Spectral Ops**     |                                                                                                                                                              |
| **stft**             | Short-time Fourier transform (STFT).                                                                                                                         |
| **istft**            | Inverse short time Fourier Transform.                                                                                                                        |
| **bartlett_window**  | Bartlett window function.                                                                                                                                    |
| **blackman_window**  | Blackman window function.                                                                                                                                    |
| **hamming_window**   | Hamming window function.                                                                                                                                     |
| **hann_window**      | Hann window function.                                                                                                                                        |
| **kaiser_window**    | Computes the Kaiser window with window length `window_length` and shape parameter `beta`.                                                                    |
| **Other Operations** |                                                                                                                                                              |
| **atleast_1d**       | Returns a 1-dimensional view of each input tensor with zero dimensions.                                                                                      |
| **atleast_2d**       | Returns a 2-dimensional view of each input tensor with zero dimensions.                                                                                      |
| **atleast_3d**       | Returns a 3-dimensional view of each input tensor with zero dimensions.                                                                                      |
| **bincount**         | Count the frequency of each value in an array of non-negative ints.                                                                                          |
| **block_diag**       | Create a block diagonal matrix from provided tensors.                                                                                                        |
| **broadcast_tensors**| Broadcasts the given tensors according to Broadcasting semantics.                                                                                           |
| **broadcast_to**     | Broadcasts `input` to the shape `shape`.                                                                                                                     |
| **broadcast_shapes** | Similar to `broadcast_tensors()` but for shapes.                                                                                                             |
| **bucketize**        | Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by `boundaries`.                    |
| **cartesian_prod**   | Do cartesian product of the given sequence of tensors.                                                                                                       |
| **cdist**            | Computes batched the p-norm distance between each pair of the two collections of row vectors.                                                                |
| **clone**            | Returns a copy of `input`.                                                                                                                                   |
| **combinations**     | Compute combinations of length `r` of the given tensor.                                                                                                      |
| **corrcoef**         | Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix, where rows are the variables and columns are the observations. |
| **cov**              | Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.                    |
| **cross**            | Returns the cross product of vectors in dimension `dim` of `input` and `other`.                                                                              |
| **cummax**           | Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of `input` in the dimension `dim`.                                  |
| **cummin**           | Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of `input` in the dimension `dim`.                                  |
| **cumprod**          | Returns the cumulative product of elements of `input` in the dimension `dim`.                                                                                |
| **cumsum**           | Returns the cumulative sum of elements of `input` in the dimension `dim`.                                                                                    |
| **diag**             | If `input` is a vector (1-D tensor), then returns a 2-D square tensor.                                                                                       |
| **diag_embed**       | Creates a tensor whose diagonals of certain 2D planes (specified by `dim1` and `dim2`) are filled by `input`.                                                |
| **diagflat**         | If `input` is a vector (1-D tensor), then returns a 2-D square tensor.                                                                                       |
| **diagonal**         | Returns a partial view of `input` with its diagonal elements with respect to `dim1` and `dim2` appended as a dimension at the end of the shape.               |
| **diff**             | Computes the n-th forward difference along the given dimension.                                                                                              |
| **einsum**           | Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.                |
| **flatten**          | Flattens `input` by reshaping it into a one-dimensional tensor.                                                                                              |
| **flip**             | Reverse the order of an n-D tensor along given axis in `dims`.                                                                                               |
| **fliplr**           | Flip tensor in the left/right direction, returning a new tensor.                                                                                            |
| **flipud**           | Flip tensor in the up/down direction, returning a new tensor.                                                                                               |
| **kron**             | Computes the Kronecker product, denoted by `⊗`, of `input` and `other`.                                                                                      |
| **rot90**            | Rotate an n-D tensor by 90 degrees in the plane specified by `dims` axis.                                                                                   |
| **gcd**              | Computes the element-wise greatest common divisor (GCD) of `input` and `other`.                                                                              |
| **histc**            | Computes the histogram of a tensor.                                                                                                                          |
| **histogram**        | Computes a histogram of the values in a tensor.                                                                                                              |
| **histogramdd**      | Computes a multi-dimensional histogram of the values in a tensor.                                                                                            |
| **meshgrid**         | Creates grids of coordinates specified by the 1D inputs in `attr:tensors`.                                                                                   |
| **lcm**              | Computes the element-wise least common multiple (LCM) of `input` and `other`.                                                                                |
| **logcumsumexp**     | Returns the logarithm of the cumulative summation of the exponentiation of elements of `input` in the dimension `dim`.                                       |
| **ravel**            | Return a contiguous flattened tensor.                                                                                                                        |
| **renorm**           | Returns a tensor where each sub-tensor of `input` along dimension `dim` is normalized such that the p-norm of the sub-tensor is lower than the value `maxnorm`.|
| **repeat_interleave**| Repeat elements of a tensor.                                                                                                                                 |
| **roll**             | Roll the tensor `input` along the given dimension(s).                                                                                                         |
| **searchsorted**     | Find the indices from the innermost dimension of `sorted_sequence` such that, if the corresponding values in `values` were

 inserted before the indices, the order of the respective innermost dimension within `sorted_sequence` would be preserved. |
| **trapz**            | Estimates the integral of `input` along `dim` using the trapezoid rule.                                                                                       |
| **tril_indices**     | Returns the indices of the lower triangular part of a row-by-row input tensor.                                                                               |
| **triu_indices**     | Returns the indices of the upper triangular part of a row-by-row input tensor.                                                                               |
| **unflatten**        | Expands the dimension of a tensor to the given named dimensions.                                                                                              |
| **vander**           | Generates a Vandermonde matrix.                                                                                                                              |
| **view_as_real**     | Returns a view of `input` as a real tensor.                                                                                                                  |
| **view_as_complex**  | Returns a view of `input` as a complex tensor.                                                                                                               |
| **view_as_channels** | Returns a view of `input` as a channel tensor.                                                                                                               |

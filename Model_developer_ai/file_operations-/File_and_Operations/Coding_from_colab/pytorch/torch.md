>>> x = torch.tensor([1, 2, 3])
- torch.is_tensor(x)
- torch.is_complex(input)
- torch.is_storage
- torch.is_conj(input)
- torch.is_floating_point(input)
- torch.is_nonzero(input)
- torch.set_default_dtype(torch.float64)
- torch.get_default_dtype() 
- torch.set_default_device(device)
  - torch.tensor([1.2, 3]).device
     torch.set_default_device('cuda')  #      current device is 0
     torch.tensor([1.2, 3]).device
     torch.set_default_device('cuda:1')
     torch.tensor([1.2, 3]).device


- torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
torch.set_default_tensor_type(torch.DoubleTensor)
torch.tensor([1.2, 3]).dtype    # a new floating point tensor

- a = torch.randn(1, 2, 3, 4, 5)
torch.numel(a)
- a = torch.zeros(4,4)
torch.numel(a)

-  Limit the precision of elements
torch.set_printoptions(precision=2)
torch.tensor([1.12345])
-  Limit the number of elements shown
torch.set_printoptions(threshold=5)
torch.arange(10)
- Restore defaults
torch.set_printoptions(profile='default')
torch.tensor([1.12345])
torch.arange(10)


- torch.set_flush_denormal

- torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)


torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])

torch.tensor([0, 1])  # Type inference on data

torch.tensor([[0.11111, 0.222222, 0.3333333]],
             dtype=torch.float64,
             device=torch.device('cuda:0'))  # creates a double tensor on a CUDA device

torch.tensor(3.14159)  # Create a zero-dimensional (scalar) tensor

torch.tensor([])  # Create an empty tensor (of size (0,))


- torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False, check_invariants=None, is_coalesced=None) → Tensor




>>> crow_indices = [0, 2, 4]
>>> col_indices = [0, 1, 0, 1]
>>> values = [1, 2, 3, 4]
>>> torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
...                         torch.tensor(col_indices, dtype=torch.int64),
...                         torch.tensor(values), dtype=torch.double)
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
       dtype=torch.float64, layout=torch.sparse_csr)

torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=None, *, dtype=None, device=None, requires_grad=False, check_invariants=None) → Tensor


torch.asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False) 
- torch.from_file(filename, shared=None, size=0, *, dtype=None, layout=None, device=None, pin_memory=False)


t = torch.randn(2, 5, dtype=torch.float64)
t.numpy().tofile('storage.pt')
t_mapped = torch.from_file('storage.pt', shared=False, size=10, dtype=torch.float64)

- torch.from_numpy(ndarray) 
- torch.from_dlpack(ext_tensor)
- torch.frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False) 


- torch.zeros(2, 3)

- torch.zeros(5)

- torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)


- torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

- torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) 

- torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

- torch.range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor


- torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

- torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)


- torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

- torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format)

- torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)


- torch.full((2, 3), 3.141592)

- torch.full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) →


- real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
z = torch.complex(real, imag)
z
z.dtype

- import numpy as np
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
z = torch.polar(abs, angle)
z

- x = torch.arange(4, dtype=torch.float)
A = torch.complex(x, x).reshape(2, 2)
A
A.adjoint()
(A.adjoint() == A.mH).all()


- x = torch.randn(2, 3)
x
torch.cat((x, x, x), 0)
torch.cat((x, x, x), 1)

- torch.chunk(input, chunks, dim=0) 


- t = torch.arange(16.0).reshape(2, 2, 4)
t
torch.dsplit(t, 2)a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.column_stack((a, b))
a = torch.arange(5)
b = torch.arange(10).reshape(5, 2)
torch.column_stack((a, b, b))

- a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.dstack((a,b))
a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[4],[5],[6]])
torch.dstack((a,b))

t = torch.arange(16.0).reshape(4,4)
t
torch.hsplit(t, 2)
torch.hsplit(t, [3, 6])

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.hstack((a,b))
a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[4],[5],[6]])
torch.hstack((a,b))


x = torch.randn(3, 4)
x
indices = torch.tensor([0, 2])
torch.index_select(x, 0, indices)
torch.index_select(x, 1, indices)


x = torch.randn(3, 4)
x
indices = torch.tensor([0, 2])
torch.index_select(x, 0, indices)
torch.index_select(x, 1, indices)


x = torch.randn(3, 4)
x
mask = x.ge(0.5)
mask
torch.masked_select(x, mask)

x = torch.randn(2, 3, 5)
x.size()
torch.permute(x, (2, 0, 1)).size()

a = torch.arange(4.)
torch.reshape(a, (2, 2))
b = torch.tensor([[0, 1], [2, 3]])
torch.reshape(b, (-1,))


a = torch.arange(10).reshape(5, 2)
a
torch.split(a, 2)
torch.split(a, [1, 4])
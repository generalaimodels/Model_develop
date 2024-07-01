## `a`

| Functionality Name        | Sample Snippet                                   |
|---------------------------|--------------------------------------------------|
| abs                       | ```torch.abs(tensor)```                          |
| abs_                      | ```tensor.abs_()```                              |
| absolute                  | ```torch.absolute(tensor)```                     |
| acos                      | ```torch.acos(tensor)```                         |
| acos_                     | ```tensor.acos_()```                             |
| acosh                     | ```torch.acosh(tensor)```                        |
| acosh_                    | ```tensor.acosh_()```                            |
| adaptive_avg_pool1d       | ```torch.nn.functional.adaptive_avg_pool1d(input, output_size)``` |
| adaptive_max_pool1d       | ```torch.nn.functional.adaptive_max_pool1d(input, output_size)``` |
| add                       | ```torch.add(tensor1, tensor2)```                |
| addbmm                    | ```torch.addbmm(tensor1, batch1, batch2, beta=1, alpha=1)``` |
| addcdiv                   | ```torch.addcdiv(tensor1, tensor2, tensor3, value=1)``` |
| addcmul                   | ```torch.addcmul(tensor1, tensor2, tensor3, value=1)``` |
| addmm                     | ```torch.addmm(tensor1, mat1, mat2, beta=1, alpha=1)``` |
| addmv                     | ```torch.addmv(tensor1, mat, vec, beta=1, alpha=1)``` |
| addmv_                    | ```tensor.addmv_(mat, vec, beta=1, alpha=1)```   |
| addr                      | ```torch.addr(tensor1, vec1, vec2, beta=1, alpha=1)``` |
| adjoint                   | ```torch.adjoint(input)```                       |
| affine_grid_generator     | ```torch.nn.functional.affine_grid(theta, size, align_corners=False)``` |
| alias_copy                | ```torch.alias_copy(tensor)```                   |
| align_tensors             | ```torch.align_tensors(tensors)```               |
| all                       | ```torch.all(tensor)```                          |
| allclose                  | ```torch.allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, equal_nan=False)``` |
| alpha_dropout             | ```torch.nn.functional.alpha_dropout(input, p, training=False, inplace=False)``` |
| alpha_dropout_            | ```tensor.alpha_dropout_(p, trainig=False)```    |
| amax                      | ```torch.amax(tensor)```                         |
| amin                      | ```torch.amin(tensor)```                         |
| aminmax                   | ```torch.aminmax(tensor)```                      |
| amp                       | ```torch.amp```                                  |
| angle                     | ```torch.angle(tensor)```                        |
| any                       | ```torch.any(tensor)```                          |
| ao                        | ```torch.ao()```                                 |
| arange                    | ```torch.arange(start=0, end, step=1, dtype=None, layout=torch.strided, device=None, requires_grad=False)``` |
| arccos                    | ```torch.arccos(tensor)```                       |
| arccos_                   | ```tensor.arccos_()```                           |
| arccosh                   | ```torch.arccosh(tensor)```                      |
| arccosh_                  | ```tensor.arccosh_()```                          |
| arcsin                    | ```torch.arcsin(tensor)```                       |
| arcsin_                   | ```tensor.arcsin_()```                           |
| arcsinh                   | ```torch.arcsinh(tensor)```                      |
| arcsinh_                  | ```tensor.arcsinh_()```                          |
| arctan                    | ```torch.arctan(tensor)```                       |
| arctan2                   | ```torch.arctan2(tensor1, tensor2)```             |
| arctan_                   | ```tensor.arctan_()```                           |
| arctanh                   | ```torch.arctanh(tensor)```                      |
| arctanh_                  | ```tensor.arctanh_()```                          |
| are_deterministic_algorithms_enabled | ```torch.are_deterministic_algorithms_enabled()``` |
| argmax                    | ```torch.argmax(input, dim, keepdim=False)```     |
| argmin                    | ```torch.argmin(input, dim, keepdim=False)```     |
| argsort                   | ```torch.argsort(input, dim=-1, descending=False)``` |
| argwhere                  | ```torch.argwhere(input)```                      |
| as_strided                | ```torch.as_strided(input, size, stride, storage_offset=0)``` |
| as_strided_               | ```tensor.as_strided_(size, stride, storage_offset=0)``` |
| as_strided_copy           | ```torch.as_strided_copy(input, size, stride, storage_offset=0)``` |
| as_strided_scatter        | ```torch.as_strided_scatter(input, size, stride, storage_offset=0)``` |
| as_tensor                 | ```torch.as_tensor(data, dtype=None, device=None)``` |
| asarray                   | ```torch.asarray(data, dtype=None, device=None)``` |
| asin                      | ```torch.asin(tensor)```                         |
| asin_                     | ```tensor.asin_()```                             |
| asinh                     | ```torch.asinh(tensor)```                        |
| asinh_                    | ```tensor.asinh_()```                            |
| atan                      | ```torch.atan(tensor)```                         |
| atan2                     | ```torch.atan2(tensor1, tensor2)```              |
| atan_                     | ```tensor.atan_()```                             |
| atanh                     | ```torch.atanh(tensor)```                        |
| atanh_                    | ```tensor.atanh_()```                            |
| atleast_1d                | ```torch.atleast_1d(tensor)```                   |
| atleast_2d                | ```torch.atleast_2d(tensor)```                   |
| atleast_3d                | ```torch.atleast_3d(tensor)```                   |
| attr                      | ```torch.attr```                                 |
| autocast                  | ```torch.autocast()```                           |
| autocast_decrement_nesting | ```torch.autocast_decrement_nesting()```         |
| autocast_increment_nesting | ```torch.autocast_increment_nesting()```         |
| autograd                  | ```torch.autograd```                             |
| avg_pool1d                | ```torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)``` |

## `b`



| Functionality Name                   | Sample Snippet                                         |
|--------------------------------------|--------------------------------------------------------|
| backends                             | ```torch.backends.cuda.is_available()```               |
| baddbmm                              | ```torch.baddbmm(tensor1, batch1, batch2, beta=1, alpha=1)``` |
| bartlett_window                      | ```torch.bartlett_window(window_length, periodic=True, alpha=0.5, dtype=torch.float32, device=None)``` |
| base_py_dll_path                     | ```torch.base_py_dll_path()```                         |
| batch_norm                           | ```torch.nn.BatchNorm2d(num_features)```               |
| batch_norm_backward_elemt            | ```torch.batch_norm_backward_elemt(grad_output, input, weight, running_mean, running_var, save_mean, save_var, epsilon, output)``` |
| batch_norm_backward_reduce           | ```torch.batch_norm_backward_reduce(grad_output, input, weight, running_mean, running_var, save_mean, save_var, epsilon, output)``` |
| batch_norm_elemt                     | ```torch.batch_norm_elemt(input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)``` |
| batch_norm_gather_stats              | ```torch.batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count)``` |
| batch_norm_gather_stats_with_counts  | ```torch.batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts)``` |
| batch_norm_stats                     | ```torch.batch_norm_stats(input, eps)```               |
| batch_norm_update_stats              | ```torch.batch_norm_update_stats(input, running_mean, running_var, momentum)``` |
| bernoulli                            | ```torch.bernoulli(tensor, *, generator=None, out=None)``` |
| bfloat16                             | ```torch.bfloat16(tensor)```                           |
| bilinear                             | ```torch.nn.functional.bilinear(input1, input2, weight, bias=None)``` |
| binary_cross_entropy_with_logits     | ```torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)``` |
| bincount                             | ```torch.bincount(tensor, weights=None, minlength=0)``` |
| binomial                             | ```torch.binomial(tensor, p, generator=None)```        |
| bits16                               | ```torch.bits16(tensor)```                             |
| bits1x8                              | ```torch.bits1x8(tensor)```                            |
| bits2x4                              | ```torch.bits2x4(tensor)```                            |
| bits4x2                              | ```torch.bits4x2(tensor)```                            |
| bits8                                | ```torch.bits8(tensor)```                              |
| bitwise_and                          | ```torch.bitwise_and(tensor1, tensor2)```              |
| bitwise_left_shift                   | ```torch.bitwise_left_shift(tensor, shift)```          |
| bitwise_not                          | ```torch.bitwise_not(tensor)```                        |
| bitwise_or                           | ```torch.bitwise_or(tensor1, tensor2)```               |
| bitwise_right_shift                  | ```torch.bitwise_right_shift(tensor, shift)```         |
| bitwise_xor                          | ```torch.bitwise_xor(tensor1, tensor2)```              |
| blackman_window                      | ```torch.blackman_window(window_length, periodic=True, alpha=0.5, dtype=torch.float32, device=None)``` |
| block_diag                           | ```torch.block_diag(mats)```                           |
| bmm                                  | ```torch.bmm(tensor1, tensor2)```                      |
| bool                                 | ```torch.bool(tensor)```                               |
| broadcast_shapes                     | ```torch.broadcast_shapes(shape1, shape2)```           |
| broadcast_tensors                    | ```torch.broadcast_tensors(tensors)```                 |
| broadcast_to                         | ```torch.broadcast_to(input, shape)```                 |
| bucketize                            | ```torch.bucketize(input, bins, out=None, right=False)``` |

## `c`

| Functionality Name                 | Sample Snippet                                                     |
|------------------------------------|--------------------------------------------------------------------|
| can_cast                           | ```torch.can_cast(tensor, dtype)```                                |
| candidate                          | ```torch.candidate()```                                           |
| cartesian_prod                     | ```torch.cartesian_prod(tensor1, tensor2)```                       |
| cat                                | ```torch.cat(tensors, dim=0)```                                   |
| ccol_indices_copy                  | ```torch.ccol_indices_copy(input, index, source)```                |
| cdist                              | ```torch.cdist(x1, x2, p=2, compute_mode=None)```                 |
| cdouble                            | ```torch.cdouble(tensor)```                                       |
| ceil                               | ```torch.ceil(tensor)```                                          |
| ceil_                              | ```tensor.ceil_()```                                              |
| celu                               | ```torch.celu(input, alpha=1.0, inplace=False)```                 |
| celu_                              | ```torch.celu_(input, alpha=1.0)```                               |
| cfloat                             | ```torch.cfloat(tensor)```                                        |
| chain_matmul                       | ```torch.chain_matmul(tensor1, tensor2, tensor3, ...)```           |
| chalf                              | ```torch.chalf(tensor)```                                         |
| channel_shuffle                    | ```torch.nn.functional.channel_shuffle(input, groups)```          |
| channels_last                      | ```torch.channels_last(tensor)```                                 |
| channels_last_3d                   | ```torch.channels_last_3d(input)```                               |
| cholesky                           | ```torch.cholesky(input, upper=False, out=None)```                |
| cholesky_inverse                   | ```torch.cholesky_inverse(input, upper=False, out=None)```        |
| cholesky_solve                     | ```torch.cholesky_solve(input, input2, upper=False, out=None)```  |
| choose_qparams_optimized           | ```torch.quantization.choose_qparams_optimized(min_val, max_val, dtype, quant_type)``` |
| chunk                              | ```torch.chunk(tensor, chunks, dim=0)```                          |
| clamp                              | ```torch.clamp(tensor, min, max)```                               |
| clamp_                             | ```tensor.clamp_(min, max)```                                     |
| clamp_max                          | ```torch.clamp_max(tensor, max)```                                |
| clamp_max_                         | ```tensor.clamp_max_(max)```                                      |
| clamp_min                          | ```torch.clamp_min(tensor, min)```                                |
| clamp_min_                         | ```tensor.clamp_min_(min)```                                      |
| classes                            | ```torch.classes```                                               |
| classproperty                      | ```torch.classproperty(fget, fset=None, fdel=None, doc=None)```  |
| clear_autocast_cache               | ```torch.cuda.amp.clear_autocast_cache()```                       |
| clip                               | ```torch.clip(tensor, min, max, out=None)```                      |
| clip_                              | ```tensor.clip_(min, max)```                                      |
| clone                              | ```tensor.clone()```                                              |
| col_indices_copy                   | ```torch.col_indices_copy(input, index, source)```                |
| column_stack                       | ```torch.column_stack(tensors)```                                 |
| combinations                       | ```torch.combinations(input, r, with_replacement=False)```        |
| compile                            | ```torch.compile(string)```                                       |
| compiled_with_cxx11_abi            | ```torch.compiled_with_cxx11_abi()```                             |
| compiler                           | ```torch.compiler()```                                            |
| complex                            | ```torch.complex(real, imag)```                                   |
| complex128                         | ```torch.complex128(tensor)```                                    |
| complex32                          | ```torch.complex32(tensor)```                                     |
| complex64                          | ```torch.complex64(tensor)```                                     |
| concat                             | ```torch.concat(tensors, dim=0)```                                |
| concatenate                        | ```torch.cat(tensors, dim=0)```                                   |
| cond                               | ```torch.cond(pred, true_fn, false_fn)```                         |
| conj                               | ```torch.conj(input)```                                           |
| conj_physical                      | ```torch.conj_physical(input)```                                  |
| conj_physical_                     | ```torch.conj_physical_(input)```                                 |
| constant_pad_nd                    | ```torch.nn.functional.constant_pad_nd(input, pad, value)```      |
| contiguous_format                  | ```torch.contiguous_format(input)```                              |
| conv1d                             | ```torch.nn.Conv1d(in_channels, out_channels, kernel_size, ...)``` |
| conv2d                             | ```torch.nn.Conv2d(in_channels, out_channels, kernel_size, ...)``` |
| conv3d                             | ```torch.nn.Conv3d(in_channels, out_channels, kernel_size, ...)``` |
| conv_tbc                           | ```torch.conv_tbc(input, weight, bias, pad=0)```                  |
| conv_transpose1d                   | ```torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, ...)``` |
| conv_transpose2d                   | ```torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, ...)``` |
| conv_transpose3d                   | ```torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, ...)``` |
| convolution                        | ```torch.nn.functional.convolution(input, weight, bias=None, ...)``` |
| copysign                           | ```torch.copysign(tensor1, tensor2)```                           |
| corrcoef                           | ```torch.corrcoef(x, rowvar=True)```                             |
| cos                                | ```torch.cos(tensor)```                                           |
| cos_                               | ```tensor.cos_()```                                               |
| cosh                               | ```torch.cosh(tensor)```                                          |
| cosh_                              | ```tensor.cosh_()```                                              |
| cosine_embedding_loss              | ```torch.nn.functional.cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean')``` |
| cosine_similarity                  | ```torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)``` |
| count_nonzero                      | ```torch.count_nonzero(input, dim=None)```                       |
| cov                                | ```torch.cov(input, bias=False, rowvar=True, ...)```             |
| cpp                                | ```torch.cpp(filename)```                                         |
| cpu                                | ```torch.cpu()```                                                 |
| cross                              | ```torch.cross(input1, input2, dim=-1)```                        |
| crow_indices_copy                  | ```torch.crow_indices_copy(input, index, source)```               |
| ctc_loss                           | ```torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)``` |
| ctypes                             | ```torch.ctypes()```                                              |
| cuda                               | ```torch.cuda```                                                  |
| cuda_path                          | ```torch.cuda_path```                                             |
| cuda_version                       | ```torch.cuda_version```                                          |
| cudnn_affine_grid_generator        | ```torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)``` |
| cudnn_batch_norm                   | ```torch.nn.functional.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)``` |
| cudnn_convolution                  | ```torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)``` |
| cudnn_convolution_add_relu         | ```torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)``` |
| cudnn_convolution_relu             | ```torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)``` |
| cudnn_convolution_transpose        | ```torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)``` |
| cudnn_grid_sampler                 | ```torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)``` |
| cudnn_is_acceptable                | ```torch.backends.cudnn.is_acceptable(tensor)```                 |
| cummax                             | ```torch.cummax(tensor, dim, dtype=None)```                       |
| cummin                             | ```torch.cummin(tensor, dim, dtype=None)```                       |
| cumprod                            | ```torch.cumprod(tensor, dim, dtype=None)```                      |
| cumsum                             | ```torch.cumsum(tensor, dim, dtype=None)```                       |
| cumulative_trapezoid               | ```torch.cumulative_trapezoid(x, y, x0=None, dx=1.0, dim=-1)```   |

## `d`

| Functionality Name   | Sample Snippet                                 |
|----------------------|------------------------------------------------|
| default_generator    | ```torch.default_generator()```               |
| deg2rad              | ```torch.deg2rad(tensor)```                   |
| deg2rad_             | ```tensor.deg2rad_()```                       |
| dequantize           | ```torch.dequantize(tensor)```                |
| det                  | ```torch.det(input)```                        |
| detach               | ```tensor.detach()```                         |
| detach_              | ```tensor.detach_()```                        |
| detach_copy          | ```tensor.detach().clone()```                 |
| device               | ```tensor.device```                           |
| diag                 | ```torch.diag(tensor, diagonal=0)```          |
| diag_embed           | ```torch.diag_embed(tensor)```                |
| diagflat             | ```torch.diagflat(input, offset=0)```         |
| diagonal             | ```torch.diagonal(tensor, offset=0, dim1=0, dim2=1)``` |
| diagonal_copy        | ```torch.diagonal(tensor, offset=0, dim1=0, dim2=1).clone()``` |
| diagonal_scatter    | ```torch.diagonal_scatter(src, dim, index, out=None)``` |
| diff                 | ```torch.diff(tensor, n=1, axis=-1)```        |
| digamma              | ```torch.digamma(tensor)```                   |
| dist                 | ```torch.dist(tensor1, tensor2, p=2)```       |
| distributed          | ```torch.distributed```                       |
| distributions        | ```torch.distributions```                     |
| div                  | ```torch.div(tensor1, tensor2)```             |
| divide               | ```torch.divide(tensor1, tensor2)```          |
| dll                  | ```torch.dll```                               |
| dll_path             | ```torch.dll_path()```                        |
| dll_paths            | ```torch.dll_paths()```                       |
| dlls                 | ```torch.dlls()```                            |
| dot                  | ```torch.dot(tensor1, tensor2)```             |
| double               | ```torch.double(tensor)```                    |
| dropout              | ```torch.nn.Dropout(p=0.5, inplace=False)```  |
| dropout_             | ```torch.nn.functional.dropout(tensor, p=0.5, training=True, inplace=False)``` |
| dsmm                 | ```torch.dsmm(tensor1, tensor2)```            |
| dsplit               | ```torch.dsplit(tensor, indices_or_sections)```|
| dstack               | ```torch.dstack(tensors)```                   |
| dtype                | ```tensor.dtype```                            |

## `e`


| Functionality Name   | Sample Snippet                                              |
|----------------------|-------------------------------------------------------------|
| e                    | ```torch.e```                                               |
| eig                  | ```torch.eig(input, eigenvectors=False, out=None)```       |
| einsum               | ```torch.einsum(equation, *operands)```                    |
| embedding            | ```torch.nn.Embedding(num_embeddings, embedding_dim)```      |
| embedding_bag        | ```torch.nn.EmbeddingBag(num_embeddings, embedding_dim)```  |
| embedding_renorm_    | ```torch.embedding_renorm_(weight, indices, max_norm, norm_type=2.0)``` |
| empty                | ```torch.empty(size, dtype=torch.float32, device=None)```   |
| empty_like           | ```torch.empty_like(tensor, dtype=None, layout=None, device=None, requires_grad=False)``` |
| empty_permuted       | ```torch.empty_permuted(size, dims, dtype=torch.float32, device=None)``` |
| empty_quantized      | ```torch.empty_quantized(size, dtype=torch.quint8, layout=torch.strided, device=None, requires_grad=False)``` |
| empty_strided        | ```torch.empty_strided(size, stride, dtype=torch.float32, device=None)``` |
| enable_grad          | ```torch.enable_grad()```                                   |
| eq                   | ```torch.eq(tensor1, tensor2)```                            |
| equal                | ```torch.equal(tensor1, tensor2)```                         |
| erf                  | ```torch.erf(tensor)```                                     |
| erf_                 | ```tensor.erf_()```                                         |
| erfc                 | ```torch.erfc(tensor)```                                    |
| erfc_                | ```tensor.erfc_()```                                        |
| erfinv               | ```torch.erfinv(tensor)```                                  |
| exp                  | ```torch.exp(tensor)```                                     |
| exp2                 | ```torch.exp2(tensor)```                                    |
| exp2_                | ```tensor.exp2_()```                                        |
| exp_                 | ```tensor.exp_()```                                         |
| expand_copy          | ```torch.expand_copy(tensor, sizes)```                      |
| expm1                | ```torch.expm1(tensor)```                                   |
| expm1_               | ```tensor.expm1_()```                                       |
| export               | ```torch.export(module, file, optimize=True, pickle_module=dill)``` |
| eye                  | ```torch.eye(n, m=None, dtype=torch.float32, device=None)```|

## `f`
| Functionality Name                            | Sample Snippet                                                     |
|-----------------------------------------------|--------------------------------------------------------------------|
| fake_quantize_per_channel_affine             | ```torch.fake_quantize_per_channel_affine(input, scale, zero_point, axis, quant_min, quant_max)``` |
| fake_quantize_per_tensor_affine              | ```torch.fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)``` |
| fbgemm_linear_fp16_weight                    | ```torch.fbgemm_linear_fp16_weight(input, weight, bias=None)```   |
| fbgemm_linear_fp16_weight_fp32_activation    | ```torch.fbgemm_linear_fp16_weight_fp32_activation(input, weight, bias=None)``` |
| fbgemm_linear_int8_weight                    | ```torch.fbgemm_linear_int8_weight(input, weight, bias=None)```   |
| fbgemm_linear_int8_weight_fp32_activation    | ```torch.fbgemm_linear_int8_weight_fp32_activation(input, weight, bias=None)``` |
| fbgemm_linear_quantize_weight                | ```torch.fbgemm_linear_quantize_weight(weight, weight_scale, weight_zero_point, weight_dtype=torch.quint8)``` |
| fbgemm_pack_gemm_matrix_fp16                 | ```torch.fbgemm_pack_gemm_matrix_fp16(input)```                   |
| fbgemm_pack_quantized_matrix                 | ```torch.fbgemm_pack_quantized_matrix(input)```                   |
| feature_alpha_dropout                        | ```torch.nn.functional.feature_alpha_dropout(input, p, train=True, inplace=False)``` |
| feature_alpha_dropout_                       | ```torch.nn.functional.feature_alpha_dropout_(input, p, train=True)``` |
| feature_dropout                              | ```torch.nn.functional.feature_dropout(input, p, train=True, inplace=False)``` |
| feature_dropout_                             | ```torch.nn.functional.feature_dropout_(input, p, train=True)``` |
| fft                                          | ```torch.fft(input, signal_ndim, normalized=False)```            |
| fill                                         | ```torch.fill(input, value)```                                    |
| fill_                                        | ```tensor.fill_(value)```                                        |
| finfo                                        | ```torch.finfo(dtype)```                                         |
| fix                                          | ```torch.fix(tensor, out=None)```                                |
| fix_                                         | ```tensor.fix_()```                                              |
| flatten                                      | ```torch.flatten(input, start_dim=0, end_dim=-1)```              |
| flip                                         | ```torch.flip(input, dims)```                                    |
| fliplr                                       | ```torch.fliplr(input)```                                        |
| flipud                                       | ```torch.flipud(input)```                                        |
| float                                        | ```torch.float(tensor)```                                        |
| float16                                      | ```torch.float16(tensor)```                                      |
| float32                                      | ```torch.float32(tensor)```                                      |
| float64                                      | ```torch.float64(tensor)```                                      |
| float8_e4m3fn                                | ```torch.float8_e4m3fn(tensor)```                                |
| float8_e4m3fnuz                              | ```torch.float8_e4m3fnuz(tensor)```                              |
| float8_e5m2                                  | ```torch.float8_e5m2(tensor)```                                  |
| float8_e5m2fnuz                              | ```torch.float8_e5m2fnuz(tensor)```                              |
| float_power                                  | ```torch.float_power(input, exponent)```                         |
| floor                                        | ```torch.floor(tensor)```                                        |
| floor_                                       | ```tensor.floor_()```                                            |
| floor_divide                                 | ```torch.floor_divide(tensor1, tensor2)```                       |
| fmax                                         | ```torch.fmax(input1, input2)```                                 |
| fmin                                         | ```torch.fmin(input1, input2)```                                 |
| fmod                                         | ```torch.fmod(input1, input2)```                                 |
| fork                                         | ```torch.fork()```                                               |
| frac                                         | ```torch.frac(tensor)```                                         |
| frac_                                        | ```tensor.frac_()```                                             |
| frexp                                        | ```torch.frexp(tensor)```                                        |
| frobenius_norm                               | ```torch.frobenius_norm(input)```                                |
| from_dlpack                                  | ```torch.from_dlpack(dltensor)```                                |
| from_file                                    | ```torch.from_file(path)```                                      |
| from_numpy                                   | ```torch.from_numpy(ndarray)```                                  |
| frombuffer                                   | ```torch.frombuffer(buffer)```                                   |
| full                                         | ```torch.full(size, fill_value, dtype=None, device=None)```      |
| full_like                                    | ```torch.full_like(input, fill_value, dtype=None, layout=None, device=None, requires_grad=False)``` |
| func                                         | ```torch.func(tensor)```                                         |
| functional                                   | ```torch.functional```                                           |
| fused_moving_avg_obs_fake_quant             | ```torch.fused_moving_avg_obs_fake_quant(input, scale, zero_point, quant_min, quant_max, axis=-1)``` |
| futures                                      | ```torch.futures```                                              |
| fx                                           | ```torch.fx(input)```                                            |


## `g`


| Functionality Name                | Sample Snippet                                         |
|-----------------------------------|--------------------------------------------------------|
| gather                            | ```torch.gather(input, dim, index, out=None)```       |
| gcd                               | ```torch.gcd(tensor1, tensor2)```                      |
| gcd_                              | ```tensor.gcd_(value)```                               |
| ge                                | ```torch.ge(tensor1, tensor2)```                       |
| geqrf                             | ```torch.geqrf(input, out=None)```                     |
| ger                               | ```torch.ger(vec1, vec2)```                            |
| get_autocast_cpu_dtype            | ```torch.get_autocast_cpu_dtype()```                   |
| get_autocast_gpu_dtype            | ```torch.get_autocast_gpu_dtype()```                   |
| get_autocast_ipu_dtype            | ```torch.get_autocast_ipu_dtype()```                   |
| get_autocast_xla_dtype            | ```torch.get_autocast_xla_dtype()```                   |
| get_default_dtype                 | ```torch.get_default_dtype()```                        |
| get_deterministic_debug_mode      | ```torch.get_deterministic_debug_mode()```             |
| get_device                        | ```torch.get_device(tensor)```                         |
| get_file_path                     | ```torch.get_file_path(filename)```                    |
| get_float32_matmul_precision      | ```torch.get_float32_matmul_precision()```             |
| get_num_interop_threads           | ```torch.get_num_interop_threads()```                  |
| get_num_threads                   | ```torch.get_num_threads()```                          |
| get_rng_state                     | ```torch.get_rng_state()```                            |
| glob                              | ```torch.glob(pattern)```                             |
| gradient                          | ```torch.gradient(tensor, edge_order=1)```            |
| greater                           | ```torch.greater(tensor1, tensor2)```                 |
| greater_equal                     | ```torch.greater_equal(tensor1, tensor2)```           |
| grid_sampler                      | ```torch.nn.functional.grid_sampler(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)``` |
| grid_sampler_2d                   | ```torch.nn.functional.grid_sampler_2d(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)``` |
| grid_sampler_3d                   | ```torch.nn.functional.grid_sampler_3d(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)``` |
| group_norm                        | ```torch.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-05, cudnn_enabled=True)``` |
| gru                               | ```torch.nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0.0, bidirectional=False)``` |
| gru_cell                          | ```torch.nn.GRUCell(input_size, hidden_size, bias=True)``` |
| gt                                | ```torch.gt(tensor1, tensor2)```                       |

## `h`

| Functionality Name        | Sample Snippet                                              |
|---------------------------|-------------------------------------------------------------|
| half                      | ```torch.half(tensor)```                                    |
| hamming_window            | ```torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, device=None, requires_grad=False)``` |
| hann_window               | ```torch.hann_window(window_length, periodic=True, dtype=None, device=None, requires_grad=False)``` |
| hardshrink                | ```torch.nn.functional.hardshrink(input, lambd=0.5)```       |
| has_lapack                | ```torch.has_lapack```                                       |
| has_mkl                   | ```torch.has_mkl```                                          |
| has_openmp                | ```torch.has_openmp```                                       |
| has_spectral              | ```torch.has_spectral```                                     |
| heaviside                 | ```torch.heaviside(input, values)```                         |
| hinge_embedding_loss      | ```torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0, reduction='mean')``` |
| histc                     | ```torch.histc(input, bins=100, min=0, max=0, out=None)```  |
| histogram                 | ```torch.histogram(tensor, bins=100, min=0, max=0, weights=None, density=False)``` |
| histogramdd               | ```torch.histogramdd(input, bins, range=None, weights=None, density=False)``` |
| hsmm                      | ```torch.hsmm(transitions, emission_log_probs, observations, lengths, hidden, durations=None, pack=False, total_states=None)``` |
| hsplit                    | ```torch.hsplit(tensor, indices_or_sections)```               |
| hspmm                     | ```torch.hspmm(mat1, mat2)```                                |
| hstack                    | ```torch.hstack(tensors)```                                  |
| hub                       | ```torch.hub.load(repo_or_dir, model, force_reload=False, verbose=False, *args, **kwargs)``` |
| hypot                     | ```torch.hypot(input1, input2, out=None)```                  |

## `i`
| Functionality Name                        | Sample Snippet                                                    |
|-------------------------------------------|-------------------------------------------------------------------|
| i0                                        | ```torch.i0(tensor)```                                            |
| i0_                                       | ```tensor.i0_()```                                                |
| igamma                                    | ```torch.igamma(tensor1, tensor2)```                              |
| igammac                                   | ```torch.igammac(tensor1, tensor2)```                             |
| iinfo                                     | ```torch.iinfo(dtype)```                                          |
| imag                                      | ```torch.imag(tensor)```                                          |
| import_ir_module                          | ```torch.import_ir_module(filename)```                            |
| import_ir_module_from_buffer              | ```torch.import_ir_module_from_buffer(buffer)```                  |
| index_add                                 | ```torch.index_add(input, dim, index, source)```                  |
| index_copy                                | ```torch.index_copy(input, dim, index, source)```                 |
| index_fill                                | ```torch.index_fill(input, dim, index, value)```                  |
| index_put                                 | ```torch.index_put(input, indices, values, accumulate=False)```   |
| index_put_                                | ```torch.index_put_(input, indices, values, accumulate=False)```  |
| index_reduce                              | ```torch.index_reduce(input, dim, index, reducer='add')```        |
| index_select                              | ```torch.index_select(input, dim, index)```                       |
| indices_copy                              | ```torch.indices_copy(input, indices)```                          |
| inf                                       | ```torch.inf```                                                   |
| inference_mode                            | ```torch.inference_mode()```                                      |
| init_num_threads                          | ```torch.init_num_threads()```                                    |
| initial_seed                              | ```torch.initial_seed()```                                        |
| inner                                     | ```torch.inner(input1, input2)```                                 |
| inspect                                   | ```torch.inspect```                                               |
| instance_norm                             | ```torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)``` |
| int                                       | ```torch.int(tensor)```                                           |
| int16                                     | ```torch.int16(tensor)```                                         |
| int32                                     | ```torch.int32(tensor)```                                         |
| int64                                     | ```torch.int64(tensor)```                                         |
| int8                                      | ```torch.int8(tensor)```                                          |
| int_repr                                  | ```torch.int_repr(tensor)```                                      |
| inverse                                   | ```torch.inverse(input)```                                        |
| is_anomaly_check_nan_enabled              | ```torch.is_anomaly_check_nan_enabled()```                        |
| is_anomaly_enabled                        | ```torch.is_anomaly_enabled()```                                  |
| is_autocast_cache_enabled                 | ```torch.is_autocast_cache_enabled()```                           |
| is_autocast_cpu_enabled                   | ```torch.is_autocast_cpu_enabled()```                             |
| is_autocast_enabled                       | ```torch.is_autocast_enabled()```                                 |
| is_autocast_ipu_enabled                   | ```torch.is_autocast_ipu_enabled()```                             |
| is_autocast_xla_enabled                   | ```torch.is_autocast_xla_enabled()```                             |
| is_complex                                | ```torch.is_complex(tensor)```                                    |
| is_conj                                   | ```torch.is_conj(tensor)```                                       |
| is_deterministic_algorithms_warn_only_enabled | ```torch.is_deterministic_algorithms_warn_only_enabled()```       |
| is_distributed                            | ```torch.is_distributed()```                                      |
| is_floating_point                        | ```torch.is_floating_point(tensor)```                             |
| is_grad_enabled                          | ```torch.is_grad_enabled()```                                     |
| is_inference                             | ```torch.is_inference(tensor)```                                 |
| is_inference_mode_enabled                | ```torch.is_inference_mode_enabled()```                           |
| is_loaded                                | ```torch.is_loaded()```                                           |
| is_neg                                   | ```torch.is_neg(tensor)```                                        |
| is_nonzero                               | ```torch.is_nonzero(tensor)```                                    |
| is_same_size                             | ```torch.is_same_size(tensor1, tensor2)```                        |
| is_signed                                | ```torch.is_signed(tensor)```                                     |
| is_storage                               | ```torch.is_storage(tensor)```                                    |
| is_tensor                                | ```torch.is_tensor(obj)```                                        |
| is_vulkan_available                      | ```torch.is_vulkan_available()```                                 |
| is_warn_always_enabled                   | ```torch.is_warn_always_enabled()```                              |
| isclose                                  | ```torch.isclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, equal_nan=False)``` |
| isfinite                                 | ```torch.isfinite(tensor)```                                      |
| isin                                     | ```torch.isin(input, other)```                                    |
| isinf                                    | ```torch.isinf(tensor)```                                         |
| isnan                                    | ```torch.isnan(tensor)```                                         |
| isneginf                                 | ```torch.isneginf(tensor)```                                      |
| isposinf                                 | ```torch.isposinf(tensor)```                                      |
| isreal                                   | ```torch.isreal(tensor)```                                        |
| istft                                    | ```torch.istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=True, length=None, return_complex=False)``` |

## `j`

| Functionality Name   | Sample Snippet                                            |
|-----------------------|-----------------------------------------------------------|
| jagged                | *Explanation: There is no specific function known as 'jagged' in the torch library.* |
| jit                   | ```import torch```<br>```@torch.jit.script```<br>```def add(x, y):```<br>&nbsp;&nbsp;```return x + y```<br>```# Usage```<br>```a = torch.tensor(3)```<br>```b = torch.tensor(4)```<br>```print(add(a, b))``` |

## `k`

| Functionality Name   | Sample Snippet                                            |
|-----------------------|-----------------------------------------------------------|
| kaiser_window         | ```torch.kaiser_window(window_length, beta, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)``` |
| kernel32              | *Explanation: 'kernel32' appears to be related to the Windows kernel and is not a Torch function.* |
| kl_div                | ```torch.nn.functional.kl_div(input, target, reduction='mean')``` |
| kron                  | ```torch.kron(tensor1, tensor2)```                         |
| kthvalue              | ```torch.kthvalue(input, k, dim=None, keepdim=False, out=None)``` |

## `l`



| Functionality Name       | Sample Snippet                                                   |
|--------------------------|------------------------------------------------------------------|
| last_error               | *Explanation: 'last_error' seems to be related to error handling and is not a Torch function.* |
| layer_norm               | ```torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)``` |
| layout                   | ```torch.layout```                                               |
| lcm                      | ```torch.lcm(tensor1, tensor2)```                                |
| lcm_                     | ```tensor.lcm_(value)```                                         |
| ldexp                    | ```torch.ldexp(mantissa, exponent)```                             |
| ldexp_                   | ```tensor.ldexp_(exponent)```                                    |
| le                       | ```torch.le(tensor1, tensor2)```                                 |
| legacy_contiguous_format | ```torch.legacy_contiguous_format```                             |
| lerp                     | ```torch.lerp(start, end, weight)```                             |
| less                     | ```torch.less(tensor1, tensor2)```                               |
| less_equal               | ```torch.less_equal(tensor1, tensor2)```                         |
| lgamma                   | ```torch.lgamma(tensor)```                                       |
| library                  | *Explanation: 'library' is not a Torch function.*                |
| linalg                   | *Explanation: 'linalg' is likely a subpackage for linear algebra functions and may have multiple functions within.* |
| linspace                 | ```torch.linspace(start, end, steps, out=None)```                |
| load                     | ```torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/usr/lib/python3.7/pickle.py'>)``` |
| lobpcg                   | *Explanation: 'lobpcg' is not a Torch function.*                 |
| log                      | ```torch.log(input)```                                           |
| log10                    | ```torch.log10(input)```                                         |
| log10_                   | ```tensor.log10_()```                                            |
| log1p                    | ```torch.log1p(input)```                                         |
| log1p_                   | ```tensor.log1p_()```                                            |
| log2                     | ```torch.log2(input)```                                          |
| log2_                    | ```tensor.log2_()```                                             |
| log_                     | ```torch.log(input)```                                           |
| log_softmax              | ```torch.nn.functional.log_softmax(input, dim=-1, _stacklevel=3, dtype=None)``` |
| logaddexp                | ```torch.logaddexp(tensor1, tensor2)```                          |
| logaddexp2               | ```torch.logaddexp2(tensor1, tensor2)```                         |
| logcumsumexp             | ```torch.logcumsumexp(input, dim, dtype=None)```                 |
| logdet                   | ```torch.logdet(input)```                                        |
| logical_and              | ```torch.logical_and(tensor1, tensor2)```                        |
| logical_not              | ```torch.logical_not(tensor)```                                  |
| logical_or               | ```torch.logical_or(tensor1, tensor2)```                         |
| logical_xor              | ```torch.logical_xor(tensor1, tensor2)```                        |
| logit                    | ```torch.logit(input, eps=None)```                               |
| logit_                   | ```tensor.logit_(eps=None)```                                    |
| logspace                 | ```torch.logspace(start, end, steps, base=10.0, out=None)```    |
| logsumexp                | ```torch.logsumexp(input, dim, keepdim=False, out=None)```      |
| long                     | ```torch.long(tensor)```                                         |
| lstm                     | ```torch.nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0.0, bidirectional=False)``` |
| lstm_cell                | ```torch.nn.LSTMCell(input_size, hidden_size, bias=True)```      |
| lstsq                    | ```torch.lstsq(input, A, *args, **kwargs)```                    |
| lt                       | ```torch.lt(tensor1, tensor2)```                                 |
| lu                       | ```torch.lu(input, pivot=True, get_infos=False)```               |
| lu_solve                 | ```torch.lu_solve(input, LU_data, LU_pivots, ... , out=None)``` |
| lu_unpack                | ```torch.lu_unpack(LU_data, LU_pivots)```                        |

## `m`

| Functionality             | Sample Snippet                                      |
|---------------------------|-----------------------------------------------------|
| `manual_seed`             | `torch.manual_seed(seed)`                           |
| `margin_ranking_loss`     | `torch.margin_ranking_loss(input1, input2, target)` |
| `masked`                  | `torch.masked_select(input, mask)`                  |
| `masked_fill`             | `torch.masked_fill(input, mask, value)`             |
| `masked_scatter`          | `torch.masked_scatter(input, mask, source)`         |
| `masked_select`           | `torch.masked_select(input, mask)`                  |
| `math`                    | `torch.math`                                        |
| `matmul`                  | `torch.matmul(input, other)`                        |
| `matrix_exp`              | `torch.matrix_exp(input)`                           |
| `matrix_power`            | `torch.matrix_power(input, n)`                      |
| `matrix_rank`             | `torch.matrix_rank(input)`                          |
| `max`                     | `torch.max(input)`                                  |
| `max_pool1d`              | `torch.max_pool1d(input, kernel_size)`             |
| `max_pool1d_with_indices` | `torch.max_pool1d_with_indices(input, kernel_size)`|
| `max_pool2d`              | `torch.max_pool2d(input, kernel_size)`             |
| `max_pool3d`              | `torch.max_pool3d(input, kernel_size)`             |
| `maximum`                 | `torch.maximum(input1, input2)`                    |
| `mean`                    | `torch.mean(input)`                                |
| `median`                  | `torch.median(input)`                              |
| `memory_format`           | `torch.memory_format()`                            |
| `meshgrid`                | `torch.meshgrid(*tensors)`                         |
| `min`                     | `torch.min(input)`                                  |
| `minimum`                 | `torch.minimum(input1, input2)`                    |
| `miopen_batch_norm`       | `torch.miopen_batch_norm(input, weight, bias)`      |
| `miopen_convolution`      | `torch.miopen_convolution(input, weight, bias)`     |
| `miopen_convolution_add_relu` | `torch.miopen_convolution_add_relu(input, weight, bias)`|
| `miopen_convolution_relu` | `torch.miopen_convolution_relu(input, weight, bias)`|
| `miopen_convolution_transpose` | `torch.miopen_convolution_transpose(input, weight, bias)`|
| `miopen_depthwise_convolution` | `torch.miopen_depthwise_convolution(input, weight, bias)`|
| `miopen_rnn`              | `torch.miopen_rnn(input, weight, bias)`            |
| `mkldnn_adaptive_avg_pool2d` | `torch.mkldnn_adaptive_avg_pool2d(input, output_size)`|
| `mkldnn_convolution`      | `torch.mkldnn_convolution(input, weight, bias)`     |
| `mkldnn_linear_backward_weights` | `torch.mkldnn_linear_backward_weights(input, weight, bias)`|
| `mkldnn_max_pool2d`       | `torch.mkldnn_max_pool2d(input, kernel_size)`      |
| `mkldnn_max_pool3d`       | `torch.mkldnn_max_pool3d(input, kernel_size)`      |
| `mkldnn_rnn_layer`        | `torch.mkldnn_rnn_layer(input, weight, bias)`       |
| `mm`                      | `torch.mm(input, mat2)`                            |
| `mode`                    | `torch.mode(input)`                                |
| `moveaxis`                | `torch.moveaxis(input, source, destination)`        |
| `movedim`                 | `torch.movedim(input, dims, destination)`           |
| `mps`                     | `torch.mps(input, weight, bias)`                   |
| `msort`                   | `torch.msort(input)`                               |
| `mul`                     | `torch.mul(input, other)`                          |
| `multinomial`             | `torch.multinomial(input, num_samples)`             |
| `multiply`                | `torch.multiply(input, other)`                     |
| `multiprocessing`         | `torch.multiprocessing`                            |
| `mv`                      | `torch.mv(input, vec)`                             |
| `mvlgamma`                | `torch.mvlgamma(input, p)`                         |


## `n`

| Functionality          | Sample Snippet                           |
|------------------------|------------------------------------------|
| `name`                 | `torch.name(tensor)`                     |
| `nan`                  | `torch.nan(tensor)`                      |
| `nan_to_num`           | `torch.nan_to_num(tensor)`               |
| `nan_to_num_`          | `torch.nan_to_num_(tensor)`              |
| `nanmean`              | `torch.nanmean(tensor)`                  |
| `nanmedian`            | `torch.nanmedian(tensor)`                |
| `nanquantile`          | `torch.nanquantile(tensor, q)`          |
| `nansum`               | `torch.nansum(tensor)`                   |
| `narrow`               | `torch.narrow(tensor, dim, start, length)` |
| `narrow_copy`          | `torch.narrow_copy(tensor, dim, start, length)` |
| `native_batch_norm`    | `torch.native_batch_norm(input, weight, bias, running_mean, running_var)` |
| `native_channel_shuffle`| `torch.native_channel_shuffle(input, groups)` |
| `native_dropout`       | `torch.native_dropout(input, p, training)` |
| `native_group_norm`    | `torch.native_group_norm(input, weight, bias, ...) `|
| `native_layer_norm`    | `torch.native_layer_norm(input, ...) `  |
| `native_norm`          | `torch.native_norm(input, p)`           |
| `ne`                   | `torch.ne(input, other)`                |
| `neg`                  | `torch.neg(input)`                      |
| `neg_`                 | `torch.neg_(input)`                     |
| `negative`             | `torch.negative(input)`                 |
| `negative_`            | `torch.negative_(input)`                |
| `nested`               | `torch.nested(input)`                   |
| `nextafter`            | `torch.nextafter(tensor1, tensor2)`     |
| `nn`                   | `torch.nn`                              |
| `no_grad`              | `torch.no_grad()`                       |
| `nonzero`              | `torch.nonzero(tensor)`                 |
| `nonzero_static`       | `torch.nonzero_static(tensor)`          |
| `norm`                 | `torch.norm(input, p)`                  |
| `norm_except_dim`      | `torch.norm_except_dim(input, dim)`     |
| `normal`               | `torch.normal(mean, std)`               |
| `not_equal`            | `torch.not_equal(input, other)`         |
| `nuclear_norm`         | `torch.nuclear_norm(input)`             |
| `numel`                | `torch.numel(input)`                    |
| `nvtoolsext_dll_path`  | `torch.nvtoolsext_dll_path()`           |

## `o`
| Functionality   | Sample Snippet                  |
|-----------------|---------------------------------|
| `obj`           | `torch.obj()`                   |
| `ones`          | `torch.ones(size)`              |
| `ones_like`     | `torch.ones_like(input)`        |
| `ops`           | `torch.ops`                     |
| `optim`         | `torch.optim`                   |
| `orgqr`         | `torch.orgqr(input, tau)`       |
| `ormqr`         | `torch.ormqr(input, input2, ...) ` |
| `os`            | `torch.os`                      |
| `outer`         | `torch.outer(input1, input2)`   |
| `overrides`     | `torch.overrides`               |

## `p`

| Functionality                       | Sample Snippet                                   |
|-------------------------------------|--------------------------------------------------|
| `package`                           | `torch.package`                                  |
| `pairwise_distance`                 | `torch.pairwise_distance(x1, x2, p)`             |
| `parse_ir`                          | `torch.parse_ir(input)`                          |
| `parse_schema`                      | `torch.parse_schema(schema)`                     |
| `parse_type_comment`                | `torch.parse_type_comment(comment)`              |
| `path_patched`                      | `torch.path_patched()`                           |
| `pca_lowrank`                       | `torch.pca_lowrank(input, q)`                    |
| `pdist`                             | `torch.pdist(input, p)`                          |
| `per_channel_affine`                | `torch.per_channel_affine(input, scale, bias)`   |
| `per_channel_affine_float_qparams`  | `torch.per_channel_affine_float_qparams`         |
| `per_channel_symmetric`             | `torch.per_channel_symmetric(input, scale)`      |
| `per_tensor_affine`                 | `torch.per_tensor_affine(input, scale, bias)`    |
| `per_tensor_symmetric`              | `torch.per_tensor_symmetric(input, scale)`       |
| `permute`                           | `torch.permute(input, dims)`                     |
| `permute_copy`                      | `torch.permute_copy(input, dims)`                |
| `pfiles_path`                       | `torch.pfiles_path()`                            |
| `pi`                                | `torch.pi`                                       |
| `pinverse`                          | `torch.pinverse(input)`                          |
| `pixel_shuffle`                     | `torch.pixel_shuffle(input, upscale_factor)`     |
| `pixel_unshuffle`                   | `torch.pixel_unshuffle(input, downscale_factor)` |
| `platform`                          | `torch.platform`                                 |
| `poisson`                           | `torch.poisson(input)`                           |
| `poisson_nll_loss`                  | `torch.poisson_nll_loss(input, target, ...) `    |
| `polar`                             | `torch.polar(input)`                             |
| `polygamma`                         | `torch.polygamma(input, n)`                      |
| `positive`                          | `torch.positive(input)`                          |
| `pow`                               | `torch.pow(input, exponent)`                     |
| `prelu`                             | `torch.prelu(input, weight)`                     |
| `prepare_multiprocessing_environment`| `torch.prepare_multiprocessing_environment()`  |
| `preserve_format`                   | `torch.preserve_format(input)`                   |
| `prev_error_mode`                   | `torch.prev_error_mode()`                        |
| `prod`                              | `torch.prod(input)`                              |
| `profiler`                          | `torch.profiler`                                 |
| `promote_types`                     | `torch.promote_types(type1, type2)`              |
| `put`                               | `torch.put(input, indices, values)`              |
| `py_dll_path`                       | `torch.py_dll_path()`                            |
| `py_float`                          | `torch.py_float`                                 |
| `py_int`                            | `torch.py_int`                                   |

## `q`

| Functionality                       | Sample Snippet                                     |
|-------------------------------------|----------------------------------------------------|
| `q_per_channel_axis`                | `torch.q_per_channel_axis(input)`                  |
| `q_per_channel_scales`              | `torch.q_per_channel_scales(input)`                |
| `q_per_channel_zero_points`         | `torch.q_per_channel_zero_points(input)`           |
| `q_scale`                           | `torch.q_scale(input)`                             |
| `q_zero_point`                      | `torch.q_zero_point(input)`                        |
| `qint32`                            | `torch.qint32(input)`                              |
| `qint8`                             | `torch.qint8(input)`                               |
| `qr`                                | `torch.qr(input)`                                  |
| `qscheme`                           | `torch.qscheme`                                    |
| `quantile`                          | `torch.quantile(input, q)`                         |
| `quantization`                      | `torch.quantization`                               |
| `quantize_per_channel`              | `torch.quantize_per_channel(input, scales, zero_points, dtype)` |
| `quantize_per_tensor`               | `torch.quantize_per_tensor(input, scale, zero_point, dtype)`    |
| `quantize_per_tensor_dynamic`       | `torch.quantize_per_tensor_dynamic(input, scale, zero_point, dtype)` |
| `quantized_batch_norm`              | `torch.quantized_batch_norm(input, ...) `          |
| `quantized_gru`                     | `torch.quantized_gru(input, ...) `                 |
| `quantized_gru_cell`                | `torch.quantized_gru_cell(input, ...) `            |
| `quantized_lstm`                    | `torch.quantized_lstm(input, ...) `                |
| `quantized_lstm_cell`               | `torch.quantized_lstm_cell(input, ...) `           |
| `quantized_max_pool1d`              | `torch.quantized_max_pool1d(input, ...) `          |
| `quantized_max_pool2d`              | `torch.quantized_max_pool2d(input, ...) `          |
| `quantized_max_pool3d`              | `torch.quantized_max_pool3d(input, ...) `          |
| `quantized_rnn_relu_cell`           | `torch.quantized_rnn_relu_cell(input, ...) `       |
| `quantized_rnn_tanh_cell`           | `torch.quantized_rnn_tanh_cell(input, ...) `       |
| `quasirandom`                       | `torch.quasirandom(input)`                         |
| `quint2x4`                          | `torch.quint2x4(input)`                            |
| `quint4x2`                          | `torch.quint4x2(input)`                            |
| `quint8`                            | `torch.quint8(input)`                              |

## `r`

| Functionality           | Sample Snippet                         |
|-------------------------|----------------------------------------|
| `rad2deg`               | `torch.rad2deg(input)`                 |
| `rad2deg_`              | `torch.rad2deg_(input)`                |
| `rand`                  | `torch.rand(size)`                     |
| `rand_like`             | `torch.rand_like(input)`               |
| `randint`               | `torch.randint(low, high, size)`       |
| `randint_like`          | `torch.randint_like(input, low, high)` |
| `randn`                 | `torch.randn(size)`                    |
| `randn_like`            | `torch.randn_like(input)`              |
| `random`                | `torch.random(size)`                   |
| `randperm`              | `torch.randperm(n)`                    |
| `range`                 | `torch.range(start, end, step)`        |
| `ravel`                 | `torch.ravel(input)`                   |
| `read_vitals`           | `torch.read_vitals()`                  |
| `real`                  | `torch.real(input)`                    |
| `reciprocal`            | `torch.reciprocal(input)`              |
| `reciprocal_`           | `torch.reciprocal_(input)`             |
| `relu`                  | `torch.relu(input)`                    |
| `relu_`                 | `torch.relu_(input)`                   |
| `remainder`             | `torch.remainder(input1, input2)`      |
| `renorm`                | `torch.renorm(input, p, dim, maxnorm)` |
| `repeat_interleave`     | `torch.repeat_interleave(input, repeats)` |
| `res`                   | `torch.res(input)`                     |
| `reshape`               | `torch.reshape(input, shape)`          |
| `resize_as_`            | `torch.resize_as_(input, other)`       |
| `resize_as_sparse_`     | `torch.resize_as_sparse_(input, ...) ` |
| `resolve_conj`          | `torch.resolve_conj(input)`            |
| `resolve_neg`           | `torch.resolve_neg(input)`             |
| `result_type`           | `torch.result_type(input1, input2)`    |
| `return_types`          | `torch.return_types(input)`            |
| `rnn_relu`              | `torch.rnn_relu(input, ...) `          |
| `rnn_relu_cell`         | `torch.rnn_relu_cell(input, ...) `     |
| `rnn_tanh`              | `torch.rnn_tanh(input, ...) `          |
| `rnn_tanh_cell`         | `torch.rnn_tanh_cell(input, ...) `     |
| `roll`                  | `torch.roll(input, shifts, dims)`      |
| `rot90`                 | `torch.rot90(input, k, dims)`          |
| `round`                 | `torch.round(input)`                   |
| `round_`                | `torch.round_(input)`                  |
| `row_indices_copy`      | `torch.row_indices_copy(input)`        |
| `row_stack`             | `torch.row_stack(tensors)`             |
| `rrelu`                 | `torch.rrelu(input)`                   |
| `rrelu_`                | `torch.rrelu_(input)`                  |
| `rsqrt`                 | `torch.rsqrt(input)`                   |
| `rsqrt_`                | `torch.rsqrt_(input)`                  |
| `rsub`                  | `torch.rsub(input1, input2)`           |


## `s`

| Functionality                     | Sample Snippet                          |
|-----------------------------------|-----------------------------------------|
| `saddmm`                          | `torch.saddmm(input, mat1, mat2, beta, alpha)` |
| `save`                            | `torch.save(obj, f)`                    |
| `scalar_tensor`                   | `torch.scalar_tensor(scalar)`           |
| `scatter`                         | `torch.scatter(input, dim, index, src)` |
| `scatter_add`                     | `torch.scatter_add(input, dim, index, src)` |
| `scatter_reduce`                  | `torch.scatter_reduce(input, dim, index, src)` |
| `searchsorted`                    | `torch.searchsorted(sorted_sequence, values)` |
| `seed`                            | `torch.seed()`                          |
| `segment_reduce`                  | `torch.segment_reduce(input, dim, index, ...) ` |
| `select`                          | `torch.select(input, dim, index)`       |
| `select_copy`                     | `torch.select_copy(input, dim, index)`  |
| `select_scatter`                  | `torch.select_scatter(input, dim, index, src)` |
| `selu`                            | `torch.selu(input)`                    |
| `selu_`                           | `torch.selu_(input)`                   |
| `serialization`                   | `torch.serialization`                   |
| `set_anomaly_enabled`             | `torch.set_anomaly_enabled(True)`       |
| `set_autocast_cache_enabled`      | `torch.set_autocast_cache_enabled(True)` |
| `set_autocast_cpu_dtype`          | `torch.set_autocast_cpu_dtype(dtype)`   |
| `set_autocast_cpu_enabled`        | `torch.set_autocast_cpu_enabled(True)`  |
| `set_autocast_enabled`            | `torch.set_autocast_enabled(True)`      |
| `set_autocast_gpu_dtype`          | `torch.set_autocast_gpu_dtype(dtype)`   |
| `set_autocast_ipu_dtype`          | `torch.set_autocast_ipu_dtype(dtype)`   |
| `set_autocast_ipu_enabled`        | `torch.set_autocast_ipu_enabled(True)`  |
| `set_autocast_xla_dtype`          | `torch.set_autocast_xla_dtype(dtype)`   |
| `set_autocast_xla_enabled`        | `torch.set_autocast_xla_enabled(True)`  |
| `set_default_device`              | `torch.set_default_device(device)`      |
| `set_default_dtype`               | `torch.set_default_dtype(dtype)`        |
| `set_default_tensor_type`         | `torch.set_default_tensor_type(type)`   |
| `set_deterministic_debug_mode`    | `torch.set_deterministic_debug_mode(True)` |
| `set_float32_matmul_precision`    | `torch.set_float32_matmul_precision(True)` |
| `set_flush_denormal`              | `torch.set_flush_denormal(True)`        |
| `set_grad_enabled`                | `torch.set_grad_enabled(True)`          |
| `set_num_interop_threads`         | `torch.set_num_interop_threads(num_threads)` |
| `set_num_threads`                 | `torch.set_num_threads(num_threads)`    |
| `set_printoptions`                | `torch.set_printoptions(...)`           |
| `set_rng_state`                   | `torch.set_rng_state(state)`            |
| `set_vital`                       | `torch.set_vital(input, ...) `          |
| `set_warn_always`                 | `torch.set_warn_always(True)`           |
| `sgn`                             | `torch.sgn(input)`                      |
| `short`                           | `torch.short(input)`                    |
| `sigmoid`                         | `torch.sigmoid(input)`                  |
| `sigmoid_`                        | `torch.sigmoid_(input)`                 |
| `sign`                            | `torch.sign(input)`                     |
| `signal`                          | `torch.signal(input)`                   |
| `signbit`                         | `torch.signbit(input)`                  |
| `sin`                             | `torch.sin(input)`                      |
| `sin_`                            | `torch.sin_(input)`                     |
| `sinc`                            | `torch.sinc(input)`                     |
| `sinc_`                           | `torch.sinc_(input)`                    |
| `sinh`                            | `torch.sinh(input)`                     |
| `sinh_`                           | `torch.sinh_(input)`                    |
| `slice_copy`                      | `torch.slice_copy(input)`               |
| `slice_scatter`                   | `torch.slice_scatter(input, dim, index, src)` |
| `slogdet`                         | `torch.slogdet(input)`                  |
| `smm`                             | `torch.smm(input, mat2)`                |
| `softmax`                         | `torch.softmax(input, dim)`             |
| `solve`                           | `torch.solve(input, A)`                 |
| `sort`                            | `torch.sort(input, dim, descending)`    |
| `sparse`                          | `torch.sparse(input)`                   |
| `sparse_bsc`                      | `torch.sparse_bsc(indices, values, size)` |
| `sparse_bsc_tensor`               | `torch.sparse_bsc_tensor(indices, values)` |
| `sparse_bsr`                      | `torch.sparse_bsr(indices, values, size)` |
| `sparse_bsr_tensor`               | `torch.sparse_bsr_tensor(indices, values)` |
| `sparse_compressed_tensor`        | `torch.sparse_compressed_tensor(sizes, values)` |
| `sparse_coo`                      | `torch.sparse_coo(indices, values, size)` |
| `sparse_coo_tensor`               | `torch.sparse_coo_tensor(indices, values)` |
| `sparse_csc`                      | `torch.sparse_csc(indices, values, size)` |
| `sparse_csc_tensor`               | `torch.sparse_csc_tensor(indices, values)` |
| `sparse_csr`                      | `torch.sparse_csr(indices, values, size)` |
| `sparse_csr_tensor`               | `torch.sparse_csr_tensor(indices, values)` |
| `special`                         | `torch.special(input)`                  |
| `split`                           | `torch.split(tensor, split_size, dim)`   |
| `split_copy`                      | `torch.split_copy(tensor, split_size, dim)` |
| `split_with_sizes`                | `torch.split_with_sizes(tensor, sizes, dim)` |
| `split_with_sizes_copy`           | `torch.split_with_sizes_copy(tensor, sizes, dim)` |
| `spmm`                            | `torch.spmm(input1, input2)`            |
| `sqrt`                            | `torch.sqrt(input)`                     |
| `sqrt_`                           | `torch.sqrt_(input)`                    |
| `square`                          | `torch.square(input)`                   |
| `square_`                         | `torch.square_(input)`                  |
| `squeeze`                         | `torch.squeeze(input, dim)`             |
| `squeeze_copy`                    | `torch.squeeze_copy(input, dim)`        |
| `sspaddmm`                        | `torch.sspaddmm(input, mat1, mat2, beta, alpha)` |
| `stack`                           | `torch.stack(tensors, dim)`             |
| `std`                             | `torch.std(input)`                      |
| `std_mean`                        | `torch.std_mean(input)`                 |
| `stft`                            | `torch.stft(input, ...) `               |
| `storage`                         | `torch.storage(...)`                    |
| `strided`                         | `torch.strided(input, ...)`             |
| `sub`                             | `torch.sub(input1, input2)`             |
| `subtract`                        | `torch.subtract(input1, input2)`        |
| `sum`                             | `torch.sum(input)`                      |
| `svd`                             | `torch.svd(input)`                      |
| `svd_lowrank`                     | `torch.svd_lowrank(input)`              |
| `swapaxes`                        | `torch.swapaxes(input, axis1, axis2)`   |
| `swapdims`                        | `torch.swapdims(input, dim1, dim2)`     |
| `sym_constrain_range`             | `torch.sym_constrain_range(input, min, max)` |
| `sym_constrain_range_for_size`    | `torch.sym_constrain_range_for_size(input, size)` |
| `sym_float`                       | `torch.sym_float(input)`                |
| `sym_int`                         | `torch.sym_int(input)`                  |
| `sym_ite`                         | `torch.sym_ite(cond, input1, input2)`   |
| `sym_max`                         | `torch.sym_max(input1, input2)`         |
| `sym_min`                         | `torch.sym_min(input1, input2)`         |
| `sym_not`                         | `torch.sym_not(input)`                  |
| `sym_sqrt`                        | `torch.sym_sqrt(input)`                 |
| `symeig`                          | `torch.symeig(input)`                   |
| `sys`                             | `torch.sys`                             |
## `T`

| Functionality              | Sample Snippet                         |
|----------------------------|----------------------------------------|
| `t`                        | `torch.t(input)`                      |
| `t_copy`                   | `torch.t_copy(input)`                 |
| `take`                     | `torch.take(input, indices)`          |
| `take_along_dim`           | `torch.take_along_dim(input, indices)`|
| `tan`                      | `torch.tan(input)`                    |
| `tan_`                     | `torch.tan_(input)`                   |
| `tanh`                     | `torch.tanh(input)`                   |
| `tanh_`                    | `torch.tanh_(input)`                  |
| `tensor`                   | `torch.tensor(data)`                  |
| `tensor_split`             | `torch.tensor_split(input, split_size, dim)` |
| `tensordot`                | `torch.tensordot(input1, input2, dims)` |
| `testing`                  | `torch.testing`                       |
| `textwrap`                 | `torch.textwrap`                      |
| `th_dll_path`              | `torch.th_dll_path()`                 |
| `threshold`                | `torch.threshold(input, threshold, value, inplace)` |
| `threshold_`               | `torch.threshold_(input, threshold, value)` |
| `tile`                     | `torch.tile(input, dims)`             |
| `to_dlpack`                | `torch.to_dlpack(tensor)`             |
| `topk`                     | `torch.topk(input, k, dim, largest, sorted)` |
| `torch`                    | `torch.torch`                         |
| `torch_version`            | `torch.torch_version`                 |
| `trace`                    | `torch.trace(input)`                  |
| `transpose`                | `torch.transpose(input, dim0, dim1)`  |
| `transpose_copy`           | `torch.transpose_copy(input, dim0, dim1)` |
| `trapezoid`                | `torch.trapezoid(input)`              |
| `trapz`                    | `torch.trapz(input)`                  |
| `triangular_solve`         | `torch.triangular_solve(input, ...) ` |
| `tril`                     | `torch.tril(input, diagonal)`         |
| `tril_indices`             | `torch.tril_indices(row, col, offset, dtype)` |
| `triplet_margin_loss`      | `torch.triplet_margin_loss(input1, input2, input3, margin)` |
| `triu`                     | `torch.triu(input, diagonal)`         |
| `triu_indices`             | `torch.triu_indices(row, col, offset, dtype)` |
| `true_divide`              | `torch.true_divide(input1, input2)`   |
| `trunc`                    | `torch.trunc(input)`                  |
| `trunc_`                   | `torch.trunc_(input)`                 |
| `typename`                 | `torch.typename(input)`               |
| `types`                    | `torch.types`                         |

## `u`
| Functionality              | Sample Snippet                         |
|----------------------------|----------------------------------------|
| `uint8`                    | `torch.uint8(input)`                  |
| `unbind`                   | `torch.unbind(input, dim)`             |
| `unbind_copy`              | `torch.unbind_copy(input, dim)`        |
| `unflatten`                | `torch.unflatten(input, dim, sizes)`   |
| `unfold_copy`              | `torch.unfold_copy(input, dim, size, step)` |
| `unify_type_list`          | `torch.unify_type_list(input)`        |
| `unique`                   | `torch.unique(input, sorted=False)`   |
| `unique_consecutive`       | `torch.unique_consecutive(input)`      |
| `unravel_index`            | `torch.unravel_index(indices, dims)`   |
| `unsafe_chunk`             | `torch.unsafe_chunk(input, chunks, dim)` |
| `unsafe_split`             | `torch.unsafe_split(input, split_size, dim)` |
| `unsafe_split_with_sizes`  | `torch.unsafe_split_with_sizes(input, split_sizes, dim)` |
| `unsqueeze`                | `torch.unsqueeze(input, dim)`         |
| `unsqueeze_copy`           | `torch.unsqueeze_copy(input, dim)`    |
| `use_deterministic_algorithms` | `torch.use_deterministic_algorithms(enabled)` |
| `utils`                    | `torch.utils`                         |

## `v`
| Functionality           | Sample Snippet                         |
|-------------------------|----------------------------------------|
| `values_copy`           | `torch.values_copy(input)`             |
| `vander`                | `torch.vander(input)`                  |
| `var`                   | `torch.var(input)`                     |
| `var_mean`              | `torch.var_mean(input)`                |
| `vdot`                  | `torch.vdot(input1, input2)`           |
| `version`               | `torch.version`                        |
| `view_as_complex`       | `torch.view_as_complex(input)`         |
| `view_as_complex_copy`  | `torch.view_as_complex_copy(input)`    |
| `view_as_real`          | `torch.view_as_real(input)`            |
| `view_as_real_copy`     | `torch.view_as_real_copy(input)`       |
| `view_copy`             | `torch.view_copy(input)`               |
| `vitals_enabled`        | `torch.vitals_enabled()`               |
| `vmap`                  | `torch.vmap(fn, input)`                |
| `vsplit`                | `torch.vsplit(tensor, split_size)`      |
| `vstack`                | `torch.vstack(tensors)`                |

## `w`
| Functionality           | Sample Snippet                         |
|-------------------------|----------------------------------------|
| `wait`                  | `torch.wait()`                         |
| `where`                 | `torch.where(condition, x, y)`         |
| `windows`               | `torch.windows()`                      |
| `with_load_library_flags` | `torch.with_load_library_flags(...)`  |

## `x`
| Functionality           | Sample Snippet                         |
|-------------------------|----------------------------------------|
| `xlogy`                 | `torch.xlogy(input1, input2)`          |
| `xlogy_`                | `torch.xlogy_(input1, input2)`         |

## `z`

| Functionality           | Sample Snippet                         |
|-------------------------|----------------------------------------|
| `zero_`                 | `torch.zero_(input)`                   |
| `zeros`                 | `torch.zeros(size)`                    |
| `zeros_like`            | `torch.zeros_like(input)`              |
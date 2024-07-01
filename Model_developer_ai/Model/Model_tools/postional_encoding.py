import torch
import torch.nn as nn
from typing import Tuple, Optional,Any
from torch import Tensor
import math


class AbstractPositionalEncoding(torch.nn.Module):
    """
    Abstract base class for positional encoding techniques.
     # Example Usage:
if __name__ == "__main__":
    # Define embedding dimension and sequence length
    d_model = 512
    seq_len = 100

    # Create sample input tensor
    input_tensor = torch.randn(1, seq_len, d_model)

    # Instantiate and apply SinePositionalEncoding
    sin_pe = SinePositionalEncoding(d_model=d_model)
    output_sin_pe = sin_pe(input_tensor)
    print("Shape after SinePositionalEncoding:", output_sin_pe.shape)

    # Instantiate and apply RotaryEmbedding
    rotary_emb = RotaryEmbedding(d_model=d_model)
    output_rotary = rotary_emb(input_tensor)
    print("Shape after RotaryEmbedding:", output_rotary.shape)
    This class enforces the implementation of forward pass and 
    provides a consistent interface for different positional encoding methods.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (Tensor): The input tensor to encode. Shape: (batch_size, sequence_length, embedding_dimension).
            **kwargs: Additional keyword arguments for specific encoding methods.

        Returns:
            Tensor: The input tensor with positional encoding applied. 
                    Shape: (batch_size, sequence_length, embedding_dimension).
        """
        raise NotImplementedError("Subclasses must implement the forward method!")


class SinePositionalEncoding(AbstractPositionalEncoding):
    """
    Applies sinusoidal positional encoding to a tensor.

    This encoding injects positional information into the input by adding 
    sinusoidal waves of different frequencies to each dimension of the embedding.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings.
        max_len (int, optional): The maximum sequence length to consider. Defaults to 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model)

        # Precompute the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the precomputed sinusoidal positional encoding to the input.

        Args:
            x (Tensor): The input tensor to encode. Shape: (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor: The input tensor with sinusoidal positional encoding applied. 
                    Shape: (batch_size, sequence_length, embedding_dimension).
        """
        return x + self.pe[:x.size(1), :] 
class RotaryEmbedding(AbstractPositionalEncoding):
    """
    Applies Rotary Positional Embeddings (RoPE) to a tensor.

    RoPE encodes position information by rotating pairs of dimensions in the embedding space.
    This method is particularly effective for modeling relative positional relationships.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings.
        max_len (int, optional): The maximum sequence length to consider. Defaults to 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model)

        # Precompute the rotation matrices for RoPE
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        inv_freq = position * div_term

        # Construct the rotation matrices (no einsolvable_closed_form needed)
        self.cos_cached = inv_freq.cos()[None, None, :, :].float()
        self.sin_cached = inv_freq.sin()[None, None, :, :].float()

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies Rotary Positional Embeddings (RoPE) to the input tensor.

        Args:
            x (Tensor): The input tensor to encode. Shape: (batch_size, sequence_length, embedding_dimension).

        Returns:
            Tensor: The input tensor with RoPE applied.
                    Shape: (batch_size, sequence_length, embedding_dimension).
        """
        seq_len = x.shape[1]
        # Apply rotation to even and odd dimensions separately
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        x_rotated_even = x_even * self.cos_cached[:, :, :seq_len, :] - x_odd * self.sin_cached[:, :, :seq_len, :]
        x_rotated_odd = x_odd * self.cos_cached[:, :, :seq_len, :] + x_even * self.sin_cached[:, :, :seq_len, :]

        return torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)



class AliBi(torch.nn.Module):
    """
    Implements the AliBi (Attention with Linear Biases) positional embedding.

    This module avoids using absolute positional embeddings, promoting scalability to
    longer sequences. It works by adding a constant bias to each attention score based
    on the distance between the query and key tokens.

    Args:
        num_heads (int): Number of attention heads.
        mp_size (int, optional): Model parallel size. Defaults to 1.
        mp_rank (int, optional): Model parallel rank. Defaults to 1.

    Attributes:
        mp_size (int): Model parallel size.
        mp_rank (int): Model parallel rank.
        num_heads (int): Number of attention heads.
        slice_size (int): Number of heads per model parallel partition.
        cached_matrix (torch.Tensor, optional): Cached AliBi matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.
        slopes (torch.Tensor): Tensor holding precomputed slopes for AliBi.

    Example:
        >>> alibi = AliBi(num_heads=8)
        >>> x = torch.randn(1, 8, 10, 10)  # Example input tensor
        >>> output = alibi(x)
        >>> print(output.shape)
        torch.Size([1, 8, 10, 10])
    """

    def __init__(self, num_heads: int, mp_size: int = 1, mp_rank: int = 1) -> None:
        super().__init__()
        # Ensure valid model parallel settings
        assert mp_size <= num_heads and mp_rank <= mp_size

        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n: int) -> list[float]:
        """
        Calculate slopes for AliBi positional embedding.

        For optimal performance, `n` (number of heads) should ideally be a power of 2.

        Args:
            n (int): Number of attention heads.

        Returns:
            list[float]: List of calculated slopes.
        """

        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_slopes(2 * closest_power_of_2)[0::2][
                : n - closest_power_of_2
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply AliBi bias to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Input tensor with AliBi bias added.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize or reuse the AliBi matrix
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = seq_len_k
            a = -torch.tril(
                torch.arange(target_seq_len, device=x.device)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1, device=x.device)
            )
            a = a.to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # Clip the AliBi matrix if it's larger than needed
        if self.cached_seq_len and self.cached_seq_len > seq_len_k:
            a = a[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # Handle inference with cached past keys
            assert (
                seq_len_q == 1
            ), "Assumes seq_len_q == seq_len_k except during inference with cached past keys (where seq_len_q == 1)"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # Use last token index from the current batch

        return x + a


class AliBi_CAPE(torch.nn.Module):
    """
    AliBi positional embedding combined with a Contextual Attention Pattern Enhancement (CAPE) module.

    This module extends the AliBi embedding by applying a learned transformation to the
    concatenation of the input and AliBi bias, allowing for more context-dependent adjustments.

    Args:
        num_heads (int): Number of attention heads.
        mp_size (int, optional): Model parallel size. Defaults to 1.
        mp_rank (int, optional): Model parallel rank. Defaults to 1.
        mlp_width (int, optional): Width of the MLP in the CAPE module. Defaults to 32.

    Attributes:
        mp_size (int): Model parallel size.
        mp_rank (int): Model parallel rank.
        num_heads (int): Number of attention heads.
        slice_size (int): Number of heads per model parallel partition.
        cached_matrix (torch.Tensor, optional): Cached AliBi matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.
        slopes (torch.Tensor): Tensor holding precomputed slopes for AliBi.
        mlp2 (torch.nn.Sequential): CAPE module's MLP.

    Example:
        >>> alibi_cape = AliBi_CAPE(num_heads=8)
        >>> x = torch.randn(1, 8, 10, 10)  # Example input tensor
        >>> output = alibi_cape(x)
        >>> print(output.shape)
        torch.Size([1, 8, 10, 10])
    """

    def __init__(
        self, num_heads: int, mp_size: int = 1, mp_rank: int = 1, mlp_width: int = 32
    ) -> None:
        super().__init__()
        # Ensure valid model parallel settings
        assert mp_size <= num_heads and mp_rank <= mp_size

        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads * 2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads),
        )

    def _get_slopes(self, n: int) -> list[float]:
        """
        Calculate slopes for AliBi positional embedding.

        For optimal performance, `n` (number of heads) should ideally be a power of 2.

        Args:
            n (int): Number of attention heads.

        Returns:
            list[float]: List of calculated slopes.
        """

        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_slopes(2 * closest_power_of_2)[0::2][
                : n - closest_power_of_2
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply AliBi bias and CAPE transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize or reuse the AliBi matrix
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = seq_len_k
            a = -torch.tril(
                torch.arange(target_seq_len, device=x.device)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1, device=x.device)
            )
            a = a.to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # Clip the AliBi matrix if it's larger than needed
        if self.cached_seq_len and self.cached_seq_len > seq_len_k:
            a = a[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # Handle inference with cached past keys
            assert (
                seq_len_q == 1
            ), "Assumes seq_len_q == seq_len_k except during inference with cached past keys (where seq_len_q == 1)"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # Use last token index from the current batch

        x_a_bias = torch.cat((x, torch.tile(a, (x.shape[0], 1, 1, 1))), dim=1)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))

        return x + a + x_a_bias


class ParallelKerpleLog(torch.nn.Module):
    """
    Kernelized T5 Relative Position Bias, parallelized across attention heads.

    This module implements a relative positional bias based on a logarithmic kernel.
    The bias calculation is parallelized across attention heads for efficiency.

    Args:
        neox_args (Any): Namespace-like object containing model arguments (expects 'num_attention_heads',
            'pos_emb', and 'params_dtype' attributes).

    Attributes:
        heads (int): Number of attention heads.
        num_heads_per_partition (int): Number of heads per partition (assumed to be equal to total heads).
        pos_emb (str): Positional embedding type (not used in this module).
        eps (float): Small epsilon value for numerical stability.
        bias_p (torch.nn.Parameter): Learnable parameter for the kernel.
        bias_a (torch.nn.Parameter): Learnable parameter for the kernel.
        cached_matrix (torch.Tensor, optional): Cached distance matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.

    Example:
        >>> neox_args = type('Namespace', (object,), {"num_attention_heads": 8, "pos_emb": "relative", "params_dtype": torch.float32})
        >>> kerple_log = ParallelKerpleLog(neox_args)
        >>> x = torch.randn(1, 8, 10, 10)  # Example input tensor
        >>> output = kerple_log(x)
        >>> print(output.shape)
        torch.Size([1, 8, 10, 10])
    """

    def __init__(self, neox_args: Any) -> None:
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale: float, init_method: str = "uniform") -> nn.Parameter:
            if init_method == "ones":
                return nn.Parameter(
                    torch.ones(
                        self.num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )[:, None, None]
                    * scale
                )
            if init_method == "uniform":
                return nn.Parameter(
                    torch.rand(
                        self.num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )[:, None, None]
                    * scale
                )
            raise ValueError(f"Invalid init_method: {init_method}")

        self.bias_p = get_parameter(2, "uniform")
        self.bias_a = get_parameter(1, "uniform")

        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

    def stats(self) -> dict[str, torch.Tensor]:
        """
        Calculate and return statistics of the learnable parameters.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing mean, standard deviation,
                maximum, and minimum values for 'bias_a' and 'bias_p'.
        """

        def get_stats(name: str, obj: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                name + "_mean": obj.mean().detach().cpu(),
                name + "_std": obj.std().detach().cpu(),
                name + "_max": obj.max().detach().cpu(),
                name + "_min": obj.min().detach().cpu(),
            }

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats("bias_a", self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats("bias_p", self.bias_p))
        return dd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the kernelized relative positional bias to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Input tensor with the bias added.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Calculate or use cached distance matrix
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(
                    1, seq_len_k
                )
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(
            1 + self.bias_a * diff
        )  # Calculate the log kernel bias

        if seq_len_q != seq_len_k:
            # Handle inference with cached past keys
            assert (
                seq_len_q == 1
            ), "Assumes seq_len_q == seq_len_k except during inference with cached past keys (where seq_len_q == 1)"

            if type(bias) != float:
                bias = bias[:, seq_len_k - 1, :].view(
                    bias.shape[0], 1, bias.shape[2]
                )  # Use last token index from the current batch

        return x + bias


class ParallelKerpleLog_CAPE(torch.nn.Module):
    """
    Kernelized T5 Relative Position Bias with CAPE, parallelized across attention heads.

    This module enhances the ParallelKerpleLog by adding a CAPE module
    after the bias calculation, providing contextual adjustments.

    Args:
        neox_args (Any): Namespace-like object containing model arguments (expects
            'num_attention_heads', 'pos_emb', 'params_dtype', and 'mlp_width' attributes).
        layer_number (int): Layer number for visualization (not used directly in calculations).

    Attributes:
        heads (int): Number of attention heads.
        num_heads_per_partition (int): Number of heads per partition (assumed to be equal to total heads).
        pos_emb (str): Positional embedding type (not used in this module).
        eps (float): Small epsilon value for numerical stability.
        layer_number_visualization (int): Layer number for visualization.
        bias_p (torch.nn.Parameter): Learnable parameter for the kernel.
        bias_a (torch.nn.Parameter): Learnable parameter for the kernel.
        cached_matrix (torch.Tensor, optional): Cached distance matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.
        mlp2 (torch.nn.Sequential): CAPE module's MLP.

    Example:
        >>> neox_args = type('Namespace', (object,), {"num_attention_heads": 8, "pos_emb": "relative", "params_dtype": torch.float32, "mlp_width": 32})
        >>> kerple_log_cape = ParallelKerpleLog_CAPE(neox_args, layer_number=1)
        >>> x = torch.randn(1, 8, 10, 10)  # Example input tensor
        >>> output = kerple_log_cape(x)
        >>> print(output.shape)
        torch.Size([1, 8, 10, 10])
    """

    def __init__(self, neox_args: Any, layer_number: int) -> None:
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.num_heads_per_partition = self.heads
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        self.layer_number_visualization = layer_number

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale: float, init_method: str = "uniform") -> nn.Parameter:
            if init_method == "ones":
                return nn.Parameter(
                    torch.ones(
                        self.num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )[:, None, None]
                    * scale
                )
            if init_method == "uniform":
                return nn.Parameter(
                    torch.rand(
                        self.num_heads_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )[:, None, None]
                    * scale
                )
            raise ValueError(f"Invalid init_method: {init_method}")

        self.bias_p = get_parameter(2, "uniform")
        self.bias_a = get_parameter(1, "uniform")

        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_heads_per_partition * 2, neox_args.mlp_width),
            nn.LeakyReLU(),
            nn.Linear(neox_args.mlp_width, self.num_heads_per_partition),
        )

    def stats(self) -> dict[str, torch.Tensor]:
        """
        Calculate and return statistics of the learnable parameters.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing mean, standard deviation,
                maximum, and minimum values for 'bias_a' and 'bias_p'.
        """

        def get_stats(name: str, obj: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                name + "_mean": obj.mean().detach().cpu(),
                name + "_std": obj.std().detach().cpu(),
                name + "_max": obj.max().detach().cpu(),
                name + "_min": obj.min().detach().cpu(),
            }

        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats("bias_a", self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats("bias_p", self.bias_p))
        return dd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the kernelized relative positional bias and CAPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Calculate or use cached distance matrix
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(
                    1, seq_len_k
                )
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(
            1 + self.bias_a * diff
        )  # Calculate the log kernel bias

        if seq_len_q != seq_len_k:
            # Handle inference with cached past keys
            assert (
                seq_len_q == 1
            ), "Assumes seq_len_q == seq_len_k except during inference with cached past keys (where seq_len_q == 1)"

            if type(bias) != float:
                bias = bias[:, seq_len_k - 1, :].view(
                    bias.shape[0], 1, bias.shape[2]
                )  # Use last token index from the current batch

        x_a_bias = torch.cat((x, torch.tile(bias, (x.shape[0], 1, 1, 1))), dim=1)
        x_a_bias = torch.permute(x_a_bias, (0, 2, 3, 1))
        x_a_bias = self.mlp2(x_a_bias)
        x_a_bias = torch.permute(x_a_bias, (0, 3, 1, 2))

        return x + bias + x_a_bias


class FIRE(nn.Module):
    """
    FIRE (Fully Interpretable Relative Encoding) positional embedding.

    FIRE represents relative distances between tokens using a combination of logarithmic
    transformation, thresholding, and an MLP. This allows for a more expressive
    and interpretable positional encoding compared to some alternatives.

    Args:
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_width (int, optional): Width of the MLP. Defaults to 32.
        init_c (float, optional): Initial value for the log transformation parameter. Defaults to 0.1.
        init_L (float, optional): Initial value for the distance threshold. Defaults to 512.0.
        eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-6.

    Attributes:
        mlp (torch.nn.Sequential): MLP for transforming relative distances.
        c (torch.nn.Parameter): Learnable parameter for the log transformation.
        init_L (torch.nn.Parameter): Initial distance threshold (non-trainable).
        L_multiplier (torch.nn.Parameter): Learnable multiplier for the distance threshold.
        eps (float): Small epsilon for numerical stability.
        cached_matrix (torch.Tensor, optional): Cached distance matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.

    Example:
        >>> fire = FIRE()
        >>> x = torch.randn(1, 12, 10, 10)  # Example input tensor
        >>> output = fire(x)
        >>> print(output.shape)
        torch.Size([1, 12, 10, 10])
    """

    def __init__(
        self,
        num_heads: int = 12,
        mlp_width: int = 32,
        init_c: float = 0.1,
        init_L: float = 512.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the FIRE positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Input tensor with the positional encoding added.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Calculate or use cached distance matrix
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(
                    1, seq_len_k
                )
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        pos_normalizer = torch.max(rel_distance_max, threshold)
        pos_normalizer = pos_normalizer[:, None]

        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = (
            torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps
        )

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        return x + fire_bias


class FIRE_CAPE(nn.Module):
    """
    FIRE (Fully Interpretable Relative Encoding) with CAPE positional embedding.

    This module combines FIRE with a CAPE module to further enhance the expressiveness
    of the positional encoding by allowing for contextual adjustments.

    Args:
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_width (int, optional): Width of the MLPs. Defaults to 32.
        init_c (float, optional): Initial value for the log transformation parameter. Defaults to 0.1.
        init_L (float, optional): Initial value for the distance threshold. Defaults to 512.0.
        eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-6.

    Attributes:
        mlp (torch.nn.Sequential): MLP for transforming relative distances in FIRE.
        mlp2 (torch.nn.Sequential): MLP for the CAPE module.
        c (torch.nn.Parameter): Learnable parameter for the log transformation in FIRE.
        init_L (torch.nn.Parameter): Initial distance threshold (non-trainable) in FIRE.
        L_multiplier (torch.nn.Parameter): Learnable multiplier for the distance threshold in FIRE.
        eps (float): Small epsilon for numerical stability.
        cached_matrix (torch.Tensor, optional): Cached distance matrix. Defaults to None.
        cached_seq_len (int, optional): Cached sequence length. Defaults to None.

    Example:
        >>> fire_cape = FIRE_CAPE()
        >>> x = torch.randn(1, 12, 10, 10)  # Example input tensor
        >>> output = fire_cape(x)
        >>> print(output.shape)
        torch.Size([1, 12, 10, 10])
    """

    def __init__(
        self,
        num_heads: int = 12,
        mlp_width: int = 32,
        init_c: float = 0.1,
        init_L: float = 512.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.LeakyReLU(), nn.Linear(mlp_width, num_heads)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(num_heads * 2, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads),
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

        self.cached_matrix: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the FIRE_CAPE positional encoding to the input tensor.

        Args:            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len_q, seq_len_k].

        Returns:
            torch.Tensor: Input tensor with the positional encoding added.
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Calculate or use cached distance matrix
        if self.cached_seq_len != seq_len_k:
            rel_distance = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(
                    1, seq_len_k
                )
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            rel_distance = rel_distance.to(torch.float32)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix

        threshold = torch.abs(self.L_multiplier * self.init_L)
        rel_distance_max = torch.max(torch.tril(rel_distance), dim=-1)[0]
        pos_normalizer = torch.max(rel_distance_max, threshold)
        pos_normalizer = pos_normalizer[:, None]

        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = (
            torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps
        )

        normalized_distance = rel_distance / pos_normalizer
        normalized_distance = normalized_distance.to(x.dtype)

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        # Apply CAPE
        x_fire_bias = torch.cat(
            (x, torch.tile(fire_bias, (x.shape[0], 1, 1, 1))), dim=1
        )
        x_fire_bias = x_fire_bias.permute(0, 2, 3, 1)
        x_fire_bias = self.mlp2(x_fire_bias)
        x_fire_bias = x_fire_bias.permute(0, 3, 1, 2)

        return x + fire_bias + x_fire_bias
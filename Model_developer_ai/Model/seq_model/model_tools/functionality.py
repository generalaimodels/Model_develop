from typing import Tuple,List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Extracts patches from an image and embeds them with positional information.

    Args:
        image_size (Tuple[int, int, int]): Size of the input image (channels, height, width).
        patch_size (int): Size of the square patches to extract.
        embedding_dim (int): Dimensionality of the patch embeddings.
    example:
    # Example usage:
                    image_size = (3, 224, 224)  # Example image size (channels, height, width)
                    patch_size = 16  # Size of the patches
                    embedding_dim = 768  # Dimensionality of the patch embeddings
                    # Create an instance of the PatchEmbedding layer
                    patch_embedding_layer = PatchEmbedding(image_size, patch_size, embedding_dim)
                    # Example input image tensor
                    image = torch.randn(16, *image_size)  # Example batch size of 16
                    # Get the patch embeddings
                    embeddings = patch_embedding_layer(image)
                    # Print the shape of the output tensor
                    print(embeddings.shape)  # Should be (16, 196, 768)
    """
    def __init__(
        self,
        image_size: Tuple[int, int, int],
        patch_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # Calculate the number of patches in each dimension
        self.num_patches_h = image_size[1] // patch_size
        self.num_patches_w = image_size[2] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Create a convolutional layer to extract patches
        self.patch_extractor = nn.Conv2d(
            in_channels=image_size[0],
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Create a learnable positional embedding matrix
        self.position_embeddings = nn.Parameter(
            torch.randn(self.num_patches, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch embedding layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_patches, embedding_dim) containing the
                patch embeddings with positional information.
        """
        # Extract patches using the convolutional layer
        patches = self.patch_extractor(x)  # (batch_size, embedding_dim, num_patches_h, num_patches_w)

        # Flatten the patches along the spatial dimensions
        patches = patches.flatten(2)  # (batch_size, embedding_dim, num_patches)

        # Transpose to get the desired output shape
        patches = patches.transpose(1, 2)  # (batch_size, num_patches, embedding_dim)

        # Add the learnable positional embeddings
        embeddings = patches + self.position_embeddings

        return embeddings



def attn_chunk_first(
    query: torch.Tensor,
    prefix_tree: List[torch.Tensor],
    cache: dict
) -> torch.Tensor:
    """
    Implements the "Self Attention: Chunk First" algorithm with partial attention caching.

    This function processes chunks of the prefix tree, computes partial attention results,
    and stores them in a cache for efficient reuse.

    Args:
        query (Tensor): The query tensor of shape (batch_size, sequence_length, embedding_dim).
        prefix_tree (List[Tensor]): The prefix tree represented as a list of tensors,
            where each tensor represents a chunk.
        cache (dict): A dictionary to store and retrieve partial attention results.

    Returns:
        Tensor: The final attention output tensor of shape (batch_size, sequence_length, embedding_dim).
    # Example usage:
    embedding_dim = (32, 123, 1024)
    batch_size = embedding_dim[0]
    # Example tensors for demonstration
    query = torch.randn(embedding_dim)
    # Create prefix_tree with both key and value tensors for each chunk
    prefix_tree = [
        (torch.randn((embedding_dim[1], embedding_dim[2])),  # Key tensor for chunk i
         torch.randn((embedding_dim[1], embedding_dim[2])))  # Value tensor for chunk i
        for i in range(4)
    ] 
    cache = {}  # Initialize an empty cache
    # Perform attention using the chunk-first approach
    output = attn_chunk_first(query, prefix_tree, cache)
    print(output.shape) 
    """

    batch_size, sequence_length, embedding_dim = query.shape  # Correct unpacking
    attention_output = torch.zeros_like(query)

    for chunk_index, chunk in enumerate(prefix_tree):
        # Check if the result for this chunk is already cached
        cache_key = f"chunk_{chunk_index}"
        if cache_key in cache:
            partial_output, start_index, end_index = cache[cache_key]
        else:    
            key_cache, value_cache = chunk  # Unpack key and value from the chunk           
            # Calculate start and end indices using key_cache.shape or value_cache.shape
            start_index = chunk_index * key_cache.shape[0] // batch_size 
            end_index = (chunk_index + 1) * key_cache.shape[0] // batch_size 
            # Compute partial attention for the current chunk
            partial_output = _partial_attn(
                query[:, start_index:end_index, :], key_cache, value_cache
            )
            # Cache the partial result
            cache[cache_key] = (partial_output, start_index, end_index)
        # Update the final attention output using the cached partial result
        attention_output[:, start_index:end_index, :] += partial_output
    return attention_output


def _partial_attn(query: torch.Tensor, key: torch.Tensor, value:torch.Tensor) ->torch.Tensor:
    """
    Computes the scaled dot-product attention for a single chunk.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.

    Returns:
        Tensor: The attention output for the given chunk.
    """
    # Scaling factor for the dot product
    scale_factor = 1.0 / (key.shape[-1] ** 0.5)

    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Apply softmax to obtain attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Calculate the weighted sum of values
    return torch.matmul(attention_weights, value)

def attn_reduce(
    o_c: torch.Tensor,
    m_c: torch.Tensor,
    n_c: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> None:
    """
    Helper function to reduce partial attention results.

    This function updates the overall attention output (o, m, n)
    with partial results (o_c, m_c, n_c) from a chunk.

    Args:
        o_c: Partial attention output.
        m_c: Partial attention mask.
        n_c: Partial normalization term.
        o: Overall attention output (in-place update).
        m: Overall attention mask (in-place update).
        n: Overall normalization term (in-place update).
    Example :
      # Sample data with embedding_dim = (16, 196, 768)
          batch_size = 32
          seq_len = 416
          embedding_dim = 768
          # Queries
          q = torch.randn(batch_size, seq_len, embedding_dim)
          # Prefix tree (example - adjust based on your data)
          total_seq_len = 512  # Example: Total sequence length in your dataset 
          t = [
              (torch.randn(batch_size, total_seq_len, embedding_dim), torch.randn(batch_size, total_seq_len, embedding_dim)),
              # ... Add more (key, value) pairs to represent the prefix tree
          ]
          # Calculate attention
          output = attnseqfirst(q, t, embedding_dim=(batch_size, seq_len, embedding_dim))
          print(output.shape)

    """
    o = o + o_c  # Update how we add o_c to o
    m = torch.maximum(m, m_c)  # Update mask (element-wise maximum)
    n = n + n_c 

def partial_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute partial attention. Handles cases where end_idx > k.shape[1]."""
    k_slice = k[:, start_idx:end_idx]  # (batch_size, chunk_size, embedding_dim)
    v_slice = v[:, start_idx:end_idx]  # (batch_size, chunk_size, embedding_dim)

    # Calculate attention
    attn_logits = q @ k_slice.transpose(-2, -1)  # (batch_size, 1, chunk_size)
    attn_weights = F.softmax(attn_logits, dim=-1)

    o = attn_weights @ v_slice  # (batch_size, 1, embedding_dim)
    m = attn_weights.max(dim=-1).values  # (batch_size, 1)
    n = attn_weights.sum(dim=-1)  # (batch_size, 1)
    return o, m, n

def attnseqfirst(
    q: torch.Tensor,
    t: List[Tuple[torch.Tensor, torch.Tensor]],
    embedding_dim: Tuple[int, int, int] = (16, 196, 768),
) -> torch.Tensor:
    batch_size, seq_len, _ = embedding_dim
    o = torch.zeros_like(q)
    m = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    n = torch.zeros((batch_size, seq_len), dtype=torch.float32)

    for q_idx in range(batch_size):
        for i in range(seq_len):
            for c_idx, (k_c, v_c) in enumerate(t):
                end_idx = min(i + 1, k_c.shape[1])
                if c_idx < i:
                    o_c, m_c, n_c = partial_attn(q[q_idx:q_idx+1, i:i+1], k_c, v_c, i, end_idx)
                    attn_reduce(
                        o_c,
                        m_c,
                        n_c,
                        o[q_idx:q_idx+1, i:i+1],
                        m[q_idx:q_idx+1, i:i+1],
                        n[q_idx:q_idx+1, i:i+1],
                    )
                else:
                    o_c, m_c, n_c = partial_attn(q[q_idx:q_idx+1, i:i+1], k_c, v_c, i, end_idx)
                    attn_reduce(
                        o_c,
                        m_c,
                        n_c,
                        o[q_idx:q_idx+1, i:i+1],
                        m[q_idx:q_idx+1, i:i+1],
                        n[q_idx:q_idx+1, i:i+1],
                    )
    return o


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) implementation.

    Args:
        input_dim (int): The dimension of the input vector.
        hidden_dims (Tuple[int, ...]): A tuple specifying the number of neurons in each hidden layer.
        output_dim (int): The dimension of the output vector.
        activation (nn.Module, optional): The activation function to use. Defaults to nn.ReLU().

    Example:
        >>> mlp = MLP(input_dim=768, hidden_dims=(512, 256), output_dim=768)
        >>> input_tensor = torch.randn(16, 196, 768)
        >>> output_tensor = mlp(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 196, 768])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(activation)

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(activation)

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.

    This layer applies a normalization technique that focuses on re-scaling invariance.
    It normalizes the summed inputs based on the root mean square (RMS) statistic.

    Args:
        normalized_shape (Tuple[int, ...]): Input shape from an expected input of
            size  (*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]).
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.

    Attributes:
        weight (torch.Tensor): Learnable scaling parameter.
        normalized_shape (Tuple[int, ...]): Shape of the input to be normalized.

    Example:
        >>> norm = RMSNorm((16, 196, 768))
        >>> input = torch.randn(2, 16, 196, 768)
        >>> output = norm(input)
        >>> print(output.shape)
        torch.Size([2, 16, 196, 768])
    """

    def __init__(self, normalized_shape: Tuple[int, ...]) -> None:
        super(RMSNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        input_rms = torch.sqrt(
            torch.mean(input**2, dim=-1, keepdim=True) + 1e-8
        )  # Add a small epsilon for numerical stability
        normalized_input = input / input_rms
        return self.weight * normalized_input
    


class KANConv(nn.Module):
    """
    Implements a KAN-based Convolutional layer (KANConv).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        grid_size (int, optional): Number of grid points for B-spline interpolation. Defaults to 4.
        basis_function (nn.Module, optional): Basis function to use in KAN. Defaults to nn.SiLU().

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the convolutional kernel.
        grid_size (int): Number of grid points for B-spline interpolation.
        basis_function (nn.Module): Basis function used in KAN.
        spline_weights (torch.Tensor): Learnable weights for B-spline interpolation.
        basis_weights (torch.Tensor): Learnable weights for the basis function.
    EXample:
    # Example usage
                 if __name__ == "__main__":
                     batch_size = 16
                     input_channels = 196
                     output_channels = 768
                     height = 196
                     width = 768
                    `
                     # Create a sample input tensor
                     input_tensor = torch.randn(1, input_channels, height, width)
                 
                     # Create a KANConv layer
                     kan_conv = KANConv(input_channels, output_channels, kernel_size=3)
                 
                     # Pass the input through the KANConv layer
                     output_tensor = kan_conv(input_tensor)
                 
                     # Print the shape of the output tensor
                     print(f"Input shape: {input_tensor.shape}")
                     print(f"Output shape: {output_tensor.shape}")

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        grid_size: int = 4,
        basis_function: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.basis_function = basis_function

        # Initialize learnable weights for B-spline and basis function
        self.spline_weights = nn.Parameter(
            torch.randn(
                out_channels, in_channels, *self.kernel_size, self.grid_size
            )
        )
        self.basis_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the KANConv layer.
    
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
    
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
            """
            x = torch.nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
            batch_size, _, height, width = x.size()
            output = torch.zeros(
                batch_size,
                self.out_channels,
                height - self.kernel_size[0] + 1,
                width - self.kernel_size[1] + 1,
                device=x.device,
            )
    
            for i in range(height - self.kernel_size[0] + 1):
                for j in range(width - self.kernel_size[1] + 1):
                    # Extract local patches from the input
                    input_patch = x[
                        :,
                        :,
                        i : i + self.kernel_size[0],
                        j : j + self.kernel_size[1],
                    ]
                    # Apply B-spline interpolation
                    spline_output = (
                        self.spline_weights[None, :, :, :, :, :] * input_patch[:, None, :, :, :].unsqueeze(-1)
                    ).sum(-1)
    
                    # Apply basis function
                    basis_output = self.basis_function(input_patch[:, None, :, :, :]) * self.basis_weights[None, :, :, :, :]
    
                    # Combine outputs and accumulate over kernel elements
                    output[:, :, i, j] = (spline_output + basis_output).sum(dim=(2, 3, 4))
    
            return output


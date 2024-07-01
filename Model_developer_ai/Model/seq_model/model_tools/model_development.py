import torch
from typing import Tuple, Any, Optional, List
from torch import LongTensor, Tensor
import torch.nn as nn

# Define constants for clarity and maintainability
EXPERT_COUNT = 4  # Number of expert networks within the MoE layer
TOP_K = 2  # Number of experts to select for each input token

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def rotational_positional_embeddings(
    batch_size: int,
    sequence_length: int,
    dimensions: int,
    frequency_scale: float = 1.0,
) -> Tensor:
    """
    Generate rotational positional embeddings using a Fourier series.

    This function creates positional embeddings by representing positions as
    angles on a circle. The embeddings are then calculated as a weighted sum
    of sine and cosine functions at various frequencies. This approach allows
    the model to learn relative position information effectively.

    Args:
        batch_size: The size of the batch.
        sequence_length: The length of the sequence.
        dimensions: The dimensionality of the embeddings.
        frequency_scale: A scaling factor for the frequencies of the Fourier series.

    Returns:
        A tensor of shape (batch_size, sequence_length, dimensions) containing
        the rotational positional embeddings.

    # Example usage
                   batch_size = 16
                   sequence_length = 1024
                   dimensions = 2048
                   # Create a sample tensor
                   tensor = torch.randn(batch_size, sequence_length, dimensions)
                   # Add rotational positional embeddings
                   tensor_with_embeddings = add_rotational_positional_embeddings(
                       tensor, frequency_scale=1.0
                   )
                   # Print the shape of the resulting tensor (should be the same as the input)
                   print(tensor_with_embeddings.shape)  # Output:
    """

    positions = torch.arange(sequence_length).unsqueeze(1)
    frequencies = frequency_scale * torch.arange(0, dimensions, 2) / dimensions
    angle_rates = positions * frequencies
    embeddings = torch.zeros(batch_size, sequence_length, dimensions)

    embeddings[:, :, 0::2] = torch.sin(angle_rates)
    embeddings[:, :, 1::2] = torch.cos(angle_rates)

    return embeddings


def add_rotational_positional_embeddings(
    tensor: Tensor, frequency_scale: float = 1.0
) -> Tensor:
    """
    Add rotational positional embeddings to a given tensor.

    Args:
        tensor: The input tensor of shape (batch_size, sequence_length, dimensions).
        frequency_scale: A scaling factor for the frequencies of the Fourier series.

    Returns:
        The input tensor with rotational positional embeddings added,
        maintaining the same shape: (batch_size, sequence_length, dimensions).
    """

    batch_size, sequence_length, dimensions = tensor.size()
    positional_embeddings = rotational_positional_embeddings(
        batch_size, sequence_length, dimensions, frequency_scale
    )
    return tensor + positional_embeddings


def advanced_normalization(user_dim: Tuple[int, int, int]) -> Tensor:
    """
    Perform advanced normalization techniques on the input tensor.

    Args:
        user_dim (Tuple[int, int, int]): The dimensions of the input tensor.

    Returns:
        torch.Tensor: The normalized tensor with the same dimensions as the input.
    """
    assert len(user_dim) == 3, "Input tensor must have 3 dimensions"
    assert user_dim[0] > 0 and user_dim[1] > 0 and user_dim[2] > 0, "Dimensions must be positive integers"

    # Create a random tensor with the specified dimensions
    tensor = torch.randn(user_dim)

    # Perform square root mean normalization
    tensor = tensor / torch.sqrt(tensor.mean())

    # Perform correlation normalization
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, keepdim=True)
    tensor = (tensor - mean) / (std + 1e-8)
    return tensor


class AdvancedSelfAttention(torch.nn.Module):
    """
    Advanced self-attention implementation with key-value caching, sliding window,
    and optional relative positional encodings.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        sliding_window_size (int, optional): Size of the sliding window for local attention.
                                                Defaults to None (global attention).
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        use_rel_pos_enc (bool, optional): Whether to use relative positional encodings.
                                            Defaults to False.
        max_rel_pos (int, optional): Maximum relative position to consider for positional encodings.
                                        Required if `use_rel_pos_enc` is True.

    # Example usage
       batch_size = 16
       seq_len = 512
       embed_dim = 768
       num_heads = 12
       head_dim = 64

       # Create an instance of the attention module
       attention = AdvancedSelfAttention(
           embed_dim=embed_dim,
           num_heads=num_heads,
           head_dim=head_dim,
           sliding_window_size=128,  # Optional sliding window size
       )

       # Create some example input data
       input_tensor = torch.randn(batch_size, seq_len, embed_dim)

       # Perform self-attention
       output, _ = attention(input_tensor)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        sliding_window_size: Optional[int] = None,
        dropout: float = 0.1,
        use_rel_pos_enc: bool = False,
        max_rel_pos: Optional[int] = None,
    ) -> None:
        super().__init__()
        if use_rel_pos_enc and max_rel_pos is None:
            raise ValueError("`max_rel_pos` must be specified when `use_rel_pos_enc` is True.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window_size = sliding_window_size
        self.scaling = self.head_dim**-0.5
        self.use_rel_pos_enc = use_rel_pos_enc

        self.query_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.key_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.value_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.output_proj = torch.nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        if self.use_rel_pos_enc:
            self.rel_pos_embeddings = torch.nn.Embedding(2 * max_rel_pos + 1, self.num_heads)

        self.reset_cache()

    def reset_cache(self) -> None:
        """Resets the key-value cache."""
        self.cache = {"keys": None, "values": None}

    def _split_heads(self, x: Tensor) -> Tensor:
        """Splits the input tensor into multiple attention heads."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """Combines the multiple attention heads into a single tensor."""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

    def _apply_sliding_window(self, attention_scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Applies the sliding window to the attention scores."""
        if self.sliding_window_size is None:
            return attention_scores

        seq_len = attention_scores.size(-1)
        if seq_len <= self.sliding_window_size:
            return attention_scores

        # Create a mask for the sliding window
        sliding_window_mask = torch.ones_like(attention_scores).triu(diagonal=self.sliding_window_size)
        sliding_window_mask = sliding_window_mask.tril(diagonal=self.sliding_window_size)

        # Apply the mask and combine with the original mask
        attention_scores = attention_scores.masked_fill(~sliding_window_mask.bool(), float("-inf"))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.bool(), float("-inf"))

        return attention_scores

    def _compute_relative_position_bias(self, seq_len: int) -> Tensor:
        """Computes the relative position bias for relative positional encodings."""
        if not self.use_rel_pos_enc:
            return 0

        pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.rel_pos_embeddings.weight.device)
        rel_pos_ids = pos_ids[None, :] - pos_ids[:, None]
        rel_pos_ids += self.rel_pos_embeddings.num_embeddings // 2  # Shift to center around 0
        rel_pos_bias = self.rel_pos_embeddings(rel_pos_ids).permute(2, 0, 1)
        return rel_pos_bias

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[LongTensor] = None, # Using LongTensor for the mask
        cache: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Performs the self-attention operation.

        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (Tensor, optional): Key tensor of shape (batch_size, seq_len, embed_dim).
                                    Defaults to None (use query as key).
            value (Tensor, optional): Value tensor of shape (batch_size, seq_len, embed_dim).
                                    Defaults to None (use query as value).
            mask (LongTensor, optional): Attention mask (should be LongTensor). Defaults to None.
            cache (bool, optional): Whether to use key-value caching. Defaults to False.

        Returns:
            Tuple[Tensor, Any]: The output tensor and the updated cache (if cache is True).
        """
        batch_size, seq_len, _ = query.size()

        # Use query as key and value if not provided
        key = query if key is None else key
        value = query if value is None else value

        # Project the query, key, and value
        query = self._split_heads(self.query_proj(query))
        key = self._split_heads(self.key_proj(key))
        value = self._split_heads(self.value_proj(value))

        # Concatenate cached keys and values if using cache
        if cache:
            if self.cache["keys"] is not None:
                key = torch.cat([self.cache["keys"], key], dim=2)
            if self.cache["values"] is not None:
                value = torch.cat([self.cache["values"], value], dim=2)
            self.cache["keys"] = key
            self.cache["values"] = value

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling

        # Add relative position bias
        attention_scores += self._compute_relative_position_bias(seq_len)

        # Apply sliding window and mask
        attention_scores = self._apply_sliding_window(attention_scores, mask)

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Calculate weighted sum of values
        attended_values = torch.matmul(attention_weights, value)

        # Combine attention heads
        output = self._combine_heads(attended_values)

        # Project the output
        output = self.output_proj(output)

        return output, self.cache


class AdvancedCrossAttention(torch.nn.Module):
    """
    Advanced cross-attention implementation with key-value caching, sliding window,
    and optional relative positional encodings.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        sliding_window_size (int, optional): Size of the sliding window for local attention.
                                                Defaults to None (global attention).
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        use_rel_pos_enc (bool, optional): Whether to use relative positional encodings.
                                            Defaults to False.
        max_rel_pos (int, optional): Maximum relative position to consider for positional encodings.
                                        Required if `use_rel_pos_enc` is True.


       # Example usage
       batch_size = 16
       seq_len = 512
       embed_dim = 768
       num_heads = 12
       head_dim = 64

       # Create an instance of the attention module
       attention = AdvancedCrossAttention(
           embed_dim=embed_dim,
           num_heads=num_heads,
           head_dim=head_dim,
           sliding_window_size=128,  # Optional sliding window size
       )

       # Create some example input data
       input_tensor = torch.randn(batch_size, seq_len, embed_dim)

       # Perform self-attention
       output, _ = attention(input_tensor,input_tensor ,input_tensor )

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        sliding_window_size: Optional[int] = None,
        dropout: float = 0.1,
        use_rel_pos_enc: bool = False,
        max_rel_pos: Optional[int] = None,
    ) -> None:
        super().__init__()
        if use_rel_pos_enc and max_rel_pos is None:
            raise ValueError("`max_rel_pos` must be specified when `use_rel_pos_enc` is True.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window_size = sliding_window_size
        self.scaling = self.head_dim**-0.5
        self.use_rel_pos_enc = use_rel_pos_enc

        self.query_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.key_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.value_proj = torch.nn.Linear(embed_dim, num_heads * head_dim)
        self.output_proj = torch.nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        if self.use_rel_pos_enc:
            self.rel_pos_embeddings = torch.nn.Embedding(2 * max_rel_pos + 1, self.num_heads)

        self.reset_cache()

    def reset_cache(self) -> None:
        """Resets the key-value cache."""
        self.cache = {"keys": None, "values": None}

    def _split_heads(self, x: Tensor) -> Tensor:
        """Splits the input tensor into multiple attention heads."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """Combines the multiple attention heads into a single tensor."""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

    def _apply_sliding_window(self, attention_scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Applies the sliding window to the attention scores."""
        if self.sliding_window_size is None:
            return attention_scores

        query_len = attention_scores.size(-2)
        key_len = attention_scores.size(-1)

        if query_len <= self.sliding_window_size or key_len <= self.sliding_window_size:
            return attention_scores

        # Create a mask for the sliding window
        sliding_window_mask = torch.ones_like(attention_scores).triu(diagonal=self.sliding_window_size)
        sliding_window_mask = sliding_window_mask.tril(diagonal=self.sliding_window_size)

        # Apply the mask and combine with the original mask
        attention_scores = attention_scores.masked_fill(~sliding_window_mask.bool(), float("-inf"))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.bool(), float("-inf"))

        return attention_scores

    def _compute_relative_position_bias(self, query_len: int, key_len: int) -> Tensor:
        """Computes the relative position bias for relative positional encodings."""
        if not self.use_rel_pos_enc:
            return 0

        query_pos_ids = torch.arange(query_len, dtype=torch.long, device=self.rel_pos_embeddings.weight.device)
        key_pos_ids = torch.arange(key_len, dtype=torch.long, device=self.rel_pos_embeddings.weight.device)
        rel_pos_ids = query_pos_ids[None, :, None] - key_pos_ids[None, None, :]
        rel_pos_ids += self.rel_pos_embeddings.num_embeddings // 2  # Shift to center around 0
        rel_pos_bias = self.rel_pos_embeddings(rel_pos_ids).permute(0, 3, 1, 2)
        return rel_pos_bias

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[LongTensor] = None, # Using LongTensor for the mask
        cache: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Performs the cross-attention operation.

        Args:
            query (Tensor): Query tensor of shape (batch_size, query_len, embed_dim).
            key (Tensor): Key tensor of shape (batch_size, key_len, embed_dim).
            value (Tensor): Value tensor of shape (batch_size, key_len, embed_dim).
            mask (LongTensor, optional): Attention mask (should be LongTensor). Defaults to None.
            cache (bool, optional): Whether to use key-value caching. Defaults to False.

        Returns:
            Tuple[Tensor, Any]: The output tensor and the updated cache (if cache is True).
        """
        # Project the query, key, and value
        query = self._split_heads(self.query_proj(query))  # (batch_size, num_heads, query_len, head_dim)
        key = self._split_heads(self.key_proj(key))  # (batch_size, num_heads, key_len, head_dim)
        value = self._split_heads(self.value_proj(value))  # (batch_size, num_heads, key_len, head_dim)

        # Concatenate cached keys and values if using cache
        if cache:
            if self.cache["keys"] is not None:
                key = torch.cat([self.cache["keys"], key], dim=2)
            if self.cache["values"] is not None:
                value = torch.cat([self.cache["values"], value], dim=2)
            self.cache["keys"] = key
            self.cache["values"] = value

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  # (batch_size, num_heads, query_len, key_len)

        # Add relative position bias
        attention_scores += self._compute_relative_position_bias(query.size(2), key.size(2))

        # Apply sliding window and mask
        attention_scores = self._apply_sliding_window(attention_scores, mask)

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, query_len, key_len)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Calculate weighted sum of values
        attended_values = torch.matmul(attention_weights, value)  # (batch_size, num_heads, query_len, head_dim)

        # Combine attention heads
        attended_values = self._combine_heads(attended_values)  # (batch_size, query_len, num_heads * head_dim)

        # Project the attended values to the output dimension
        output = self.output_proj(attended_values)  # (batch_size, query_len, embed_dim)

        return output, self.cache


class ExpertNetwork(nn.Module):
    """
    A single expert network within the Mixture-of-Experts (MoE) layer.
    This implementation uses MLPs for each expert.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GatingNetwork(nn.Module):
    """
    A gating network that predicts the expert selection probabilities for each input token.
    """

    def __init__(self, input_dim: int, expert_count: int):
        super().__init__()

        self.gating_mlp = nn.Sequential(
            nn.Linear(input_dim, expert_count),
            nn.Softmax(dim=-1)  # Probabilities across experts
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gating_mlp(x)


class GatedMoE(nn.Module):
    """
    A Mixture-of-Experts (MoE) layer with gating, top-k expert selection,
    and residual connections for robust and efficient processing.
    # Example Usage:
          input_tensor = torch.randn(32, 128, 512)  # Example input

          expert_learner = ExpertLevelLearner(input_dim=512, hidden_dim=1024, output_dim=512)
          output_tensor = expert_learner(input_tensor)

          print(output_tensor.shape)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 expert_count: int = EXPERT_COUNT, top_k: int = TOP_K):
        super().__init__()

        self.expert_count = expert_count
        self.top_k = top_k

        # Create a list to hold expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim) for _ in range(expert_count)
        ])

        # Gating Network: Predicts which experts are suitable for each input
        self.gating_network = GatingNetwork(input_dim, expert_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.size()

        # Gating: Get expert selection probabilities for each token
        gating_scores = self.gating_network(x)

        # Top-k Expert Selection for efficiency and specialization
        top_expert_indices = torch.topk(gating_scores, self.top_k, dim=-1).indices

        # Efficiently process inputs through selected experts
        expert_outputs = torch.zeros(batch_size, sequence_length, self.expert_count, x.size(-1),
                                     device=x.device)

        for i in range(self.expert_count):
            # Create a mask to select tokens assigned to this expert
            expert_mask = (top_expert_indices == i).any(dim=-1, keepdim=True)

            # Select the input tokens that belong to this expert
            selected_inputs = x * expert_mask

            # Pass selected inputs through the corresponding expert
            expert_outputs[:, :, i, :] = self.experts[i](selected_inputs)

        # Combine expert outputs - weighted average based on gating scores
        weighted_outputs = expert_outputs * gating_scores.unsqueeze(-1)
        combined_output = weighted_outputs.sum(dim=2)

        # Apply a residual connection to help with training stability
        output = x + combined_output
        return output


class ExpertLevelLearner(nn.Module):
    """
    A module that learns expert-level representations from input tensors.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 expert_count: int = EXPERT_COUNT, top_k: int = TOP_K):
        super().__init__()

        self.gated_moe = GatedMoE(input_dim, hidden_dim, output_dim, expert_count, top_k)

    def forward(self, x: Tensor) -> Tensor:
        return self.gated_moe(x)


class TransformerBlock(nn.Module):
    """
    A transformer block consisting of advanced self-attention, expert routing, and residual connections.
    EXample :
    batch_size = 16
      seq_len = 512
      embed_dim = 768
      num_heads = 12
      head_dim = 64
      input_tensor = torch.randn(batch_size, seq_len, embed_dim)
      m=TransformerBlock(
          embed_dim=768,
          num_heads=12,
          head_dim=64,
          expert_hidden_dim=512,
          dropout = 0.1
      )
      output=m(input_tensor)
      print(output.shape)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        expert_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = AdvancedSelfAttention(embed_dim, num_heads, head_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.expert_learner = ExpertLevelLearner(embed_dim, expert_hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[LongTensor] = None) -> Tensor: # Using LongTensor for the mask
        """
        Forward pass through the transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
            mask (LongTensor, optional): Attention mask (should be LongTensor). Defaults to None.

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # Self-attention sub-layer
        attn_output, _ = self.attention(x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Expert routing sub-layer
        expert_output = self.expert_learner(x)
        x = x + self.dropout(expert_output)
        x = self.norm2(x)
        return x



class HemanthTransformerModel(nn.Module):
    """
    HemanthTransformerModel is a transformer-based model designed for
    natural language understanding and generation tasks. It leverages
    advanced techniques like rotational positional embeddings,
    a Mixture-of-Experts (MoE) layer for enhanced expressiveness, and
    a novel expert-level learning mechanism to promote deeper context
    understanding.

    Attributes:
        embed_dim (int): Dimensionality of token embeddings.
        num_heads (int): Number of attention heads in multi-head attention.
        head_dim (int): Dimensionality of each attention head.
        expert_hidden_dim (int): Hidden dimension for expert networks in MoE.
        num_layers (int): Number of transformer blocks.
        vocab_size (int): Size of the vocabulary.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.

    Example:
        >>> model = HemanthTransformerModel(
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     head_dim=64,
        ...     expert_hidden_dim=1024,
        ...     num_layers=6,
        ...     vocab_size=30522
        ... )
        >>> input_ids = torch.randint(0, 30522, (16, 512), dtype=torch.long) # Input should be LongTensor
        >>> output = model(input_ids)
        >>> print(output.shape)
        torch.Size([16, 512, 30522])

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        expert_hidden_dim: int,
        num_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: LongTensor) -> Tensor: # Input should be LongTensor
        """
        Forward pass through the HemanthTransformerModel.

        Args:
            input_ids (LongTensor): Input token IDs of shape
                                      (batch_size, sequence_length). Should be LongTensor.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(input_ids)
        x = add_rotational_positional_embeddings(x)
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.output_layer(x)
        return logits




model = HemanthTransformerModel(
    embed_dim=512,
    num_heads=8,
    head_dim=64,
    expert_hidden_dim=1024,
    num_layers=10,
    vocab_size=30522 
)
input_ids = torch.randint(0, 30522, (16, 512)) 
output = model(input_ids)
print(model)
print(output.shape)
print(model)
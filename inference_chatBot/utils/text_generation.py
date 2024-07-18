import torch
from typing import Optional


def modify_logits_for_top_k_filtering(logits: torch.Tensor, top_k: int) -> None:
    """Set the logits for non-top-k values to -inf."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float('-inf'))


def modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> None:
    """Set the logits for non-top-p values to -inf."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the range (0, 1].")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    filter_ = cumulative_probs > top_p
    filter_[:, 1:] = filter_[:, :-1].clone()
    filter_[..., 0] = 0
    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float('-inf'))


def sample(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, temperature: float = 1.0,
           vocab_size: Optional[int] = None) -> torch.Tensor:
    """Sample and generate a token.

    Args:
        logits (torch.Tensor): The logits tensor of shape [b, v], where b is the batch size
                               and v is the vocabulary size.
        top_k (int): The number of top-k tokens to consider for sampling. Default is 0.
        top_p (float): The cumulative probability threshold for top-p sampling. Default is 0.0.
        temperature (float): The temperature value for scaling the logits. Default is 1.0.
        vocab_size (Optional[int]): The vocabulary size. If provided, the generated samples
                                    will be clamped to the range [0, vocab_size). Default is None.

    Returns:
        torch.Tensor: The generated token samples of shape [b].

    Raises:
        AssertionError: If the logits tensor does not have the expected shape or data type.
        ValueError: If top_k is not a positive integer or top_p is not in the range (0, 1].
    """
    if logits.ndim != 2:
        raise AssertionError("Expected the logits to be of shape [b, v].")
    if logits.dtype != torch.float32:
        raise AssertionError("Input logits should be of type torch.float32.")

    if top_k == 1:
        if top_p != 0.0:
            raise ValueError("Cannot set both greedy and top-p samplings.")
        samples = torch.argmax(logits, dim=-1)
    else:
        logits = logits.clone()
        if temperature != 1.0:
            logits /= temperature

        if top_k > 1:
            if top_p != 0.0:
                raise ValueError("Cannot set both top-k and top-p samplings.")
            if top_k > logits.size(1):
                raise ValueError("top_k is larger than logit size.")
            if vocab_size and top_k >= vocab_size:
                raise ValueError("top_k is larger than or equal to vocab size.")
            modify_logits_for_top_k_filtering(logits, top_k)
        elif top_p > 0.0:
            modify_logits_for_top_p_filtering(logits, top_p)

        probs = logits.softmax(dim=-1)
        samples = torch.multinomial(probs, num_samples=1).view(-1)

    if vocab_size:
        samples = torch.clamp(samples, min=0, max=(vocab_size - 1))

    return samples
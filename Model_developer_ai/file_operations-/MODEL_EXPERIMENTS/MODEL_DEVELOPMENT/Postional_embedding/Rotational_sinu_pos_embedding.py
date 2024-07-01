import torch
from typing import Tuple
import matplotlib.pyplot as plt

def rotational_sinusoidal_positional_embeddings(
    tensor: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: float = 10000,
    normalize: bool = False,
    scale: float = None,
) -> torch.Tensor:
    """
    Generate rotational sinusoidal positional embeddings for an input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of any dimensions.
        num_pos_feats (int): Number of positional features (channels) to generate.
        temperature (float): Temperature for scaling the embeddings.
        normalize (bool): Whether to normalize the embeddings.
        scale (float): Scale factor for the embeddings.

    Returns:
        torch.Tensor: Positional embeddings with the same shape as the input tensor.
    """
    if scale is None:
        scale = 2 * torch.pi
    
    shape = tensor.shape
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_embeddings = []
    for dim in range(len(shape)):
        pos = torch.arange(shape[dim], dtype=torch.float32, device=tensor.device)
        pos_embed = pos[:, None] / dim_t[None, :]
        pos_embed[:, 0::2] = torch.sin(pos_embed[:, 0::2])
        pos_embed[:, 1::2] = torch.cos(pos_embed[:, 1::2])
        pos_embeddings.append(pos_embed)

    # Reshape each positional embedding to match the input tensor shape
    reshaped_embeddings = []
    for i, embedding in enumerate(pos_embeddings):
        dims = [1] * len(shape)
        dims[i] = shape[i]
        reshaped_embeddings.append(embedding.view(*dims, -1).expand(*shape, -1))

    pos_embeddings = torch.cat(reshaped_embeddings, dim=-1)

    if normalize:
        # Apply square root mean normalization
        pos_embeddings = pos_embeddings / torch.sqrt(torch.mean(pos_embeddings ** 2, dim=-1, keepdim=True))
        pos_embeddings = pos_embeddings * scale

    return pos_embeddings

# Example usage
tensor = torch.randn(3, 224, 224)
pos_embeddings = rotational_sinusoidal_positional_embeddings(tensor, normalize=True)
print(pos_embeddings.shape)  # Output: torch.Size([3, 224, 224, 192])

# # Visualize the positional embeddings
# num_plots = min(pos_embeddings.shape[-1], 8)  # Limit the number of plots
# fig, axes = plt.subplots(2, num_plots // 2, figsize=(12, 6))
# axes = axes.flatten()

# for i in range(num_plots):
#     ax = axes[i]
#     embedding = pos_embeddings[0, :, :, i].detach().cpu().numpy()
#     ax.imshow(embedding, cmap='viridis')
#     ax.set_title(f"Embedding {i+1}")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
# def visualize_embeddings(embeddings, title):
#     plt.figure(figsize=(12, 6))
#     for i in range(embeddings.shape[-1] // 2):  # Only plotting a subset for clarity
#         plt.subplot(3, embeddings.shape[-1] // 6, i+1)
#         plt.imshow(embeddings[:, :, i].cpu(), cmap='viridis')
#         plt.axis('off')
#     plt.suptitle(title)
#     plt.savefig("./embedding_rotational_sinu_")

# visualize_embeddings(pos_embeddings[0], 'Positional Embeddings Visualization')
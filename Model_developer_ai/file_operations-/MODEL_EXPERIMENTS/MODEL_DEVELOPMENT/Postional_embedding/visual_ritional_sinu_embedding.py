import torch
import matplotlib.pyplot as plt

def rotational_sinusoidal_positional_embeddings(
    tensor: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: float = 10000,
    normalize: bool = False,
    scale: float = None,
) -> torch.Tensor:
    if scale is None and normalize:
        scale = 2 * torch.pi

    shape = tensor.shape
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=tensor.device)
    dim_t = temperature ** (2 * (dim_t) / num_pos_feats)

    pos_x = torch.arange(shape[-2], dtype=torch.float32, device=tensor.device).unsqueeze(1)
    pos_y = torch.arange(shape[-1], dtype=torch.float32, device=tensor.device).unsqueeze(0)
    
    sin_embed_x = torch.sin(pos_x / dim_t).flatten(1)
    cos_embed_x = torch.cos(pos_x / dim_t).flatten(1)
    sin_embed_y = torch.sin(pos_y / dim_t).flatten(0)
    cos_embed_y = torch.cos(pos_y / dim_t).flatten(0)

    pos_embed_x = torch.cat((sin_embed_x, cos_embed_x), dim=1).unsqueeze(1).unsqueeze(0)
    pos_embed_y = torch.cat((sin_embed_y, cos_embed_y), dim=1).unsqueeze(0).unsqueeze(0)

    pos_embed = pos_embed_x.repeat(shape[0], shape[1], 1, 1) + pos_embed_y.repeat(shape[0], shape[1], 1, 1)

    if normalize:
        pos_embed = pos_embed / (pos_embed.norm(dim=-1, keepdim=True) + 1e-6) * scale

    return pos_embed

# Example usage
tensor = torch.randn(3, 224, 224)  # Simulating a batch of 3 images with height and width 224
pos_embeddings = rotational_sinusoidal_positional_embeddings(tensor, normalize=True)
print(pos_embeddings.shape)  # Output should be: torch.Size([3, 224, 224, num_pos_feats])

# Visualization
def visualize_embeddings(embeddings, title):
    plt.figure(figsize=(12, 6))
    for i in range(embeddings.shape[-1] // 2):  # Only plotting a subset for clarity
        plt.subplot(3, embeddings.shape[-1] // 6, i+1)
        plt.imshow(embeddings[0, :, :, i].cpu(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

visualize_embeddings(pos_embeddings[0], 'Positional Embeddings Visualization')
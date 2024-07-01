import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch import optim
import numpy as np
from torch.hub import tqdm
from typing import Tuple


class PatchExtractor(nn.Module):
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input_data.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Input height ({height}) and width ({width}) must be divisible by patch size ({self.patch_size})"

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
            unfold(3, self.patch_size, self.patch_size). \
            permute(0, 2, 3, 1, 4, 5). \
            contiguous(). \
            view(batch_size, num_patches, -1)

        # Expected shape of a patch on default settings is (4, 196, 768)

        return patches


class InputEmbedding(nn.Module):
    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection for image patches
        self.image_linear_projection = nn.Linear(self.input_size, self.latent_size)
        # Linear projection for audio features
        self.audio_linear_projection = nn.Linear(args.audio_feature_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size).to(self.device))
        # Positional embedding for image patches
        self.image_pos_embedding = nn.Parameter(torch.randn(1, 1, self.latent_size).to(self.device))
        # Positional embedding for audio features
        self.audio_pos_embedding = nn.Parameter(torch.randn(1, 1, self.latent_size).to(self.device))

    def forward(self, image_data: torch.Tensor, audio_data: torch.Tensor) -> torch.Tensor:
        image_data = image_data.to(self.device)
        audio_data = audio_data.to(self.device)

        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        image_patches = patchify(image_data)

        # Linear projection for image patches
        image_linear_projection = self.image_linear_projection(image_patches).to(self.device)
        b, n, _ = image_linear_projection.shape
        image_linear_projection = torch.cat((self.class_token.expand(b, -1, -1), image_linear_projection), dim=1)
        image_pos_embed = self.image_pos_embedding.expand(b, n + 1, -1)
        image_linear_projection += image_pos_embed

        # Linear projection for audio features
        audio_linear_projection = self.audio_linear_projection(audio_data).to(self.device)
        audio_linear_projection = audio_linear_projection.unsqueeze(1)  # Add an extra dimension for concatenation
        audio_linear_projection = torch.cat((self.class_token.view(1, 1, -1).expand(b, -1, -1), audio_linear_projection), dim=1)
        audio_pos_embed = self.audio_pos_embedding.expand(b, audio_linear_projection.shape[1], -1)
        # audio_pos_embed = self.audio_pos_embedding.expand(b, audio_data.shape[1] + 1, -1)
        audio_linear_projection += audio_pos_embed

        # Concatenate image and audio embeddings
        combined_embeddings = torch.cat((image_linear_projection, audio_linear_projection), dim=1)

        return combined_embeddings


class EncoderBlock(nn.Module):
    def __init__(self, args):
        super(EncoderBlock, self).__init__()

        self.latent_size = args.latent_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.norm = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, emb_patches: torch.Tensor) -> torch.Tensor:
        first_norm = self.norm(emb_patches)
        attention_out = self.attention(first_norm, first_norm, first_norm)[0]
        first_added = attention_out + emb_patches
        second_norm = self.norm(first_added)
        mlp_out = self.enc_mlp(second_norm)
        output = mlp_out + first_added

        return output


class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()

        self.num_encoders = args.num_encoders
        self.latent_size = args.latent_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.embedding = InputEmbedding(args)
        # Encoder Stack
        self.encoders = nn.ModuleList([EncoderBlock(args) for _ in range(self.num_encoders)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes),
        )

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        enc_output = self.embedding(image_input, audio_input)
        for enc_layer in self.encoders:
            enc_output = enc_layer(enc_output)

        class_token_embed = enc_output[:, 0]
        return self.mlp_head(class_token_embed)


import torch

# Define the input sizes
batch_size = 1
image_channels = 3
image_height = 224
image_width = 224
audio_feature_size = 1000000

# Create dummy image input
dummy_image_input = torch.randn(batch_size, image_channels, image_height, image_width)

# Create dummy audio input
dummy_audio_input = torch.randn(batch_size, audio_feature_size)

# Create an instance of the ViT model
class Args:
    def __init__(self):
        self.patch_size = 16
        self.n_channels = image_channels
        self.latent_size = 768
        self.num_heads = 12
        self.dropout = 0.1
        self.num_encoders = 12
        self.num_classes = 10
        self.batch_size = batch_size
        self.audio_feature_size = audio_feature_size
        self.no_cuda = False

args = Args()
model = ViT(args)

# Pass the dummy inputs through the model
output = model(dummy_image_input, dummy_audio_input)

# Print the output shape
print("Output shape:", output.shape)

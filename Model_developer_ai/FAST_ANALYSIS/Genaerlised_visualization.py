import os
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
from torchvision.transforms import functional as TF
from plotly.subplots import make_subplots



def plot_tensor_grid_sequence(tensor: torch.Tensor, grid_size: Tuple[int, int], output_file: str) -> None:
    """
    Plot a tensor in a grid format using Plotly and save the plot as an HTML file.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, time_stamp, sequence_length).
        grid_size (Tuple[int, int]): Size of the grid in the format [N, N].
        output_file (str): Path to save the output HTML file.

    Returns:
        None
      import torch 

      tensor=torch.rand(4, 50 ,512)
      grid_size=(2,2)
      plot_tensor_grid_sequence(tensor, grid_size, "tensor_grid.html")
    """
    batch_size, time_stamp, sequence_length = tensor.shape
    num_plots = batch_size
    fig = make_subplots(
        rows=grid_size[0],
        cols=grid_size[1],
        subplot_titles=[f"Sequence {i+1}" for i in range(num_plots)]
    )
    for i in range(num_plots):
        sequence = tensor[i]
        trace = go.Heatmap(
            z=sequence.numpy(),
            colorscale="Viridis",
            showscale=False
        )

        fig.add_trace(trace, row=(i // grid_size[1]) + 1, col=(i % grid_size[1]) + 1)
    fig.update_layout(
        title=f"Tensor Grid Plot",
        height=600,
        width=800,
        showlegend=False
    )
    fig.write_html(output_file)

def plot_tensor_grid_image(
    tensor: torch.Tensor, 
    grid_size: Tuple[int, int], 
    filename: str = "tensor_grid.html"
) -> None:
    """
    Plots a 4D tensor (batch_size, channels, height, width) as a grid of images.

    Args:
        tensor (torch.Tensor): The input tensor. Should be 4D with shape 
                               (batch_size, channels, height, width).
        grid_size (Tuple[int, int]): The desired grid size (rows, columns).
        filename (str, optional): The filename to save the plot. 
                                  Defaults to "tensor_grid.html".

    Raises:
        ValueError: If the tensor is not 4D or the grid size is invalid.
    """
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must be 4D (batch_size, channels, height, width)")
    batch_size, channels, height, width = tensor.shape
    rows, cols = grid_size
    if rows * cols != batch_size:
        raise ValueError(
            f"Grid size ({rows}, {cols}) cannot accommodate {batch_size} images."
        )
    fig = px.imshow(
        tensor.permute(0, 2, 3, 1).numpy(),  # (B, H, W, C) for Plotly
        facet_col=0,  
        facet_col_wrap=cols,  
        facet_row_spacing=0.01,  
        facet_col_spacing=0.01,  
        # color_continuous_scale="gray" if channels == 1 else None,
    )
    fig.update_layout(
        title="Tensor Grid Visualization",
        height=rows * 250,  # Adjust height based on the number of rows
        width=cols * 250,   # Adjust width based on the number of columns
    )
    fig.update_xaxes(showticklabels=False)  
    fig.update_yaxes(showticklabels=False)  
    fig.write_html(filename)
    print(f"Plot saved to: {filename}")

def convert_tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a single tensor to a PIL Image."""
    return TF.to_pil_image(tensor)

def create_image_grid(images: List[Image.Image], grid_size: Tuple[int, int]) -> go.Figure:
    """Create a grid of images using Plotly."""
    rows, cols = grid_size
    fig = make_subplots(rows=rows, cols=cols)

    for index, image in enumerate(images):
        row = (index // cols) + 1
        col = (index % cols) + 1
        img = TF.resize(image, (1024, 1024))
        fig.add_trace(
            go.Image(z=img),
            row=row, col=col
        )
    
    fig.update_layout(height=8000, width=8000, showlegend=False)
    return fig

def save_individual_images(images: List[Image.Image], folder_path: str) -> None:
    """Save individual images to specified folder."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for index, image in enumerate(images):
        img = TF.resize(image, (1024, 1024))
        img.save(os.path.join(folder_path, f'image_{index + 1}.png'))

def Plot_tensor_image(input_tensor: torch.Tensor,Grid:Tuple,dir_path:str) -> None:
    """
    import torch 

    input_tensor = torch.randn(4, 3, 1024,1024)
    grid=(2,2)
    Plot_tensor_image(input_tensor,grid,dir_path="./plot")
    """
    images = [convert_tensor_to_image(input_tensor[i]) for i in range(input_tensor.shape[0])]
    grid = create_image_grid(images, Grid)
    grid.write_html(f"{dir_path}/image_grid.html")
    save_individual_images(images, folder_path=dir_path)

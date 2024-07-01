from typing import Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchinfo import summary

def get_model_details(model: Any) -> dict:
    """
    Returns a summary of the provided model.

    Parameters:
    model (Any): A machine learning model.

    Returns:
    dict: A dictionary containing model details.
    """
    model_sum = summary(model, verbose=0)
    print(model_sum)
    return {
        "total_parameters": model_sum.total_params,
        "trainable_parameters": model_sum.trainable_params,
        "model_size": model_sum.total_params * 4 / (1024**2)  # Assuming 4 bytes per parameter (float32)
    }
def calculate_operations(model: Any, num_tokens: int) -> int:
    """
    Calculate the total number of operations for a given number of tokens.

    Parameters:
    model (Any): A machine learning model.
    num_tokens (int): The number of tokens to process.

    Returns:
    int: The estimated number of operations.
    """
    total_operations = 0

    # Iterate over the model's modules
    for module in model.modules():
        # Check if the module is a linear layer
        if isinstance(module, torch.nn.Linear):
            # For a linear layer, the number of operations is:
            # (input_features + 1) * output_features
            # where +1 accounts for the bias term
            total_operations += (module.in_features + 1) * module.out_features

        # Check if the module is a convolutional layer
        elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            # For a convolutional layer, the number of operations is:
            # (kernel_size * input_channels + 1) * output_channels * output_spatial_size
            # where kernel_size is the product of the kernel dimensions,
            # and output_spatial_size is the product of the output spatial dimensions
            kernel_size = torch.prod(torch.tensor(module.kernel_size)).item()
            output_spatial_size = torch.prod(torch.tensor(module.output_size()[2:])).item()
            total_operations += (kernel_size * module.in_channels + 1) * module.out_channels * output_spatial_size

        # Check if the module is a batch normalization layer
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # For a batch normalization layer, the number of operations is:
            # 2 * num_features
            # where num_features is the number of features (channels)
            total_operations += 2 * module.num_features

        # Check if the module is an activation function
        elif isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            # For activation functions, the number of operations is equal to the number of elements
            # in the input tensor
            input_size = torch.prod(torch.tensor(module.input_size)).item()
            total_operations += input_size

        # Check if the module is a pooling layer
        elif isinstance(module, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
                                 torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d)):
            # For pooling layers, the number of operations is equal to the number of elements
            # in the output tensor
            output_size = torch.prod(torch.tensor(module.output_size)).item()
            total_operations += output_size

        # Add more conditions for other types of layers as needed

    # Multiply the total number of operations by the number of tokens
    total_operations *= num_tokens

    return total_operations

def estimate_memory_usage(model: Any, num_tokens: int, duration_hours: float) -> float:
    """
    Estimate the memory required for running the model for a given duration.

    Parameters:
    model (Any): A machine learning model.
    num_tokens (int): The number of tokens to process.
    duration_hours (float): The duration of operation in hours.

    Returns:
    float: The estimated memory in GB required for the operation.
    """
    # This is a placeholder for the actual memory calculation.
    # Actual memory usage could depend on the model architecture, batch size, etc.
    memory_per_token_per_hour = 0.001  # An example value; replace with actual calculation.
    return num_tokens * memory_per_token_per_hour * duration_hours

def recommend_hardware(memory_requirement: float) -> str:
    """
    Recommend the best choice of hardware based on the memory requirement.

    Parameters:
    memory_requirement (float): The memory requirement in GB.

    Returns:
    str: A string containing the hardware recommendation.
    """
    # Hardware recommendation is a complex topic and would require more context.
    # Below is a simplistic approach based on memory requirements.
    if memory_requirement <= 16:
        return "Standard GPU (e.g., NVIDIA GTX 1080)"
    elif memory_requirement <= 32:
        return "High-end GPU (e.g., NVIDIA RTX 2080)"
    else:
        return "Datacenter GPU (e.g., NVIDIA Tesla V100 or higher)"

# Example usage
if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_details = get_model_details(model)
    print("Model details:", model_details)
    
    num_tokens = 2048  # Example number of tokens
    operations = calculate_operations(model, num_tokens)
    print(f"Total number of calculations for {num_tokens} tokens: {operations}")
    
    duration_hours = 1  # Example duration
    memory_requirement = estimate_memory_usage(model, num_tokens, duration_hours)
    print(f"Estimated memory required for {duration_hours} hour: {memory_requirement} GB")
    
    hardware_recommendation = recommend_hardware(memory_requirement)
    print("Recommended hardware:", hardware_recommendation)
import json
import json
import torch
import logging
import os
import plotly.graph_objects as go
from typing import Dict, Any, Union
from collections import OrderedDict
from transformers import PreTrainedModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ================================================================================
# Extract model details
# ================================================================================

def extract_model_details(model: Union[torch.nn.Module, PreTrainedModel]) -> Dict[str, Any]:
    """
    Extract key-value pairs from the model's state dictionary.

    Args:
        model (Union[torch.nn.Module, PreTrainedModel]): The model to extract details from.

    Returns:
        Dict[str, Any]: The extracted key-value pairs.
    """
    if isinstance(model, PreTrainedModel):
        state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    output_data = extract_state_dict_details(state_dict)
    return output_data

# ================================================================================
# Extract state dictionary details
# ================================================================================

def extract_state_dict_details(state_dict: OrderedDict) -> Dict[str, Any]:
    """
    Recursively extract key-value pairs from the model's state dictionary.

    Args:
        state_dict (OrderedDict): The state dictionary to extract key-value pairs from.

    Returns:
        Dict[str, Any]: The extracted key-value pairs.
    """
    result = {}

    for key, value in state_dict.items():
        if isinstance(value, OrderedDict):
            result[key] = extract_state_dict_details(value)
        elif isinstance(value, torch.Tensor):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "requires_grad": value.requires_grad
            }
        else:
            result[key] = value

    return result

# ================================================================================
# Write to JSON file
# ================================================================================

def write_to_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Write the extracted data to a JSON file.

    Args:
        data (Dict[str, Any]): The data to be written to the JSON file.
        file_path (str): The path to the output JSON file.
    """
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)




# ================================================================================
# Summarize Model
# ================================================================================

def summarize_model(model: torch.nn.Module) -> Dict[str, Any]:
    """Summarize a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
        Dict[str, Any]: A dictionary containing the model summary.
    """
    summary = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "layers": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module):
            layer_summary = {
                "name": name,
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "non_trainable_parameters": sum(p.numel() for p in module.parameters() if not p.requires_grad),
                "input_shape": None,
                "output_shape": None,
                "device": None,
            }

            # Get device if the module has parameters
            try:
                layer_summary["device"] = next(module.parameters()).device
            except StopIteration:
                pass

            # Get input and output shapes if possible
            try:
                example_input = torch.randn(1, *getattr(module, "input_shape", (1,)))
                output = module(example_input)
                layer_summary["input_shape"] = tuple(example_input.shape)
                layer_summary["output_shape"] = tuple(output.shape)
            except Exception:
                pass

            summary["layers"].append(layer_summary)

    return summary

# ================================================================================
# Print Model Summary
# ================================================================================

def print_model_summary(model_summary: Dict[str, Any]) -> None:
    """Print the model summary in a professional and boxed format using the logging module.

    Args:
        model_summary (Dict[str, Any]): The model summary dictionary.
    """
    max_name_length = max(len(layer["name"]) for layer in model_summary["layers"])
    max_type_length = max(len(layer["type"]) for layer in model_summary["layers"])
    max_params_length = max(len(f"{layer['parameters']:,}") for layer in model_summary["layers"])
    max_trainable_length = max(len(str(layer["trainable_parameters"] > 0)) for layer in model_summary["layers"])
    max_device_length = max(len(str(layer["device"])) for layer in model_summary["layers"])

    header_format = (
        "| {name:<{name_width}} | {type:<{type_width}} | {params:<{params_width}} | "
        "{trainable:<{trainable_width}} | {device:<{device_width}} |"
    )
    row_format = (
        "| {name:<{name_width}} | {type:<{type_width}} | {params:>{params_width},} | "
        "{trainable:>{trainable_width}} | {device:>{device_width}} |"
    )
    divider = "+" + "-" * (max_name_length + 2) + "+" + "-" * (max_type_length + 2) + "+" + \
              "-" * (max_params_length + 2) + "+" + "-" * (max_trainable_length + 2) + "+" + \
              "-" * (max_device_length + 2) + "+"

    logging.info("+" + "-" * (max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 10) + "+")
    logging.info("|" + " Model Summary ".center(max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 10) + "|")
    logging.info("+" + "-" * (max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 10) + "+")
    logging.info("| Total Parameters: {:<{width},} |".format(model_summary["total_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 8))
    logging.info("| Trainable Parameters: {:<{width},} |".format(model_summary["trainable_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 8))
    logging.info("| Non-Trainable Parameters: {:<{width},} |".format(model_summary["non_trainable_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 8))
    logging.info(divider)
    logging.info(header_format.format(
        name="Layer Name", type="Type", params="Parameters", trainable="Trainable", device="Device",
        name_width=max_name_length, type_width=max_type_length, params_width=max_params_length,
        trainable_width=max_trainable_length, device_width=max_device_length
    ))
    logging.info(divider)

    for layer in model_summary["layers"]:
        logging.info(row_format.format(
            name=layer["name"], type=layer["type"], params=layer["parameters"],
            trainable=str(layer["trainable_parameters"] > 0), device=str(layer["device"]),
            name_width=max_name_length, type_width=max_type_length, params_width=max_params_length,
            trainable_width=max_trainable_length, device_width=max_device_length
        ))
        if layer["input_shape"] is not None:
            logging.info("|   Input Shape: {:<{width}} |".format(str(layer["input_shape"]), width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 8))
        if layer["output_shape"] is not None:
            logging.info("|   Output Shape: {:<{width}} |".format(str(layer["output_shape"]), width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 8))
        logging.info(divider)

    logging.info("+" + "-" * (max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 10) + "+")


# ================================================================================
# Plot model weights
# ================================================================================

def plot_model_weights(model_details: Dict[str, Any], output_dir: str) -> None:
    """
    Plot the model weights using Plotly.

    Args:
        model_details (Dict[str, Any]): The extracted model details.
        output_dir (str): The directory to save the plotted images.
    """
    for key, value in model_details.items():
        if isinstance(value, dict) and "shape" in value:
            shape = value["shape"]
            if len(shape) == 4:  # Video (4D)
                fig = go.Figure(data=go.Surface(
                    z=torch.randn(shape[0], shape[1], shape[2], shape[3]).numpy()
                ))
                fig.update_layout(title=f"4D Weights - {key}", autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))
                fig.write_image(os.path.join(output_dir, f"{key}_4d.png"))
            elif len(shape) == 3:  # Color Image (3D)
                fig = go.Figure(data=go.Surface(
                    z=torch.randn(shape[0], shape[1], shape[2]).numpy()
                ))
                fig.update_layout(title=f"3D Weights - {key}", autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))
                fig.write_image(os.path.join(output_dir, f"{key}_3d.png"))
            elif len(shape) == 2:  # Grayscale Image (2D)
                fig = go.Figure(data=go.Heatmap(
                    z=torch.randn(shape[0], shape[1]).numpy()
                ))
                fig.update_layout(title=f"2D Weights - {key}", autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))
                fig.write_image(os.path.join(output_dir, f"{key}_2d.png"))
            elif len(shape) == 1:  # 1D
                fig = go.Figure(data=go.Scatter(
                    y=torch.randn(shape[0]).numpy()
                ))
                fig.update_layout(title=f"1D Weights - {key}", autosize=False,
                                  width=500, height=500,
                                  margin=dict(l=65, r=50, b=65, t=90))
                fig.write_image(os.path.join(output_dir, f"{key}_1d.png"))
        elif isinstance(value, dict):
            plot_model_weights(value, output_dir)
# ================================================================================
# Example Usage
# ================================================================================


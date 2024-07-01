import json
import torch
import logging
import os
import re
import time
import cv2
import torch
import torch.nn as nn
import gc
import threading
import psutil
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from transformers import PreTrainedModel
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Union, Tuple, List,Any
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format="%(message)s")


def list_png_files(directory: str) -> List[str]:
    """
    Recursively list all .png files in the given directory.
    
    Args:
        directory (str): The path to the directory.
    
    Returns:
        List[str]: A list of file paths for all .png images.
    """
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    png_files.sort()  # Ensure the files are sorted for sequential video frames
    return png_files

def create_video_from_images(image_files: List[str], output_video_path: str, frame_rate: int):
    """
    Create a video from a list of image files.
    
    Args:
        image_files (List[str]): List of image file paths.
        output_video_path (str): Path to the output video file.
        frame_rate (int): Frame rate for the video.
    
    Raises:
        FileNotFoundError: If any of the image files do not exist.
        ValueError: If image list is empty.
    """
    if not image_files:
        raise ValueError("No image files found to create a video.")

    # Check if all files exist
    for file in image_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Image file not found: {file}")

    # Read the first image to get the size
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError(f"Could not read the first image: {image_files[0]}")
    
    height, width, layers = first_image.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, size)

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            raise ValueError(f"Could not read image: {image_file}")
        
        # Ensure the image size is consistent
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size)
        
        video_writer.write(image)

    video_writer.release()



EXAMPLE_SNIPPET_CODE=   """
    if __name__ == "__main__":
    from transformers import (AutoModelForCausalLM,AutoImageProcessor)
    from transformers import AutoImageProcessor, ResNetForImageClassification

    output_json_path = "model_details.json"
    output_plot_dir = "model_weights_plots"
    model_name = "bert-base-uncased"
    output_weights_dir = "model_weights"
    try:
        logging.info(f"Loading model {model_name}...")
        # model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir="./model")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50",cache_dir="./model")
        
        logging.info("Extracting model details...")
        model_details = extract_model_details(model)
        
        logging.info(f"Writing model details to {output_json_path}...")
        write_to_json_file(model_details, output_json_path)
        
        logging.info("Summarizing model...")
        summary = summarize_model(model)
        
        logging.info("Printing model summary...")
        print_model_summary(summary)
        
        logging.info(f"Plotting model weights to {output_plot_dir}...")
        plot_model_weights(model_details, output_plot_dir)
        
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    try:
        logging.info(f"Loading model {model_name}...")
        # model = BertModel.from_pretrained(model_name)
        
        logging.info(f"Creating directory structure at {output_weights_dir}...")
        create_directory_structure(output_weights_dir)
        
        logging.info("Saving model weights...")
        save_model_weights(model, output_weights_dir)
        
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    try:
        logging.info(f"Loading model architecture {model_name}...")
        # model = BertModel.from_pretrained(model_name, state_dict=None)
        
        logging.info(f"Loading model weights from {output_weights_dir}...")
        load_model_weights(model, output_weights_dir)
        
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    
    """


def extract_keys_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively extract all keys and values from a nested dictionary.

    Args:
        data (Dict[str, Any]): The input dictionary.

    Returns:
        Dict[str, Any]: A flattened dictionary with all keys and values.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested_result = extract_keys_values(value)
            for nested_key, nested_value in nested_result.items():
                result[f"{key}.{nested_key}"] = nested_value
        else:
            result[key] = value
    return result



def load_tensor(file_path: str) -> torch.Tensor:
    """
    Load a tensor from a file.

    Args:
        file_path (str): The file path to load the tensor from.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    try:
        tensor = torch.load(file_path)
        logging.info(f"Tensor loaded from {file_path}")
        return tensor
    except Exception as e:
        logging.error(f"Failed to load tensor: {e}")
        raise




def load_model_weights(model: nn.Module, base_path: str) -> None:
    """
    Load the weights of a model from files.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        base_path (str): The base directory path to load the weights from.

    Raises:
        FileNotFoundError: If any weight file is not found.
        RuntimeError: If there is an error loading the state dictionary.
    """
    start_time = time.time()
    try:
        state_dict = model.state_dict()
        loaded_state_dict = _load_state_dict_weights(state_dict, base_path)
        model.load_state_dict(loaded_state_dict)
        logging.info("Model weights loaded successfully in %.2f seconds.", time.time() - start_time)
    except FileNotFoundError as e:
        logging.error(f"Failed to load model weights: {e}")
        raise
    except RuntimeError as e:
        logging.error(f"Error loading state dictionary: {e}")
        raise



def _load_state_dict_weights(state_dict: OrderedDict, base_path: str) -> OrderedDict:
    """
    Recursively load the weights from the state dictionary.

    Args:
        state_dict (OrderedDict): The state dictionary to load the weights into.
        base_path (str): The base directory path to load the weights from.

    Returns:
        OrderedDict: The loaded state dictionary.
    """
    loaded_state_dict = OrderedDict()

    for key, value in state_dict.items():
        file_name = key.replace(".", "_") + ".pt"
        file_path = os.path.join(base_path, file_name)

        if os.path.isfile(file_path):
            loaded_state_dict[key] = load_tensor(file_path)
        elif isinstance(value, OrderedDict):
            sub_base_path = os.path.join(base_path, key)
            loaded_state_dict[key] = _load_state_dict_weights(value, sub_base_path)
        else:
            loaded_state_dict[key] = value

    return loaded_state_dict





def save_tensor(tensor: torch.Tensor, file_path: str) -> None:
    """
    Save a tensor to a file.

    Args:
        tensor (torch.Tensor): The tensor to save.
        file_path (str): The file path to save the tensor.
    """
    try:
        torch.save(tensor, file_path)
        logging.info(f"Tensor saved at {file_path}")
    except Exception as e:
        logging.error(f"Failed to save tensor: {e}")
        raise



def save_model_weights(model: torch.nn.Module, base_path: str) -> None:
    """
    Save the weights of a model to files.

    Args:
        model (torch.nn.Module): The model whose weights are to be saved.
        base_path (str): The base directory path to save the weights.
    """
    try:
        state_dict = model.state_dict()
        _save_state_dict_weights(state_dict, base_path)
    except Exception as e:
        logging.error(f"Failed to save model weights: {e}")
        raise



def _save_state_dict_weights(state_dict: OrderedDict, base_path: str) -> None:
    """
    Recursively save the weights in the state dictionary.

    Args:
        state_dict (OrderedDict): The state dictionary containing the weights.
        base_path (str): The base directory path to save the weights.
    """
    for key, value in state_dict.items():
        file_path = os.path.join(base_path, key.replace(".", "_") + ".pt")
        if isinstance(value, torch.Tensor):
            save_tensor(value, file_path)
        elif isinstance(value, OrderedDict):
            sub_base_path = os.path.join(base_path, key)
            create_directory_structure(sub_base_path)
            _save_state_dict_weights(value, sub_base_path)




def create_directory_structure(base_path: str) -> None:
    """
    Create the directory structure for saving model weights.

    Args:
        base_path (str): The base directory path.
    """
    try:
        os.makedirs(base_path, exist_ok=True)
        logging.info(f"Directory structure created at {base_path}")
    except Exception as e:
        logging.error(f"Failed to create directory structure: {e}")
        raise



def extract_model_details(model: Union[torch.nn.Module, PreTrainedModel]) -> Dict[str, Any]:
    """
    Extract key-value pairs from the model's state dictionary.

    Args:
        model (Union[torch.nn.Module, PreTrainedModel]): The model to extract details from.

    Returns:
        Dict[str, Any]: The extracted key-value pairs.
    """
    try:
        state_dict = model.state_dict()
        return extract_state_dict_details(state_dict)
    except Exception as e:
        logging.error(f"Failed to extract model details: {e}")
        raise



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
                "requires_grad": value.requires_grad,
            }
        else:
            result[key] = value
    return result



def write_to_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Write the extracted data to a JSON file.

    Args:
        data (Dict[str, Any]): The data to be written to the JSON file.
        file_path (str): The path to the output JSON file.
    """
    try:
        with open(file_path, "w", encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"Model details written to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write to JSON file: {e}")
        raise



def summarizemodelforhemanth(model: torch.nn.Module) -> Dict[str, Any]:
    """Summarize a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
        Dict[str, Any]: A dictionary containing the model summary.
    """
    try:
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

                summary["layers"].append(layer_summary)

        return summary
    except Exception as e:
        logging.error(f"Failed to summarize model: {e}")
        raise



def printmodelsummaryforhemanth(model_summary: Dict[str, Any]) -> None:
    """Print the model summary in a professional and boxed format using the logging module.

    Args:
        model_summary (Dict[str, Any]): The model summary dictionary.
    """
    try:
        max_name_length = max(len(layer["name"]) for layer in model_summary["layers"])
        max_type_length = max(len(layer["type"]) for layer in model_summary["layers"])
        max_params_length = max(len(f"{layer['parameters']:,}") for layer in model_summary["layers"])
        max_trainable_length = max(len(str(layer["trainable_parameters"] > 0)) for layer in model_summary["layers"])
        max_device_length = max(len(str(layer["device"])) for layer in model_summary["layers"])

        header_format = (
            "| {name:<{name_width}} | {type:<{type_width}} | {params:<{params_width}} | "
            "{trainable:<{trainable_width}} |{device:<{device_width}}|"
        )
        row_format = (
            "| {name:<{name_width}} | {type:<{type_width}} | {params:>{params_width},} | "
            "{trainable:>{trainable_width}} | {device:>{device_width}} |"
        )
        divider = "+" + "-" * (max_name_length + 2) + "+" + "-" * (max_type_length + 2) + "+" + \
                  "-" * (max_params_length + 2) + "+" + "-" * (max_trainable_length + 2) + "+" + \
                  "-" * (max_device_length + 2) + "+"

        logging.info("+" + "-" * (max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 14) + "+")
        logging.info("|" + " Model Summary ".center(max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 14) + "|")
        logging.info("+" + "-" * (max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length + 14) + "+")
        logging.info("| Total Parameters: {:<{width},} |".format(model_summary["total_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length - 6))
        logging.info("| Trainable Parameters: {:<{width},} |".format(model_summary["trainable_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length -10))
        logging.info("| Non-Trainable Parameters: {:<{width},} |".format(model_summary["non_trainable_parameters"], width=max_name_length + max_type_length + max_params_length + max_trainable_length + max_device_length - 14))
        logging.info(divider)
        logging.info(header_format.format(
            name="Layer Name", type="Type", params="Parameters", trainable="Train", device="Device",
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
    except Exception as e:
        logging.error(f"Failed to print model summary: {e}")
        raise




def plot_model_weights(model_details: Dict[str, Any], output_dir: str) -> None:
    """
    Plot the model weights using Plotly or Matplotlib.

    Args:
        model_details (Dict[str, Any]): The extracted model details.
        output_dir (str): The directory to save the plotted images.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        with ThreadPoolExecutor() as executor:
            futures = []
            _plot_model_weights_recursive(model_details, output_dir, executor, futures)
            for future in futures:
                future.result()  # Ensure all futures are completed
        logging.info(f"Model weights plotted and saved in {output_dir}")
    except Exception as e:
        logging.error(f"Failed to plot model weights: {e}")
        raise



def _plot_model_weights_recursive(details: Dict[str, Any], output_dir: str, executor: ThreadPoolExecutor, futures: list) -> None:
    for key, value in details.items():
        if isinstance(value, dict) and "shape" in value:
            shape = value["shape"]
            file_path = os.path.join(output_dir, f"{key.replace('.', '_')}.png")
            if len(shape) == 4:  # Video (4D)
                futures.append(executor.submit(plot_4d_weights, key, shape, file_path))
            elif len(shape) == 3:  # Color Image (3D)
                futures.append(executor.submit(plot_3d_weights, key, shape, file_path))
            elif len(shape) == 2:  # Grayscale Image (2D)
                futures.append(executor.submit(plot_2d_weights, key, shape, file_path))
            elif len(shape) == 1:  # 1D
                futures.append(executor.submit(plot_1d_weights, key, shape, file_path))
        elif isinstance(value, dict):
            _plot_model_weights_recursive(value, output_dir, executor, futures)



def plot_4d_weights(key: str, shape: list, file_path: str) -> None:
    try:
        tensor_data = torch.randn(shape).numpy()
        # Take the mean across the first dimension to reduce dimensionality
        tensor_data = np.mean(tensor_data, axis=0)
        fig = go.Figure(data=go.Surface(z=tensor_data[:, :, 0]))  # Simplified to 2D slice for plotting
        fig.update_layout(title=f"4D Weights - {key}", autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.write_image(file_path)
    except Exception as e:
        logging.error(f"Failed to plot 4D weights for {key}: {e}")



def plot_3d_weights(key: str, shape: list, file_path: str) -> None:
    try:
        tensor_data = torch.randn(shape).numpy()
        tensor_data = np.mean(tensor_data, axis=0)  # Simplify to 2D
        fig = go.Figure(data=go.Surface(z=tensor_data))
        fig.update_layout(title=f"3D Weights - {key}", autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.write_image(file_path)
    except Exception as e:
        logging.error(f"Failed to plot 3D weights for {key}: {e}")



def plot_2d_weights(key: str, shape: list, file_path: str) -> None:
    try:
        tensor_data = torch.randn(shape).numpy()
        fig = go.Figure(data=go.Heatmap(z=tensor_data))
        fig.update_layout(title=f"2D Weights - {key}", autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.write_image(file_path)
    except Exception as e:
        logging.error(f"Failed to plot 2D weights for {key}: {e}")



def plot_1d_weights(key: str, shape: list, file_path: str) -> None:
    try:
        tensor_data = torch.randn(shape).numpy()
        fig = go.Figure(data=go.Scatter(y=tensor_data))
        fig.update_layout(title=f"1D Weights - {key}", autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.write_image(file_path)
    except Exception as e:
        logging.error(f"Failed to plot 1D weights for {key}: {e}")



def sample_train_Listofstr(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, train_data: List[str],
          batch_size: int = 8, epochs: int = 3, learning_rate: float = 1e-5,
          device: str = "cuda") -> AutoModelForCausalLM:
    """
    Train the model on the given data.

    Args:
        model (AutoModelForCausalLM): The model to train.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        train_data (List[str]): The training data.
        batch_size (int, optional): The batch size for training. Defaults to 8.
        epochs (int, optional): The number of training epochs. Defaults to 3.
        learning_rate (float, optional): The learning rate for training. Defaults to 1e-5.
        device (str, optional): The device to train the model on. Defaults to "cuda".

    Returns:
        AutoModelForCausalLM: The trained model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    return model


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    num_return_sequences: int = 2,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs
) -> List[str]:
    """
    Generate text using the given model and prompt with advanced decoding strategies.

    Args:
        model (AutoModelForCausalLM): The model to use for text generation.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The prompt to generate text from.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.
        num_return_sequences (int, optional): The number of sequences to generate. Defaults to 1.
        do_sample (bool, optional): Whether to use sampling for generation. Defaults to True.
        temperature (float, optional): The temperature for text generation. Defaults to 0.7.
        top_k (int, optional): The top-k sampling parameter. Defaults to 50.
        top_p (float, optional): The top-p sampling parameter. Defaults to 0.95.
        repetition_penalty (float, optional): The repetition penalty. Defaults to 1.2.
        pad_token_id (int, optional): The ID of the padding token. Defaults to None.
        eos_token_id (int, optional): The ID of the end-of-sequence token. Defaults to None.
        device (str or torch.device, optional): The device to generate text on. Defaults to "cuda".
        **kwargs: Additional keyword arguments for model.generate().

    Returns:
        List[str]: The generated text sequences.

    Raises:
        ValueError: If the input parameters are invalid.
        RuntimeError: If there's an issue with text generation.
    """
    try:
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be a positive integer")
        if temperature <= 0:
            raise ValueError("temperature must be a positive float")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be a float between 0 and 1")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be greater than or equal to 1")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    except ValueError as ve:
        raise ValueError(f"Invalid input parameter: {str(ve)}")
    except RuntimeError as re:
        raise RuntimeError(f"Error during text generation: {str(re)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


def generate_text_with_strategies(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50,
    num_return_sequences: int = 1,
    device: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate text using different decoding strategies with a text generation API.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The input prompt for text generation.
        max_length (int): Maximum length of generated text.
        num_return_sequences (int): Number of sequences to return per strategy.
        device (Optional[str]): The device to run the model on ('cuda' or 'cpu').

    Returns:
        Dict[str, str]: A dictionary containing the generated text for each decoding strategy.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    strategies = {
        "Greedy Decoding": {
            "num_beams": 1,
            "do_sample": False
        },
        "Beam Search": {
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "early_stopping": True
        },
        "Top-k Sampling": {
            "do_sample": True,
            "top_k": 50,
            "no_repeat_ngram_size": 2
        },
        "Top-p Sampling": {
            "do_sample": True,
            "top_p": 0.95,
            "no_repeat_ngram_size": 2
        },
        "Temperature Scaling": {
            "do_sample": True,
            "temperature": 0.7,
            "no_repeat_ngram_size": 2
        },
        "Diverse Beam Search": {
            "num_beams": 5,
            "num_beam_groups": 5,
            "diversity_penalty": 1.0,
            "no_repeat_ngram_size": 2
        }
    }

    generated_texts = {}

    for strategy, params in strategies.items():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **params
        )
        generated_texts[strategy] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_texts


def summary(
    model: nn.Module,
    input_size: Union[Tuple[int, ...], List[Tuple[int, ...]]],
    batch_size: int = -1,
    device: str = "cpu",
    ) -> Dict[str, Dict[str, Union[List[int], int, bool]]]:
    """
    Summarize the given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to summarize.
        input_size (Union[Tuple[int, ...], List[Tuple[int, ...]]]): The input size(s) of the model.
        batch_size (int, optional): The batch size for the model summary. Defaults to -1.
        device (str, optional): The device to use for the model summary. Defaults to "cuda".

    Returns:
        Dict[str, Dict[str, Union[List[int], int, bool]]]: A dictionary containing the model summary.
    """

    def register_hook(module: nn.Module) -> None:
        """
        Register a forward hook for the given module.

        Args:
            module (nn.Module): The module to register the forward hook for.
        """
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            """
            The forward hook function.

            Args:
                module (nn.Module): The module being hooked.
                input (Tuple[torch.Tensor]): The input tensors to the module.
                output (torch.Tensor): The output tensor from the module.
            """
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = [list(i.size()) for i in input]
            summary[m_key]["input_shape"][0][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor

    # Multiple input types handling
    if not isinstance(input_size[0], (list, tuple)):
        input_size = [input_size]

    # Batch size handling
    if batch_size == -1:
        batch_size = 1
    x = [torch.rand(batch_size, *in_size).type(dtype) for in_size in input_size]

    # Distributed training support
    if isinstance(model, DistributedDataParallel):
        model = model.module

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(register_hook)

    # Mixed precision training
    scaler = GradScaler()
    with autocast():
        with torch.no_grad():
            try:
                model(*x)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print(f"Input shape passed to model: {[inp.shape for inp in x]}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger.info("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    logger.info(line_new)
    logger.info("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # Input shape, output shape, trainable, nb_params
        output_shape = summary[layer].get("output_shape", "")
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(output_shape),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        if output_shape:
            total_output += np.prod(output_shape)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        logger.info(line_new)
    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(sum([np.prod(x) for x in input_size]) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))

    logger.info("================================================================")
    logger.info("Total params: {0:,}".format(total_params))
    logger.info("Trainable params: {0:,}".format(trainable_params))
    logger.info("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger.info("----------------------------------------------------------------")
    logger.info("Input size (MB): %0.2f" % total_input_size)
    logger.info("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger.info("Params size (MB): %0.2f" % total_params_size)
    # logger.info("Estimated Total Size (MB): %0.2f" % total_size)
    logger.info("----------------------------------------------------------------")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(summary)), [v["nb_params"] for k, v in summary.items()], align="center")
    plt.xticks(range(len(summary)), [k for k, v in summary.items()], rotation=90)
    plt.xlabel("Layers")
    plt.ylabel("Number of Parameters")
    plt.title("Model Architecture")
    plt.tight_layout()
    plt.show()

    # Plotly visualization
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])

    # Bar chart of layer-wise parameters
    fig.add_trace(go.Bar(x=[k for k, v in summary.items()], y=[v["nb_params"] for k, v in summary.items()],
                         name="Layer-wise Parameters"), row=1, col=1)

    # Pie chart of total parameters distribution
    labels = ["Trainable Parameters", "Non-trainable Parameters"]
    values = [trainable_params, total_params - trainable_params]
    fig.add_trace(go.Pie(labels=labels, values=values, name="Parameters Distribution"), row=1, col=2)

    fig.update_layout(title_text="Model Summary", height=600, width=1200,
                      template="plotly_white", showlegend=False)
    fig.show()

    return summary


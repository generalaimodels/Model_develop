import gc
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Union,Tuple



def get_device() -> torch.device:
    """
    Automatically set the device to CPU, CUDA, or other available hardware backends.
    
    Returns:
        torch.device: The chosen device.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("||Using CUDA (GPU support)||\n")
        return torch.device('cuda')
    
    # Check for other backends like MCOs or TPUs
    # Note: PyTorch doesn't have native TPU support without third-party libraries like PyTorch/XLA
    # We'll check for TPU availability using the xla library as an example
    try:
        import torch_xla.core.xla_model as xm
        print("Using TPU")
        return xm.xla_device()
    except ImportError:
        pass  # TPU not available or xla not installed
    
    # Fallback to CPU
    print("Using CPU")
    return torch.device('cpu')

def load_model(
    model_name: str,
    max_memory: Dict[Union[int, str], str],
    quantize: Optional[bool] = False,
    device_map: str = "auto"
) -> AutoModelForCausalLM:
    """
    Load the model with optional quantization.

    Args:
        model_name (str): The name of the model to load.
        max_memory (Dict[Union[int, str], str]): Max memory configuration.
        quantize (Optional[bool], optional): Whether to quantize the model. Defaults to False.
        device_map (str, optional): The device map configuration. Defaults to "auto".

    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    device = get_device()
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            max_memory=max_memory
        ).to(device)
    return model


def calculate_model_parameters(model: AutoModelForCausalLM) -> None:
    """
    Calculate and print the model parameters.

    Args:
        model (AutoModelForCausalLM): The loaded model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("|---------Total Parameters-------|:")
    print("|================================|:")
    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print('\n')

    print("|==========================Calculations::=================================|")
    print("|==========================================================================|")
    for name, param in model.named_parameters():
        param_count = param.numel()
        trainable = param.requires_grad
        print(f"|  {name} | :   |{param_count:,} |  parameters|  , Trainable |  : '{trainable}'|")

    print()
    print(f"|Total parameters calculation: sum(p.numel() for p in model.parameters()) = {total_params:,}|")
    print(f"|Trainable parameters calculation: sum(p.numel() for p in model.parameters() if p.requires_grad) = {trainable_params:,}|")


def b2mb(bytes: int) -> float:
    """
    Convert bytes to megabytes.

    Args:
        bytes (int): The number of bytes.

    Returns:
        float: The equivalent value in megabytes.
    """
    return bytes / 1024 / 1024


class ResourceMonitor:
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.peak = 0
        self.used = 0
        self.peaked = 0
        self.cpu_begin = 0
        self.cpu_end = 0
        self.cpu_peak = -1
        self.cpu_used = 0
        self.cpu_peaked = 0
        self.peak_monitoring = False
        self.process = psutil.Process()

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self) -> int:
        """
        Get resident set size memory for the current process.

        Returns:
            int: The memory used by the current process in bytes.
        """
        return self.process.memory_info().rss

    def peak_monitor_func(self) -> None:
        """
        Monitor the peak CPU memory usage.
        """
        while self.peak_monitoring:
            current_cpu_usage = self.cpu_mem_used()
            self.cpu_peak = max(current_cpu_usage, self.cpu_peak)

    def __exit__(self, *exc) -> None:
        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        print(f"Memory used: {self.used} MB, Memory peaked: {self.peaked} MB")
        print(f"CPU used: {self.cpu_used} MB, CPU peaked: {self.cpu_peaked} MB")


def load_model_and_tokenizer(
    model_name: str,
    device_map: str,
    max_memory: Dict[Union[int, str], str]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.
        device_map (str): The device map configuration.
        max_memory (Dict[Union[int, str], str]): Max memory configuration.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    with ResourceMonitor():
        # Load the GPT-2 model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            max_memory=max_memory
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
            max_memory=max_memory
        )

    return model, tokenizer


from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_io_dimensions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str) -> Tuple[int, int]:
    """
    Determine the input and output dimensions of the model.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        device (str): The device to perform computation on.

    Returns:
        Tuple[int, int]: The input and output dimensions of the model.
        
    Example:
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        device = 'cpu'
        input_dim, output_dim = get_model_io_dimensions(model, tokenizer, device)
        print(input_dim, output_dim)  # Example output: 13, 50257
    """
    try:
        # Use a sample text to encode
        sample_text = "Hello, world!"
        encoded_input = tokenizer.encode_plus(
            sample_text, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_dim = encoded_input['input_ids'].shape[-1]  # Input dimension
        print(f"Token's shape : {encoded_input['input_ids'].shape}")
        
        # Move the encoded input to the correct device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Move the model to the correct device
        model.to(device)
        
        # Use the model to get the output for the encoded input
        with torch.no_grad():
            output = model(**encoded_input)
        
        print(f"Output shape : {output.logits.shape}")
        # The output dimension is the last dimension of the logits tensor shape
        output_dim = output.logits.shape[-1]
        
        return input_dim, output_dim
    except Exception as e:
        # Handle potential exceptions that could occur
        print(f"An error occurred: {e}")
        raise



# Configuration for model loading
max_memory = {0: "10GIB", "cpu": "30GB"}
device_map = "auto"
model_name = "gpt2"

with ResourceMonitor():
    model = load_model( model_name=model_name, device_map=device_map,max_memory=max_memory)



# Example usage within your code:
device = get_device()
model, tokenizer = load_model_and_tokenizer(model_name=model_name, device_map=device_map, max_memory=max_memory)
model.to(device)  # Move the model to the chosen device
calculate_model_parameters(model=model)
input_dim, output_dim = get_model_io_dimensions(model=model, tokenizer=tokenizer, device=device)
print(f"Input Dimension: {input_dim}")
print(f"Output Dimension: {output_dim}")


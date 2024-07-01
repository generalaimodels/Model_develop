import gc
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Union,Tuple

import gc
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Union, Tuple,List
import matplotlib.pyplot as plt
import numpy as np



def get_device() -> torch.device:
    """
    Automatically set the device to CPU, CUDA, or other available hardware backends.
    
    Returns:
        torch.device: The chosen device.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("\n" + "="*30)
        print("| Using CUDA (GPU support) |")
        print("="*30 + "\n")
        return torch.device('cuda')
    
    # Check for other backends like MCOs or TPUs
    # Note: PyTorch doesn't have native TPU support without third-party libraries like PyTorch/XLA
    # We'll check for TPU availability using the xla library as an example
    try:
        import torch_xla.core.xla_model as xm
        print("\n" + "="*30)
        print("| Using TPU |")
        print("="*30 + "\n")
        return xm.xla_device()
    except ImportError:
        pass  # TPU not available or xla not installed
    
    # Fallback to CPU
    print("\n" + "="*30)
    print("| Using CPU |")
    print("="*30 + "\n")
    return torch.device('cpu')



def calculate_model_parameters(model: AutoModelForCausalLM) -> None:
    """
    Calculate and print the model parameters.

    Args:
        model (AutoModelForCausalLM): The loaded model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"╔══════════════════════════════════╗")
    print(f"║        Total Parameters          ║")
    print(f"╠══════════════════════════════════╣")
    print(f"║ Total parameter: {total_params:,}   ║")
    print(f"║ Trainable parameters:{trainable_params:,} ║")
    print(f"║ Non-trainable parameters: {total_params - trainable_params:,}      ║")
    print(f"╚══════════════════════════════════╝")
    print()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                            Calculations                                  ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    for name, param in model.named_parameters():
        param_count = param.numel()
        trainable = param.requires_grad
        print(f"║ {name:<60} │ {param_count:>12,} parameters │ Trainable: {str(trainable):<5}║")

    print(f"╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ Total parameters calculation:                                            ║")
    print(f"║ sum(p.numel() for p in model.parameters()) = {total_params:,}               ║")
    print(f"║ Trainable parameters calculation:                                        ║")
    print(f"║ sum(p.numel() for p in model.parameters() if p.requires_grad) = {trainable_params:,}║")
    print(f"╚══════════════════════════════════════════════════════════════════════════╝")
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
        print("╔═════════════════════════════════════════════════╗")
        print("║               Memory Usage                      ║")
        print("╠═════════════════════════════════════════════════╣")
        print(f"║ Memory used:  {self.used:>10} MB                ║")
        print(f"║ Memory peaked: {self.peaked:>9} MB               ║")
        print("╚═════════════════════════════════════════════════╝")
        
        print("╔═════════════════════════════════════════════════╗")
        print("║                 CPU Usage                       ║")
        print("╠═════════════════════════════════════════════════╣")
        print(f"║ CPU used:  {self.cpu_used:>13} MB                     ║")
        print(f"║ CPU peaked: {self.cpu_peaked:>12} MB                     ║")
        print("╚═════════════════════════════════════════════════╝")







def create_tokenizer(
    tokenizer_name_or_path: Union[str,List] ) -> AutoTokenizer:
    """
    Initializes and returns a tokenizer based on the specified pretrained model or path.

    Args:
        tokenizer_name_or_path (str): The name or path of the tokenizer's pretrained model.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # Set special tokens if they are not already set
    special_tokens = {
        'pad_token': tokenizer.eos_token,
        'bos_token': tokenizer.eos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.eos_token,
        'sep_token': tokenizer.eos_token,
        'cls_token': tokenizer.eos_token,
        'mask_token':tokenizer.eos_token
    }
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, f"{token_name}_id") is None:
            setattr(tokenizer, token_name, token_value)
    
    return tokenizer


def load_model(
    model_name: str,
    max_memory: Dict[Union[int, str], str],
    quantize: Optional[bool] = True,
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
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            max_memory=max_memory
        ).to(device)
    return model

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


def get_model_io_dimensions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str) -> Tuple[int, int]:
    """
    Determine the input and output dimensions of the model.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        device (str): The device to perform computation on.

    Returns:
        Tuple[int, int]: The input and output dimensions of the model.
    """
    # Use a sample text to encode
    sample_text = "Hello, world!"
    encoded_input = tokenizer.encode_plus(sample_text, return_tensors="pt",truncation=True,max_length=512)
    input_dim = encoded_input['input_ids'].shape[-1]  # The last dimension of the input tensor shape is the input dimension
    print("Token's shape :",encoded_input['input_ids'].shape)
    # Move the encoded input to the correct device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Move the model to the correct device
    # model = model.to(device)

    # Use the model to get the output for the encoded input
    with torch.no_grad():
        output = model(**encoded_input)
    
    print("Output shape :",output.logits.shape)
    # Assuming the output logits are in the first element of the output tuple
    output_dim = output.logits.shape[-1]  # The last dimension of the logits tensor shape is the output dimension
    
    return input_dim, output_dim



def visualize_text_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    num_tokens: int,
    device: torch.device
) -> None:
    """
    Visualize the model's text generation process.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        input_text (str): The input text to generate from.
        num_tokens (int): The number of tokens to generate.
        device (torch.device): The device to perform computation on.
    """
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            do_sample=True,
            top_k=100,
            top_p=0.1,
            num_return_sequences=1
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Tokenize the generated text
    generated_tokens = tokenizer.tokenize(generated_text)

    # Create a grid of subplots
    num_rows = num_tokens // 5 + 1
    fig, axs = plt.subplots(num_rows, 5, figsize=(10, 4 * num_rows))
    axs = axs.flatten()

    # Iterate over the generated tokens
    for i in range(num_tokens):
        # Get the token embedding
        token_embedding = model.model.embed_tokens.weight[output[0, i]].detach().cpu().numpy()

        # Plot the token embedding
        axs[i].imshow(token_embedding.reshape(1, -1), cmap="viridis", aspect="auto")
        axs[i].set_title(f"{i}:{generated_tokens[i]}")
        axs[i].axis("off")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()

# # Configuration for model loading
# max_memory = {0: "10GIB", "cpu": "30GB"}
# device_map = "auto"
# model_name = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
# tokenizer_name="TinyLlama/TinyLlama-1.1B-step-50K-105b"




# with ResourceMonitor():
#     model = load_model( model_name=model_name, device_map=device_map,max_memory=max_memory)
#     tokenizer=create_tokenizer(tokenizer_name_or_path=tokenizer_name, )

# calculate_model_parameters(model=model)
# input_dim, output_dim = get_model_io_dimensions(model=model, tokenizer=tokenizer, device=device)
# print(f"Input Dimension: {input_dim}")
# print(f"Output Dimension: {output_dim}")

# # Visualize text generation
# input_text = "tell a concepts of love? "
# num_tokens = 100
# visualize_text_generation(model, tokenizer, input_text, num_tokens, device)

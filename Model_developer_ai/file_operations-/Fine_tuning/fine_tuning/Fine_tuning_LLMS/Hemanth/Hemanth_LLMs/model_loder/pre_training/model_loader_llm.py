# 🔒 Confidentiality Notice 
#
# DARFA-LLMs, is developed under the auspices of CDAC (Center for Development of Advanced Computing). 
# The project's codebase, algorithms, and any associated documentation are considered confidential 
# and proprietary to CDACB. 
#
# Copyright Statement ©️
#
# All code within this project is the intellectual property of Kandimalla Hemanth. 👨‍💻 Any reproduction, 
# distribution, or unauthorized use of the code or its components without explicit permission from the
# copyright holder is strictly prohibited. 
#
# Project Purpose 
#
# The DARFA-llms project is dedicated to advancing human development through the application of artificial 
# intelligence (AI) technologies. Its mission is to address real-world challenges and contribute 
# to the betterment of society. 
#
#  Non-Disclosure Agreement (NDA) 
#
# Access to this Python project and its associated materials is restricted to authorized personnel only.
# 🔒 By accessing or using any part of this project, you agree to maintain strict confidentiality 
# and adhere to any applicable non-disclosure agreements (NDAs) or confidentiality policies established
# by CDACB. 
#
# ⚠️ Disclaimer of Liability 📜
#
# While every effort has been made to ensure the accuracy and reliability of the code and information 
# contained within this project, CDACB and Kandimalla Hemanth shall not be held liable for any direct, 
# indirect, incidental, special, or consequential damages arising out of the use or inability to use the
# project, even if advised of the possibility of such damages. 
#
# 🚫 Usage Restrictions 📝
#
# The code and algorithms developed as part of the DARFA project are intended for research, educational,
# and non-commercial purposes only. Any commercial use, modification, or adaptation of the codebase 
# requires explicit written consent from CDACB and Kandimalla Hemanth.
#
#  Ethical Considerations 
#
# The DARFA project adheres to ethical principles and guidelines governing the responsible 
# use of AI technologies. All efforts are made to ensure that the project's objectives 
# align with ethical standards and promote positive outcomes for humanity. 



import gc
import threading
import psutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Optional, Union, Tuple,List



import plotly.subplots as sp



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

def load_model_TEST(model_name_or_path: Union[str,List]) -> AutoModelForCausalLM:
    """
    Function to load a transformers model.
    
    Args:
      model_name_or_path (Union[str, Path]): The name or path of the model.

    Returns:
        model (AutoModelForCausalLM): The loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                token ="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",
                                trust_remote_code=True)
    return model

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

def get_model_io_dimensions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text:str,device: str) -> Tuple[int, int]:
    """
    Determine the input and output dimensions of the model.

    Args:
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The associated tokenizer.
        device (str): The device to perform computation on.

    Returns:
        Tuple[int, int]: The input and output dimensions of the model.
    """
    
    encoded_input = tokenizer.encode_plus(text, return_tensors="pt",truncation=True,max_length=512)
    input_dim = encoded_input['input_ids'].shape[-1] 
    print("Token's shape :",encoded_input['input_ids'].shape)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    
    model = model.to(device)

    with torch.no_grad():
        output = model(**encoded_input)
    
    print("Output shape :",output.logits.shape)
   
    output_dim = output.logits.shape[-1] 
    
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

def visualize_text_generation_test(
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

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + num_tokens,
            do_sample=True,
            top_k=100,
            top_p=0.1,
            num_return_sequences=1
        )


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


    generated_tokens = tokenizer.tokenize(generated_text)


    num_rows = num_tokens // 5 + 1
    fig = sp.make_subplots(rows=num_rows, cols=5, subplot_titles=[f"{i}:{token}" for i, token in enumerate(generated_tokens[:num_tokens])])


    for i in range(num_tokens):

        token_embedding = model.embed_tokens.weight[output[0, i]].detach().cpu().numpy()

        fig.add_heatmap(z=token_embedding.reshape(1, -1), colorscale="Viridis", showscale=False, row=(i // 5) + 1, col=(i % 5) + 1)

    fig.update_layout(height=400 * num_rows, width=1000, title_text="Token Embeddings")


    fig.show()

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str, max_length: int, num_beams: int) -> str:
    

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text











#========================================================development=======================
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
#========================================================development=======================

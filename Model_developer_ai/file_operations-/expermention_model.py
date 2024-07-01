import gc
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from typing import Dict, Optional,Union
# def load_model(model_name: str, quantize: Optional[bool] = False) -> AutoModelForCausalLM:
#     """
#     Load the model with optional quantization.

#     Args:
#         model_name (str): The name of the model to load.
#         quantize (bool, optional): Whether to quantize the model. Defaults to False.

#     Returns:
#         AutoModelForCausalLM: The loaded model.
#     """
#     if quantize:
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=False,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#         model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(model_name)

#     return model


def calculate_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("|---------Total Parameters-------|:")
    print("|================================|:")
    print(f"Total parameters::         {total_params:,}")
    print(f"Trainable parameters::     {trainable_params:,}")
    print(f"Non-trainable parameters:: {total_params - trainable_params:,}")
    print('\n')
    
    print("|==========================Calculations::=================================|")
    print("|==========================================================================|")
    for name, param in model.named_parameters():
        param_count = param.numel()
        trainable = param.requires_grad
        print(f"|  {name} | :   |{param_count:,} |  parameters|   , Trainable |  : '{trainable}'|")
    
    print()
    print(f"|Total parameters calculation: sum(p.numel() for p in model.parameters()) = {total_params:,}|")
    print(f"|Trainable parameters calculation: sum(p.numel() for p in model.parameters() if p.requires_grad) = {trainable_params:,}|")




# Function to convert bytes to megabytes
def b2mb(bytes: int) -> float:
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
        """Get resident set size memory for the current process."""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        while self.peak_monitoring:
            current_cpu_usage = self.cpu_mem_used()
            self.cpu_peak = max(current_cpu_usage, self.cpu_peak)

    def __exit__(self, *exc):
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
    max_memory: Dict
):
    with ResourceMonitor():
        # Load the GPT-2 model
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, max_memory=max_memory)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map, max_memory=max_memory)

    return model, tokenizer



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
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, quantization_config=bnb_config, device_map=device_map, max_memory=max_memory)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, device_map=device_map, max_memory=max_memory)
    return model


# Configuration for model loading
max_memory = {0: "10GIB", "cpu": "30GB"}
device_map = "auto"
model_name = "gpt2"
with ResourceMonitor():
  model= load_model(model_name, device_map=device_map, max_memory=max_memory)
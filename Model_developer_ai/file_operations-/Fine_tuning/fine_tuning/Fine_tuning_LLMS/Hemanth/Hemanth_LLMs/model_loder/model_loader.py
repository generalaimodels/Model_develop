
import torch







from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
 
)

from peft import (
    
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
    
)
load_in_8bit = False
# 4-bit quantization
Device=''
USE_NESTED_QUANT=True            # use_nested_quant
BNB_4BIT_COMPUTE_DTYPE="bfloat16"# bnb_4bit_compute_dtype

compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)



#========================LORA======================
# LORA
LORA_R=8                         # lora_r
LORA_ALPHA=32                    # lora_alpha
LORA_DROPOUT=0.0                 # lora_dropout
LORA_TARGET_MODULES="c_proj,c_attn,q_attn,c_fc,c_proj"    # lora_target_modules

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

def Load_model_AutoModelForCausalLM(model_name_or_path:str):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model

def load_model_AutoModelForCausalLM_Quantization(model_name_or_path:str):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=Device,
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
)
    return model


def load_model_AutoModelForCausalLM_Lorq(model_name_or_path:str):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    )
    model = get_peft_model(model, peft_config)
    return model


def load_model_AutoModelForCausalLM_Lorq_quant(model_name_or_path:str):

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=Device,
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=True,
)
    model = get_peft_model(model, peft_config)
    return model




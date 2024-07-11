import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline ,BitsAndBytesConfig

torch.random.manual_seed(0) 
MODEL_NAME='/scratch/hemanth/LLMs/Fraud_detection/best_model_epoch_1'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
)
model=AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    # cache_dir=cache_dir,
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True, 
    # token=Token,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct') 

messages = [ 
    {"role": "system", "content": "You are a fraud detection AI assistant."}, 
    {"role": "user", "content": "0,0.3,0.986506310633034,-1,25,40,0.0067353870811739,102.45371092469456,AA,1059,13096.035018400871,7850.955007125409,6742.080561007602,5,5,CB,163,1,BC,0,1,9,0,1500.0,0,INTERNET,16.224843433978073,linux,1,1,"}, 
   
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": True, 
    
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])

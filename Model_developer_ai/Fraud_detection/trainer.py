import os
import logging
from typing import Any, Dict, Union,Tuple
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from tqdm import tqdm

from data_processing import (
    load_data,
    create_data_split,
    preprocess_data,
    create_data_loaders
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model_weights(model: PeftModel, save_path: str) -> None:
    try:
        model.save_pretrained(save_path)
        logging.info(f"Model weights saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving model weights: {e}")

def calculate_perplexity(loss: torch.Tensor) -> float:
    try:
        perplexity = torch.exp(loss)
        logging.info(f"Calculated perplexity: {perplexity.item()}")
        return perplexity.item()
    except OverflowError as e:
        logging.error("OverflowError in calculating perplexity: {e}")
        return float('inf')

def train_and_evaluate(
    train_loader: DataLoader,
    test_loader: DataLoader,
    base_model_path: str,
    config: Union[LoraConfig, Dict[str, Any]],
    num_epochs: int,
    lr: float,
    device: str
) -> None:
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, low_cpu_mem_usage=True)
    
    # # Apply PEFT configuration
    # peft_config = config if isinstance(config, LoraConfig) else LoraConfig(**config)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * num_epochs),
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
   
            print(batch)
            outputs = model(**batch)
            
            # outputs = model(torch.stack(batch['input_ids']))
            loss = outputs.loss
            print("loss:", loss)
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_loader)
        train_ppl = calculate_perplexity(train_epoch_loss)
        logging.info(f"Epoch {epoch+1}: Train Perplexity: {train_ppl}, Train Loss: {train_epoch_loss.item()}")
        
        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(test_loader)
        eval_ppl = calculate_perplexity(eval_epoch_loss)
        logging.info(f"Epoch {epoch+1}: Eval Perplexity: {eval_ppl}, Eval Loss: {eval_epoch_loss.item()}")
        
        save_model_weights(model, save_path=f"weights/epoch_{epoch + 1}")
        
    logging.info("Training and evaluation completed.")



def pre_processing_pipeline(file_path: str, model_name: str, max_length: int = 10, stride: int = 2, batch_size: int = 5) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """
    Main function to process the data and create DataLoaders.
    """
    
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    dataset = load_data(file_path)
    train_dataset, test_dataset = create_data_split(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
    # tokenizer.add_special_tokens(special_tokens)
    system_prompt = "You are an AI assistant tasked with detecting fraud. Analyze the following content carefully."
    train_dataset = preprocess_data(train_dataset, tokenizer, max_length, stride, system_prompt)
    test_dataset = preprocess_data(test_dataset, tokenizer, max_length, stride, system_prompt)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size)
    logger.info("Data processing completed successfully")
    return train_loader, test_loader

train_loader, test_loader,=pre_processing_pipeline(
    file_path=r'C:\Users\heman\Desktop\Coding\Synthetic_Financial_datasets_log.csv1',
    model_name=r"C:\Users\heman\Desktop\Coding\data\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e",
    max_length=100,
    stride=20,
    batch_size=5
)




config = {
    "r":64, 
    "lora_alpha":128,
    "lora_dropout":0.0, 
    "target_modules":["embed_tokens", "lm_head", "q_proj", "v_proj"]
}
base_model_path=r'C:\Users\heman\Desktop\Coding\data\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e'
# Example usage
train_and_evaluate(train_loader, test_loader, base_model_path, config, num_epochs=3, lr=5e-5, device='cpu')
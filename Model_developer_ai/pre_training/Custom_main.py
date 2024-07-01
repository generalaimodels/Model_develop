

import argparse
import yaml
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from evaluate import load
import data_loader_llm
import model_loader_llm
import data_load_cleaning
import pre_processing_data
import fine_tuning_pre_processing
import functionality_LLMS
import Testing_preprocessing
from pathlib import Path
import plotly.graph_objects as go



file = Path(__file__)
print('File:', file)
print('Parent directory:', file.parent)

dataset = data_loader_llm.load_and_prepare_dataset("fka/awesome-chatgpt-prompts")
tokenizer = model_loader_llm.create_tokenizer('gpt2')
model = model_loader_llm.load_model_TEST(model_name_or_path='gpt2')

train_datasets1 = Testing_preprocessing.ConstantLengthDataset1(
    tokenizer,
    dataset['train'],
    infinite=True,
    seq_length=512,
    chars_per_token=2.5,
    fim_rate=0.5,
    fim_spm_rate=0.25,
    seed=0,
)

test_datasets1 = Testing_preprocessing.ConstantLengthDataset1(
    tokenizer,
    dataset['test'],
    infinite=True,
    seq_length=512,
    chars_per_token=2.5,
    fim_rate=0.5,
    fim_spm_rate=0.25,
    seed=0,
)

# Define the data collator
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metrics = load("accuracy")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",

)

trainer = Trainer(
    model=model,
    train_dataset=train_datasets1,
    eval_dataset=test_datasets1,
    data_collator= default_data_collator,
    compute_metrics=metrics,
    args=training_args,
    
)

# Train the model
train_result = trainer.train()

# Save the trained model
trainer.save_model("./output")

# Plot training loss using Plotly
epochs = list(range(1, 10 + 1))
train_losses = train_result.history['train_loss']

fig = go.Figure(data=go.Scatter(x=epochs, y=train_losses, mode='lines+markers'))
fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
fig.write_html("training_loss.html")

# Perform inference on sample text
sample_text = "What is the capital of France?"
inputs = tokenizer(sample_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Sample Text:", sample_text)
print("Generated Text:", generated_text)







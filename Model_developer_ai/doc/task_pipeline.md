# Text_generation 

```python
model = AiModelForHemanth.load_model(model_type="causal_lm", model_name_or_path="gpt2", cache_dir=r"E:\LLMS\Fine-tuning\data")
print(f"Model config: {model.config}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
tokenizer = AdvancedPreProcessForHemanth(model_type="text", pretrained_model_name_or_path="gpt2", cache_dir=r"E:\LLMS\Fine-tuning\data")
tokenizer = tokenizer.process_data()
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
# Test tokenizer
test_text = "Hello, world!"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"Test tokenization: '{test_text}' -> {encoded} -> '{decoded}'")
# Test model forward pass
test_input = torch.tensor([encoded], dtype=torch.long).to(model.device)
with torch.no_grad():
    test_output = model(test_input)
print(f"Test output shape: {test_output.logits.shape}")
generator = LLMGenerator(
    model=model,
    tokenizer=tokenizer,
    max_seq_len=512,
    max_batch_size=3,
)

prompts = [
        "Once upon a time, in a land far away,",
        "The quick brown fox",
        "In the year 2050, technology had advanced to the point where",
    ]

completions = generator.text_completion(
        prompts=prompts,
        max_gen_len=100,
        temperature=3.0,  # Increased temperature
        top_p=0.5,
        logprobs=True,
        echo=True,
    )
for i, completion in enumerate(completions):
    print(f"Prompt {i + 1}:")
    print(f"Input: {prompts[i]}")
    print(f"Generated text: {completion['generated_text']}")
    print(f"Number of generated tokens: {completion['num_generated_tokens']}")
    print("---")

```
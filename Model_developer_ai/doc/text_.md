```python
if __name__ == "__main__":
    epochs = 5
    learning_rate = 2e-5

    Model and Data Loaders
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader, test_loader, eval_loader = create_data_loaders(dataset, batch_size)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training
    history = train(model, train_loader, optimizer, scheduler, epochs, device)
    plot_training_metrics(history)

    # Evaluation
    eval_results = evaluate(model, eval_loader, device)
    print("Evaluation Results:", eval_results)

    # Testing (You can structure this similarly to evaluation)
    test_results = evaluate(model, test_loader, device) 
    print("Testing Results:", test_results) 
```
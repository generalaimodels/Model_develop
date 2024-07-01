from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define a collate function for padding
def collate_fn(batch):
    # Extract input_ids from the batch, which should already be tensorized by the dataset's __getitem__
    input_ids = [item['input_ids'] for item in batch]

    # Pad sequences to the max length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True)

    # Return a dictionary with padded input_ids
    return {'input_ids': input_ids}

# Assuming 'train_datasets' and 'eval_datasets' are instances of the ConstantLengthDataset containing tokenized data
train_loader = DataLoader(
    dataset=train_datasets,
    batch_size=batch_size,
    shuffle=(shuffle if shuffle is not None else True),  # Shuffle by default for training
    sampler=sampler,
    collate_fn=collate_fn,  # Use our custom collate function defined above
    pin_memory=pin_memory,
    drop_last=drop_last,
    timeout=timeout,
    num_workers=num_workers,
    # Other arguments can be passed if needed (commented out here for clarity)
    # worker_init_fn=worker_init_fn,
    # multiprocessing_context=multiprocessing_context,
    # generator=generator,
    # prefetch_factor=prefetch_factor,
    # persistent_workers=persistent_workers,
    # pin_memory_device=pin_memory_device,
)

eval_loader = DataLoader(
    dataset=eval_datasets,
    batch_size=batch_size,
    shuffle=(shuffle if shuffle is not None else False),  # Typically no need to shuffle for evaluation
    sampler=sampler,
    collate_fn=collate_fn,  # Use our custom collate function defined above
    pin_memory=pin_memory,
    drop_last=drop_last,
    timeout=timeout,
    num_workers=num_workers,
    # Other arguments can be passed if needed (commented out here for clarity)
    # worker_init_fn=worker_init_fn,
    # multiprocessing_context=multiprocessing_context,
    # generator=generator,
    # prefetch_factor=prefetch_factor,
    # persistent_workers=persistent_workers,
    # pin_memory_device=pin_memory_device,
)


import evaluate

# Load the metrics
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')  # Use 'binary' for binary classification
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

# Add the compute_metrics function to the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_datasets,  # Should be train_datasets here, not eval_datasets
    eval_dataset=eval_datasets,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,  # Add this line
)

trainer.train()
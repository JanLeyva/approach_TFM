train_dataset = load_examples(tokenizer, evaluate=False)
eval_dataset = load_examples(tokenizer, evaluate=True)   

train_sampler = RandomSampler(train_dataset)
eval_sampler = SequentialSampler(eval_dataset)

train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collate_fn
    )


eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=eval_batch_size, 
        collate_fn=collate_fn
    )
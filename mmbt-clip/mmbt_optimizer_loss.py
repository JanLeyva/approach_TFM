# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", 
            "LayerNorm.weight"
           ]
weight_decay = 0.0005

optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

t_total = (len(train_dataloader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = t_total // 10

optimizer = MADGRAD(optimizer_grouped_parameters, lr=2e-4)

scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, t_total
    )

criterion = nn.BCEWithLogitsLoss()
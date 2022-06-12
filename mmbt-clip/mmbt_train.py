optimizer_step = 0
global_step = 0
train_step = 0
tr_loss, logging_loss = 0.0, 0.0
best_valid_auc = 0.75
global_steps_list = []
train_loss_list = []
val_loss_list = []
val_acc_list = []
val_auc_list = []
eval_every = len(train_dataloader) // 7
running_loss = 0
file_path="models/"

model.zero_grad()

for i in range(num_train_epochs):
    print("Epoch", i+1, f"from {num_train_epochs}")
    whole_y_pred=np.array([])
    whole_y_t=np.array([])
    for step, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        labels = batch[5]
        inputs = {
            "input_ids": batch[0],
            "input_modal": batch[2],
            "attention_mask": batch[1],
            "modal_start_tokens": batch[3],
            "modal_end_tokens": batch[4],
            "return_dict": False
        }
        outputs = model(**inputs)
        logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = criterion(logits, labels)        
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward()
        
        tr_loss += loss.item()
        running_loss += loss.item()
        global_step += 1
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule         
            
            optimizer_step += 1
            optimizer.zero_grad()   
                        
        if (step + 1) % eval_every == 0:
            
            average_train_loss = running_loss / eval_every
            train_loss_list.append(average_train_loss)
            global_steps_list.append(global_step)
            running_loss = 0.0  
            
            val_result = evaluate(model, tokenizer, criterion, eval_dataloader)
            
            val_loss_list.append(val_result['loss'])
            val_acc_list.append(val_result['accuracy'])
            val_auc_list.append(val_result['AUC'])
            
            # checkpoint
            if val_result['AUC'] > best_valid_auc:
                best_valid_auc = val_result['AUC']
                val_loss = val_result['loss']
                val_acc = val_result['accuracy']
                model_path = f'{file_path}/model-embs{num_image_embeds}-seq{max_seq_length}-auc{best_valid_auc:.3f}-loss{val_loss:.3f}-acc{val_acc:.3f}.pt'
                print(f"AUC improved, so saving this model")  
                save_checkpoint(model_path, model, val_result['loss'])              
            
            print("Train loss:", f"{average_train_loss:.4f}", 
                  "Val loss:", f"{val_result['loss']:.4f}",
                  "Val acc:", f"{val_result['accuracy']:.4f}",
                  "AUC:", f"{val_result['AUC']:.4f}")   
    print('\n')
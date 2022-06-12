def evaluate(model, tokenizer, criterion, dataloader, tres = 0.5): 
    
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    proba = None
    out_label_ids = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
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
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > tres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:            
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > tres, axis=0)
            proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps

    result = {
        "loss": eval_loss,
        "accuracy": accuracy_score(out_label_ids, preds),
        "AUC": roc_auc_score(out_label_ids, proba),
        "micro_f1": f1_score(out_label_ids, preds, average="micro"),
        "prediction": preds,
        "labels": out_label_ids,
        "proba": proba
    }
    
    return result
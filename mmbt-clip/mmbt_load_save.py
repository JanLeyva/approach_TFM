
def load_examples(tokenizer, evaluate=False):
    path = os.path.join(data_dir, "dev_seen_clean.jsonl" if evaluate else f"train_augmented.jsonl")
    transforms = preprocess
    dataset = JsonlDataset(path, tokenizer, transforms, max_seq_length - num_image_embeds - 2)
    return dataset
    
def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    
def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']
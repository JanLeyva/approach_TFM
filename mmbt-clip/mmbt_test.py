import pandas as pd

    num_labels = 1
    data_dir = './dataset'
    test_batch_size = 16

    class TestJsonlDataset(Dataset):
        def __init__(self, data_path, tokenizer, transforms, max_seq_length):
            self.data = [json.loads(l) for l in open(data_path)]
            self.data_dir = os.path.dirname(data_path)
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            self.transforms = transforms

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
            start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
            sentence = sentence[:self.max_seq_length]

            id = torch.LongTensor([self.data[index]["id"]])        
            image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
            sliced_images = slice_image(image, 288)
            sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
            image = resize_pad_image(image, image_encoder_size)
            image = np.array(self.transforms(image))        
            sliced_images = [image] + sliced_images        
            sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

            return {
                "image_start_token": start_token,            
                "image_end_token": end_token,
                "sentence": sentence,
                "image": sliced_images,
                "id": id,
            }

    def final_collate_fn(batch):
        lens = [len(row["sentence"]) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)

        mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
        text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            text_tensor[i_batch, :length] = input_row["sentence"]
            mask_tensor[i_batch, :length] = 1

        img_tensor = torch.stack([row["image"] for row in batch])
        id_tensor = torch.stack([row["id"] for row in batch])
        img_start_token = torch.stack([row["image_start_token"] for row in batch])
        img_end_token = torch.stack([row["image_end_token"] for row in batch])

        return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, id_tensor

    def load_test_examples(test_file="test_seen.jsonl"):
        path = os.path.join(data_dir, test_file)
        dataset = TestJsonlDataset(path, tokenizer, preprocess, max_seq_length - num_image_embeds - 2)
        return dataset

    def final_prediction(model, dataloader): 
        preds = None
        proba = None
        all_ids = None
        for batch in tqdm(dataloader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                ids = batch[5]
                inputs = {
                    "input_ids": batch[0],
                    "input_modal": batch[2],
                    "attention_mask": batch[1],
                    "modal_start_tokens": batch[3],
                    "modal_end_tokens": batch[4],
                    "return_dict": False
                }
                outputs = model(**inputs)
                logits = outputs[0]
            if preds is None:
                all_ids = ids.detach().cpu().numpy()
                preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
                proba = torch.sigmoid(logits).detach().cpu().numpy()            
            else:  
                all_ids = np.append(all_ids, ids.detach().cpu().numpy(), axis=0)
                preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
                proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)

        result = {
            "ids": all_ids,
            "preds": preds,
            "probs": proba,
        }

        return result

    final_test = load_test_examples()

    final_test_sampler = SequentialSampler(final_test)

    final_test_dataloader = DataLoader(
            final_test, 
            sampler=final_test_sampler, 
            batch_size=test_batch_size, 
            collate_fn=final_collate_fn
        )
class JsonlDataset(Dataset):
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

        label = torch.FloatTensor([self.data[index]["label"]])

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
            "label": label            
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs
    
    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels
   
  def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1
    
    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor
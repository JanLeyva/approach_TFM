model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
transformer_config = AutoConfig.from_pretrained(model_name) 
transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
img_encoder = ClipEncoderMulti(num_image_embeds)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

config = MMBTConfig(transformer_config, num_labels=num_labels, modal_hidden_size=image_features_size)
model = MMBTForClassification(config, transformer, img_encoder)
model.to(device);
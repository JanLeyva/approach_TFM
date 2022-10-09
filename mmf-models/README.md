# VisualBERT-mmf


The VisualBERT use MMF framework. This model is tested with python ython
3.7.11, Tesla P100 and 32 GB. Not use the official repository since the omegacon
error are not fix. Instead use https://github.com/JanLeyva/mmf.

You can run the model in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dq5u-cFHVc3MqoP6nT5DPc7k1Uy7ltj4?usp=sharing).


1. Once the dataset is download, we must unzip and store in the correct format:

```bash
mmf_convert_hm --zip_file="hateful_memes.zip" \
				--password="password" \
				--bypass_checksum 1
```

2. Then, download the features and store in `~/root/.cache/torch/mmf/data/datasets/hateful_memes/`defaults/ or use the default features. This will store in the property format the dataset in `~/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/`.

3. Now we can star the train and fine-tune the model:

```bash
python /content/approach_TFM/mmf-models/grid-search/grid-search.py
```

4. Once the models are run we can inference with the following script:

```bash
python ./approach_TFM/mmf-models/grid-search/inference.py \
		--path_pkt '/content/path_with_pkt_models_from_grid-search' \
		--output_path '/content/results' \
		--dst_path '/content/files'
```

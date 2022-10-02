# In-painting Images




Once, the dataset is **downloaded** and **unzipped** we can start in-painting the images.
The code used in this section is borrowed from [HimariO](https://github.com/HimariO/HatefulMemesChallenge), he use [mmdetectionmeme](https://github.com/HimariO/mmdetection-meme)
repository for in-painting the memes. The dataset must be stored in the
route `/content/data/hateful_memes`. Tested in environment with python 3.7.13.


1. Detect: First step is detect where the text is with an OCR, using ocr.py script.

```bash
#!/bin/bash
python3 /content/HatefulMemesChallenge/data_utils/ocr.py detect \
/content/data/hateful_memes
```

2. OCR: Once the text is detected, the second step is transform the text points

```bash
#!/bin/bash
python3 ./HatefulMemesChallenge/data_utils/ocr.py point_to_box \
./data/hateful_memes/ocr.json
```

3. OCR to Box: now, we going to generate the mask. First mask with the text in black and the other one with black image and text box in white.

```bash
#!/bin/bash
python3 ./HatefulMemesChallenge/data_utils/ocr.py generate_mask \
./data/hateful_memes/ocr.box.json \
./data/hateful_memes/img \
./data/img_mask_3px
```

4. In-painting Images: Last step is predict what is behind the image, with this process we can in-painting the image.

```bash
#!/bin/bash
python3 ./mmediting-meme/demo/inpainting_demo.py \
./mmediting-meme/configs/inpainting/deepfillv2_256x256_8x2_places.py \
./pretrain_model/deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
./data/img_mask_3px/ /content/data/img_clean
```

![](/inpainting_hm/img/gen_mask.png)

![](/inpainting_hm/img/in-painting.png)
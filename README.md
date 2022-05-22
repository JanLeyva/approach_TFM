# First approach HM competition

This repsository content all the code needed to reproduce que approach for Hateful Memes comptetition by Facebook AI. The approach follow this structure:

1. **Inpainting the images** [notebook](https://colab.research.google.com/drive/1XBiNhKOV4uv532swUWaXcT_VatU7qGl2#scrollTo=JOlHTcZBv-2B): in order to get more information from images we try to inpainting the words and got only the image. The code is borrowed from [github](https://github.com/HimariO/mmdetection-meme.git) and [github](https://github.com/HimariO/HatefulMemesChallenge.git) both codes from HimariO.

2. **Feature extraction** [notebook](https://colab.research.google.com/drive/1IJt5ViL6tG205209EyGwGp435rIH_tzW): we extract features with different size of features to be used in `ERNIE-Vil` model.

3. **Fair Face** [noteebok](none):

4. **Models**
	4.1. MMF (Pytorch):
		4.1.1 VisualBERT (small)
		4.1.2 VisualBERT (COCO)
		4.1.3 VilBERT


	4.2 ERNIE-Vil (Paddle):
		4.2.1 ERNIE-Vil (small):
		4.2.1 ERNIE-Vil (large):

5. Ensemble:

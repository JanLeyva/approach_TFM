## 1. Feature extraction 

The feature extraction is did in two parts, one for the `mmf` models and the other for the `ernie-vil` models.

- In this [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y8RBKamXcWKSRxYTwj4vJpYl0RtXhNoy) is reproduced the features extraction using py-bottom-up-attention for `mmf` models. The output is a `.npy` objects, one for each image with 100 number of features extracted. Then the `.npy` objects are transform to `.mdb` to be used in `mmf` models.
 
- In this[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IJt5ViL6tG205209EyGwGp435rIH_tzW) is reproduced the features extraction using py-bottom-up-attention. Output features with different size (number of features) to be used in `ERNIE-Vil` model (`.tsv` format).


## 2. FairFace 

This model help us to extract features from the photos: Age, gender and race. The main feature that we are interested in is the reace, because a important part of hateful memes are racist.
To reproduce this part read the repository [README](https://github.com/JanLeyva/approach_TFM/tree/master/feature_extraction/FairFace_features) or follow the following [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/JanLeyva/approach_TFM/blob/master/feature_extraction/FairFace_features/FairFace_features.ipynb).
# We create the csv file to be used in FairFace
import pandas as pd
import os


def createCSV(path = "/Content/Hateful_memes/img", outpath = "/content/FairFace"):
	imagesList=os.listdir(path)
	csvImages = pd.DataFrame(imagesList)

	csvImg = []
	for i in range(len(csvImages)):
	  csvImg.append(os.path.join("/content", path, csvImages.at[i,0]))

	csvImg = pd.DataFrame(csvImg).rename(columns={0: "img_path"})
	csvImg.to_csv(os.path.join(outpath, "csvImg.csv"),index=False)


# We create the csv file to be used in FairFace
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description='scrip to create csv file to be used in FairFace. Output csvImg.csv')
parser.add_argument("--path", help="path where images are stored.")
parser.add_argument("--outpath", help = "output path for the .csv")
args = parser.parse_args()

imagesList=os.listdir(args.path)
csvImages = pd.DataFrame(imagesList)

csvImg = []
for i in range(len(csvImages)):
	csvImg.append(os.path.join("/content", args.path, csvImages.at[i,0]))

csvImg = pd.DataFrame(csvImg).rename(columns={0: "img_path"})
csvImg.to_csv(os.path.join(args.outpath, "csvImg.csv"),index=False)

import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='scrip to create json file to structure FairFace output. Output `fairface.json`.')
parser.add_argument("--csv", help=".csv output of FairFace.")
args = parser.parse_args()

fairface = pd.read_csv(args.csv)

results = {}
for i in range(len((fairface['index']).unique())):
    
    match = fairface[fairface['index'] == fairface['index'][i]]    
    boxes = match['bbox']
    race = match['race']
    race4 = match['race4']
    gender = match['gender']
    
    results[str(match['index'].iloc[0])] = {
            'face_boxes': [c for c in boxes],
            'face_race':  [c for c in race],
            'face_race4': [c for c in race4],
            'face_gender': [c for c in gender],
        }


with open('fairface.json', 'w') as fp:
    json.dump(results, fp,  indent=2)

import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='scrip to create json file to structure FairFace output. Output `fairface.json`.')
parser.add_argument("--csv", help=".csv output of FairFace.")
args = parser.parse_args()

df = pd.read_csv(args.csv)
dff = []
for i in range(len(df)):
  dff.append(df['face_name_align'][i].split('/')[1].split('_')[0])

df['index'] = dff

results = {}
for i in range(len((df['index']).unique())):
    
    match = df[df['index'] == df['index'][i]]    
    boxes = match['bbox']
    race = match['race']
    race4 = match['race4']
    gender = match['gender']
    
    results[str(match['index'].iloc[0])] = {
            'id': str(match['index'].iloc[0]),
            'face_boxes': [c for c in boxes],
            'face_race':  [c for c in race],
            'face_race4': [c for c in race4],
            'face_gender': [c for c in gender],
        }


with open('fairface.json', 'w') as fp:
    json.dump(results, fp,  indent=2)

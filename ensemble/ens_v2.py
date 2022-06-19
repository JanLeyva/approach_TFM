import pandas as pd
import numpy as np
import os

import glob
import shutil

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

from scipy.stats import rankdata

import math

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enspath", type=str, default="./data", help="Path to folder with all csvs")
    parser.add_argument("--enstype", type=str, default="loop", help="Type of ensembling to be performed - Current options: loop / sa")
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    parser.add_argument('--subdata', action='store_const', default=False, const=True)
    
    # Parse the arguments.
    args = parser.parse_args()

    return args





def merge(dfs):
    return sum([df.proba.values for df in dfs]) / len(dfs)


def get_mean_predict(out_path):

    csv_list = glob.glob(visualBERT_)
    csv_list += glob.glob(visualBERTCoco_)
    csv_list += glob.glob(vilBERT_)
    csv_list += glob.glob(ernie_vil_)



    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            print("Included in Simple Average: ", csv)
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values

    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    test_unseen_org = pd.read_json("test_unseen.jsonl", lines= True)
    test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0])

    fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], test_unseen_SA['proba'],  pos_label=1)
    print("AUROC:", metrics.auc(fpr, tpr)


    print(f"Found {len(csv_list)} csv eval result!")

    ensem_list = []
    All = False
    for csv_file in csv_list:
        # print(csv_file)
        if not All:
            yn = input(f"Include {csv_file} to ensemble? (y/n/all)")
        else:
            yn = 'y'
        yn = yn.strip().lower()
        if yn == 'all':
            All = True
        
        if yn == 'y' or All:
            ensem_list.append(csv_file)
            # dir_name = os.path.basename(os.path.dirname(csv_file))
            # shutil.copy(
            #     csv_file,
            #     os.path.join(
            #         gather_dir,
            #         f"{dir_name}_{os.path.basename(csv_file)}"
            #     )
            # )
    assert len(ensem_list) >= 2, f'You must select at least two file to ensemble, only {len(ensem_list)} is picked'
    
    base = pd.read_csv(ensem_list[0])
    print(len(ensem_list))
    ensem_list = [pd.read_csv(c) for c in ensem_list]
    base.proba = merge(ensem_list)



    # rasicm_idx = rasicm_det(
    #     os.path.join(root_dir, 'data/hateful_memes/test_unseen.jsonl'),
    #     os.path.join(root_dir, 'data/hateful_memes/box_annos.race.json'),
    #     os.path.join(root_dir, 'data/hateful_memes/img_clean'),
    # )
    # for i in rasicm_idx:
    #     base.at[int(base.index[base['id']==i].values), 'proba'] = 1.0

    base.to_csv(out_path, index=False)


if __name__ == "__main__":

    args = parse_args()
    
    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")

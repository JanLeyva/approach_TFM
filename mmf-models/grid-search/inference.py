import subprocess
from subprocess import call
import os
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import pandas as pd
import numpy as np
import shutil
from sklearn import metrics
from scipy.stats import rankdata
import math

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-path_pkt",         "--path_pkt",type=str , help=".pkt file with weigth of the model.")
    parser.add_argument("-output_path",      "--output_path",type=str , help="output prediction.")
    parser.add_argument("-dst_path",         "--dst_path",type=str ,  default = './', help="path to store results.")

    # Parse the arguments.
    args = parser.parse_args()
    return args

# Assign corresponding variables
# home = args.home

# def metric functions -----------------------------------------------------------------------------
def get_acc_and_best_threshold_from_roc_curve_(org_file, ens_file):
    fpr, tpr, thresholds = roc_curve(org_file['label'], ens_file['proba'],  pos_label=1)
    num_pos_class, num_neg_class = len(ens_file['label'] == 1), len(ens_file['label'] == 0)

    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]

    return np.amax(acc), best_threshold

def set_label(org_file, file):
    max_acc, best_threshold = get_acc_and_best_threshold_from_roc_curve_(org_file, file)
    results_tresh = []
    for i in range(len(file['label'])):
        if file['proba'][i] >= best_threshold:
            results_tresh.append(1)
        else:
            results_tresh.append(0)
    file['label'] = results_tresh
    return file


# original annotations -----------------------------------------------------------------------------
annotations_file='/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations'
test_seen=pd.read_json(os.path.join(annotations_file, "test_seen.jsonl"), lines = True)
test_unseen=pd.read_json(os.path.join(annotations_file, "test_unseen.jsonl"), lines = True)
dev_seen=pd.read_json(os.path.join(annotations_file, "dev_seen.jsonl"), lines = True)
dev_unseen=pd.read_json(os.path.join(annotations_file, "dev_unseen.jsonl"), lines = True)


dev_seen_order=dev_seen['id']
dev_unseen_order=dev_unseen['id']
test_seen_order=test_seen['id']
test_unseen_order=test_unseen['id']


def inference(path_pkt, output_path, dst_path):
    ls_model=os.listdir(weigth_model)
    for i, file in enumerate(ls_model):
        rc = call(f"/content/approach_TFM/mmf-models/grid-search/inference.sh {os.path.join(ls_model[i], 'best.ckpt')} {os.path.join(output_path, str(ls_model[i] + "_test_seen"))}{os.path.join(output_path, str(ls_model[i] + "_dev_seen"))} {os.path.join(output_path, str(ls_model[i] + "_dev_unseen"))}", shell=True)
    results=[]
    for j, folder in enumerate(os.listdir(output_path)):
        for k, path in enumerate(os.listdir(folder)):
            if "test_seen" in path:
                for l, path_ in enumerate(os.listdir(path)):
                    if path_ startswith("hateful_memes"):
                        file_=os.listdir(os.path.join(output_path, folder, "reports"))
                        output=os.path.join(output_path, folder, "reports", file_[0]) # rute to .csv pred file.
                        prediction=pd.read_csv(output)
                        pred=set_label(test_seen, pd.merge(test_seen_order, prediction))
                        results.append({
                            'set': "test_seen_"file_[0],
                            'auc': roc_auc_score(test_seen['label'], pred['proba'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, path+"_test_seen.csv"))
            elif "test_unseen" in path:
                for l, path_ in enumerate(os.listdir(path)):
                    if path_ startswith("hateful_memes"):
                        file_=os.listdir(os.path.join(output_path, folder, "reports"))
                        output=os.path.join(output_path, folder, "reports", file_[0]) # rute to .csv pred file.
                        prediction=pd.read_csv(output)
                        pred=set_label(test_unseen, pd.merge(test_unseen_order, prediction))
                        results.append({
                            'set': "test_unseen_"file_[0],
                            'auc': roc_auc_score(test_seen['label'], pred['proba'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, path+"_test_unseen.csv"))
            elif "dev_seen" in path:
                for l, path_ in enumerate(os.listdir(path)):
                    if path_ startswith("hateful_memes"):
                        file_=os.listdir(os.path.join(output_path, folder, "reports"))
                        output=os.path.join(output_path, folder, "reports", file_[0]) # rute to .csv pred file.
                        prediction=pd.read_csv(output)
                        pred=set_label(dev_seen, pd.merge(dev_seen_order, prediction))
                        results.append({
                            'set': "dev_seen_"+file_[0],
                            'auc': roc_auc_score(test_seen['label'], pred['proba'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, path+"_dev_seen.csv"))
            elif "dev_unseen" in path:
                for l, path_ in enumerate(os.listdir(path)):
                    if path_ startswith("hateful_memes"):
                        file_=os.listdir(os.path.join(output_path, folder, "reports"))
                        output=os.path.join(output_path, folder, "reports", file_[0]) # rute to .csv pred file.
                        prediction=pd.read_csv(output)
                        pred=set_label(dev_unseen, pd.merge(dev_unseen_order, prediction))
                        results.append({
                            'set': "dev_unseen_"+file_[0],
                            'auc': roc_auc_score(test_seen['label'], pred['proba'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, path+"_dev_unseen.csv"))
            else:
                print("Assertion: error file.")



    results=pd.DataFrame(results)
    results.to_csv("results.csv", index=False)




def main(path_pkt, output_path, dst_path):
    # ls of files/path in folder
    inference_ls(path_pkt, output_path, dst_path)



if __name__ == "__main__":
    args = parse_args()
    main(args.path_pkt, args.output_path, args.dst_path)
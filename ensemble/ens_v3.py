# This code is mainly borrowed from: https://github.com/Muennighoff/vilio
import pandas as pd
import numpy as np
import json
import os

import os
import glob
import shutil

import fire
import spacy
import pandas as pd
nlp = spacy.load("en_core_web_lg")

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

from scipy.stats import rankdata

import math

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enspath", type=str, default="./results/", help="Path to folder with all csvs")
    parser.add_argument("--fileFairface", type=str, default="fairface.json", help="Path to folder with all csvs")
    parser.add_argument("--enstype", type=str, default="loop", help="Type of ensembling to be performed - Current options: loop / sa")
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    parser.add_argument("--meme_anno_path", type=str, default="./annotations", help="path annotations")    
    parser.add_argument("--racism_rule", type=str, default="True", help="True/False racism rule")  
    parser.add_argument("--normalize", type=str, default="False", help="Normalize False by default")    
    parser.add_argument('--subdata', action='store_const', default=False, const=True)
    
    # Parse the arguments.
    args = parser.parse_args()

    return args

### FUNCTIONS IMPLEMENTING ENSEMBLE METHODS ###

### HELPERS ###

# Optimizing accuracy based on ROC AUC 
# Source: https://albertusk95.github.io/posts/2019/12/best-threshold-maximize-accuracy-from-roc-pr-curve/
# ACC = (TP + TN)/(TP + TN + FP + FN) = (TP + TN) / P + N   (= Correct ones / all)
# Senstivity / tpr = TP / P 
# Specificity / tnr = TN / N

def get_acc_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):

    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]

    return np.amax(acc), best_threshold

def set_acc(row, threshold):
    if row['proba'] >= threshold:
        val = 1
    else:
        val = 0
    return val


### AVERAGES ###

def simple_average(targets, example, weights=None, power=1, normalize=False):
    """
    targets: df with target values as columns
    example: output df example (e.g. including ID - make sure to adjust iloc below if target is not at 1)
    weights: per submission weights; default is equal weighting 
    power: optional for power averaging
    normalize: Whether to normalize targets btw 0 & 1
    """
    if weights is None:
        weights = len(targets.columns) * [1.0 / len(targets.columns)]
    else:
        weights = weights / np.sum(weights)

    preds = example.copy()
    preds.iloc[:,1] = np.zeros(len(preds))

    if normalize:
        targets = (targets - targets.min())/(targets.max()-targets.min())
    for i in range(len(targets.columns)):
        preds.iloc[:,1] = np.add(preds.iloc[:, 1], weights[i] * (targets.iloc[:, i].astype(float)**power))
    
    return preds




### SIMPLEX ###

### Similar to scipy optimize
# Taken & adapted from:
# https://github.com/chrisstroemel/Simple



### APPLYING THE HELPER FUNCTIONS ###

def sa_wrapper(data_path="./results/"):
    """
    Applies simple average.

    data_path: path to folder with  X * (dev_seen, test_seen & test_unseen) .csv files
    """
    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]

    # original test_unseen
    test_unseen_org = pd.read_json("test_unseen.jsonl", lines= True)


    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], pd.read_csv(data_path + csv).proba.values,  pos_label=1)
            print("Included in Simple Average: ", csv, metrics.auc(fpr, tpr))
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test_seen" in csv:
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values


    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0])

    # ---------------------------------------------------------------------------------- #
    # Optimal Threshold for Imbalanced Classification
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
    # ---------------------------------------------------------------------------------- #
    fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], test_unseen_SA['proba'],  pos_label=1)

    # G-mean
    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    # print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    # print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))
    # tresholder for the labeling
    results = []
    for i in range(len(test_unseen_SA['label'])):
        if test_unseen_SA['proba'][i] > gmeanOpt:
            results.append(1)
        else:
            results.append(0)

    test_unseen_SA['label'] = results

    # ---------------------------------------------------------------------------------- #
    # racism rule
    # ---------------------------------------------------------------------------------- #

    _racism_rule = args.racism_rule
    if _racism_rule:
        # detect keyword in text annotations
        meme_anno = {}
        anno_file = os.path.join(args.meme_anno_path, '/test_unseen.jsonl')
        with open(args.meme_anno_path, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno[data['id']] = data


        keyword = ['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla']
        keyword_tok = list(nlp(' '.join(keyword)))

        rasicm_sample_idx = []
        for i, (id, anno) in enumerate(meme_anno.items()):
            match = any([
            any([token.similarity(kwt) > 0.8 for kwt in keyword_tok])
                for token in nlp(anno['text'])
        ])
            if match:
                rasicm_sample_idx.append(id)

        # print("rasicm_sample_idx", rasicm_sample_idx)
        # detect race = 'black'
        fairface_ = pd.read_json(args.fileFairface)

        fairface_anno = []
        for i in range(len(fairface_['id'])):
            if('Black' or 'White' in fairface_.loc[i]['face_race4']):
                fairface_anno.append(str(fairface_.loc[i]['id']))

        # print("fairface_anno:", (fairface_anno))
        # check match in annotations text and fairface 'black'
        racism_results = []
        for i in range(len(fairface_anno)):
            if fairface_anno[i] in rasicm_sample_idx:
                racism_results.append(fairface_anno[i])

        print("match", racism_results)
        # change proba to 1 if racism detected
        for i in racism_results:
            if int(i) in test_unseen_SA['id'].values:
                test_unseen_SA.at[int(test_unseen_SA.index[test_unseen_SA['id']==int(i)].values), 'proba'] = 1.0
                test_unseen_SA.at[int(test_unseen_SA.index[test_unseen_SA['id']==int(i)].values), 'label'] = 1.0

    fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], test_unseen_SA['proba'],  pos_label=1)
    print("AUROC:", metrics.auc(fpr, tpr)) 
    print("Acc:", metrics.accuracy_score(test_unseen_org['label'], test_unseen_SA['label']))


    # Create output dir
    os.makedirs(os.path.join(data_path, args.exp), exist_ok=True)

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                dev_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_dev_seen_SA.csv"), index=False)   
            elif "test_unseen" in csv:
                test_unseen_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_unseen_SA.csv"), index=False)   
            elif "test_seen" in csv:
                test_SA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_seen_SA.csv"), index=False)



  
if __name__ == "__main__":

    args = parse_args()
    
    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")
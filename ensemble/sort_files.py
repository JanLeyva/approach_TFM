# This scrip will sort the results file in the same order as the original files dev_seen, test_seen, test_unseen
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, default="./annotations/", help="Path to folder with original annotations, jsnol")
    parser.add_argument("--path", type=str, default="./results/", help="Path to folder with all csv's results")
    parser.add_argument("--output_path", type=str, default="./output_path/", help="the path where the new files should be stored")
    parser.add_argument("--showmetrics", type=bool, default=False, help="if True show the metrics for each file")

    # Parse the arguments.
    args = parser.parse_args()

    return args

args = parse_args()

# help functions
def metrics(y_real, y_predict, model_name):
    
    print(model_name)
    print("-"*25)
    print("AUROC score {}".format(roc_auc_score(y_real['label'], y_predict['proba'])))
    print("-"*25)
    print("Accuracy {}".format(accuracy_score(y_real['label'], y_predict['label'])))
    print("-"*25)
    conf_matrix = confusion_matrix(y_real['label'], y_predict['label'])
    print("Confusion Matrix")
    print("-"*25)
    print("|", conf_matrix[0][0], "|", conf_matrix[0][1], "|")
    print("-"*25)
    print("|", conf_matrix[1][0], "|", conf_matrix[1][1], "|")
    print("-"*25)
    print("-"*50,"next...")

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
test_unseen_=pd.read_json(os.path.join(args.annotations_file, "test_unseen.jsonl"), lines = True)
test_seen_=pd.read_json(os.path.join(args.annotations_file, "test_seen.jsonl"), lines = True)
dev_seen=pd.read_json(os.path.join(args.annotations_file, "dev_seen.jsonl"), lines = True)


dev, test, test_unseen = [], [], []
dev_name, test_name, test_unseen_name = [], [], []
for csv in sorted(os.listdir(args.path)):
    # print(csv)
    if ".csv" in csv:
        if ("dev_seen" in csv) or ("val" in csv):
            dev.append(pd.read_csv(os.path.join(args.path, csv)))
            dev_name.append(csv)
        elif "test_unseen" in csv:
            test_unseen.append(pd.read_csv(os.path.join(args.path, csv)))
            test_unseen_name.append(csv)
        elif "test" in csv:
            test.append(pd.read_csv(os.path.join(args.path, csv)))
            test_name.append(csv)

dev_seen_order = dev_seen['id']
test_seen_order = test_seen_['id']
test_unseen_order = test_unseen_['id']
# Order files `to_csv` -----------------------------------------------------------------------------
for i, file in enumerate(dev):
    #print(dev_name[i])
    dev_to_csv = pd.merge(dev_seen_order, file)
    dev_to_csv = set_label(dev_seen, dev_to_csv)
    dev_to_csv.to_csv(os.path.join(args.output_path, (dev_name[i])))
    
for i, file in enumerate(test_unseen):
    #print(dev_name[i])
    test_to_csv = pd.merge(test_unseen_order, file)
    test_to_csv = set_label(test_unseen_, test_to_csv)
    test_to_csv.to_csv(os.path.join(args.output_path, (test_unseen_name[i])))
    
for i, file in enumerate(test):
    #print(dev_name[i])
    test_unseen_to_csv = pd.merge(test_seen_order, file)
    test_unseen_to_csv = set_label(test_seen_, test_unseen_to_csv)
    test_unseen_to_csv.to_csv(os.path.join(args.output_path, (test_name[i])))




if args.showmetrics:
	# read again the results now ordered
	dev, test, test_unseen = [], [], []
	dev_name, test_name, test_unseen_name = [], [], []
	for csv in sorted(os.listdir(args.output_path)):
	    # print(csv)
	    if ".csv" in csv:
	        if ("dev_seen" in csv) or ("val" in csv):
	            dev.append(pd.read_csv(os.path.join(args.output_path, csv)))
	            dev_name.append(csv)
	        elif "test_unseen" in csv:
	            test_unseen.append(pd.read_csv(os.path.join(args.output_path, csv)))
	            test_unseen_name.append(csv)
	        elif "test" in csv:
	            test.append(pd.read_csv(os.path.join(args.output_path, csv)))
	            test_name.append(csv)


	# show the results for each file

	# `dev_seen`
	print("#"*50)
	print("Show dev_seen metrics")
	print("#"*50)
	for i in range(3):
	    metrics(dev_seen, dev[i], dev_name[i])

	# `test_seen`
	print("#"*50)
	print("Show test_seen metrics")
	print("#"*50)
	for i in range(3):
	    metrics(test_seen_, test[i], test_name[i])
	  
	print("#"*50)
	print("Show test_unseen metrics")
	print("#"*50)
	for i in range(3):
	    metrics(test_unseen_, test_unseen[i], test_unseen_name[i])
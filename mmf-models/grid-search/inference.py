import subprocess
from subprocess import call
import os
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import shutil

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("-home", "--home",  help="home directory.")
    parser.add_argument("-conf_file",        "--conf_file",type=str, default='projects/hateful_memes/configs/visual_bert/from_coco.yaml', help="config file stored.")
    parser.add_argument("-weigth_model",     "--weigth_model",type=str , help=".pkt file with weigth of the model.")
    parser.add_argument("-annotations_file", "--annotations_file",type=str ,  default = 'hateful_memes/defaults/annotations', help="inference file to use [test/train/val]")
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
test_seen_=pd.read_json(os.path.join(args.annotations_file, "test_seen.jsonl"), lines = True)
test_unseen=pd.read_json(os.path.join(args.annotations_file, "test_unseen.jsonl"), lines = True)
dev_seen=pd.read_json(os.path.join(args.annotations_file, "dev_seen.jsonl"), lines = True)
dev_unseen=pd.read_json(os.path.join(args.annotations_file, "dev_unseen.jsonl"), lines = True)


dev_seen_order=dev_seen['id']
dev_unseen_order=dev_unseen['id']
test_seen_order=test_seen_['id']
test_unseen_order=test_unseen_['id']


# apply bash inference to 4 annotations

def inference_ls(inf_file, conf_file, output_path, dst_path):
    
    inf_file = ['test_seen.jsonl',
                'test_unseen.jsonl',
                'dev_seen.jsonl',
                'dev_unseen.jsonl']


    for file in inf_file:
        rc = call(f"/content/approach_TFM/mmf-models/grid-search/inference.sh {conf_file} {os.path.join(ls_model[i], 'best.ckpt')} {os.join.path(inf_file, file)} {output_path}", shell=True)


        results=[]
        for a, folder in enumerate(os.listdir(output_path)):
            if folder stratwith("hateful_memes"):
                file_=os.listdir(os.path.join(output_path, folder, "reports"))  # find the .csv pred file
                output_path=os.path.join(output_path, folder, "reports", file_) # rute to .csv pred file.
                prediction=pd.read_csv(output)
                if inf_file == "test_seen.jsonl":
                    pred=set_label(test_seen, pd.merge(test_seen_order, prediction))
                    results.append({
                        'set': 'test_seen',
                        "auc": roc_auc_score(test_seen['label'], pred['proba']),
                        "acc": accuracy_score(test_seen['label'], pred['label'])
                        })
                elif inf_file == "test_unseen.jsonl":
                    pred=set_label(test_unseen, pd.merge(test_unseen_order, prediction))
                    results.append({
                        'set': 'test_unseen',
                        "auc": roc_auc_score(test_unseen['label'], pred['proba']),
                        "acc": accuracy_score(test_unseen['label'], pred['label'])
                        })
                elif inf_file == "dev_seen.jsonl":
                    pred=set_label(dev_seen, pd.merge(dev_seen_order, prediction))
                    results.append({
                        'set': 'dev_seen',
                        "auc": roc_auc_score(dev_seen['label'], pred['proba']),
                        "acc": accuracy_score(dev_seen['label'], pred['label'])
                        })
                elif inf_file == "dev_unseen.jsonl":
                    pred=set_label(dev_unseen, pd.merge(dev_unseen_order, prediction))
                    results.append({
                        'set': 'dev_unseen',
                        "auc": roc_auc_score(dev_unseen['label'], pred['proba']),
                        "acc": accuracy_score(dev_unseen['label'], pred['label'])
                        })
                else:
                    print("Error no file found! in {}".format(output))

                shutil.move(output, dst_path)

    return results






def main():
    # ls of files/path in folder
    inference_ls(args.inf_file, args.conf_file, args.output_path, args.dst_path)


if __name__ == "__main__":
    args = parse_args()
    main()
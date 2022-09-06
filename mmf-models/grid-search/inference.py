import subprocess
from subprocess import call
import os
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import shutil

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-conf_file",        "--conf_file",type=str, default='projects/hateful_memes/configs/visual_bert/from_coco.yaml', help="config file stored.")
    parser.add_argument("-weigth_model",     "--weigth_model",type=str , help=".pkt file with weigth of the model.")
    parser.add_argument("-output_path",      "--output_path",type=str , help="output prediction.")
    # parser.add_argument("-annotations_file", "--annotations_file",type=str ,  default = 'hateful_memes/defaults/annotations', help="inference file to use [test/train/val]")
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


# apply bash inference to 4 annotations

def inference_ls(conf_file, weigth_model, output_path, dst_path):
    
    inf_file = ['test_seen.jsonl',
                'test_unseen.jsonl',
                'dev_seen.jsonl',
                'dev_unseen.jsonl']

    ls_model=os.listdir(weigth_model)
    for i, file in enumerate(ls_model):
        
        for e, ann in enumerate(inf_file):
            rc = call(f"/content/approach_TFM/mmf-models/grid-search/inference.sh {conf_file} {os.path.join(ls_model[i], 'best.ckpt')} {os.path.join('/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations', ann)} {output_path}", shell=True)


            results=[]
            for a, folder in enumerate(os.listdir(output_path)):
                if folder.startswith("hateful_memes"):
                    file_=os.listdir(os.path.join(output_path, folder, "reports"))  # find the .csv pred file
                    output=os.path.join(output_path, folder, "reports", file_[0]) # rute to .csv pred file.
                    prediction=pd.read_csv(output)
                    if inf_file == "test_seen.jsonl":
                        pred=set_label(test_seen, pd.merge(test_seen_order, prediction))
                        results.append({
                            'set': 'test_seen',
                            "auc": roc_auc_score(test_seen['label'], pred['proba']),
                            "acc": accuracy_score(test_seen['label'], pred['label'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, file+"_test_seen.csv"))
                    elif inf_file == "test_unseen.jsonl":
                        pred=set_label(test_unseen, pd.merge(test_unseen_order, prediction))
                        results.append({
                            'set': 'test_unseen',
                            "auc": roc_auc_score(test_unseen['label'], pred['proba']),
                            "acc": accuracy_score(test_unseen['label'], pred['label'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, file+"_test_unseen.csv"))
                    elif inf_file == "dev_seen.jsonl":
                        pred=set_label(dev_seen, pd.merge(dev_seen_order, prediction))
                        results.append({
                            'set': 'dev_seen',
                            "auc": roc_auc_score(dev_seen['label'], pred['proba']),
                            "acc": accuracy_score(dev_seen['label'], pred['label'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, file+"_dev_seen.csv"))
                    elif inf_file == "dev_unseen.jsonl":
                        pred=set_label(dev_unseen, pd.merge(dev_unseen_order, prediction))
                        results.append({
                            'set': 'dev_unseen',
                            "auc": roc_auc_score(dev_unseen['label'], pred['proba']),
                            "acc": accuracy_score(dev_unseen['label'], pred['label'])
                            })
                        shutil.copyfile(output, os.path.join(dst_path, file+"_dev_unseen.csv"))
                    else:
                        print("Error no file found! in {}".format(output))

                    # shutil.copyfile(output, os.path.join(dst_path, file_[0]))
                    results=pd.DataFrame(results)
                    results.to_csv(file+"_results.csv", index=False)
                    
    return results




def main(conf_file, weigth_model, output_path, dst_path):
    # ls of files/path in folder
    results=inference_ls(args.conf_file, args.weigth_model, args.output_path, args.dst_path)
    results.to_csv("results.csv", index=False)
    print(results)



if __name__ == "__main__":
    args = parse_args()
    main(args.conf_file, args.weigth_model, args.output_path, args.dst_path)
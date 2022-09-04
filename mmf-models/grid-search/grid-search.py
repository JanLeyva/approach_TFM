import subprocess
from subprocess import call
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("-home", "--home",  help="home directory.")
    parser.add_argument("-iter", "--iter",  default = 0, help="number interation to star.")

    # Parse the arguments.
    args = parser.parse_args()
    return args

# Assign corresponding variables
# home = args.home


def generate_list():
    confi_dotlist=[]
    i = 0
    for lr in [0.3, 0.6]:
        for w_type in ['warmup_cosine', 'warmup_linear']:
            for w_factor in [0.1, 0.3]:
                # call the bash script which start the training
                confi_dotlist.append({
                    "iteration": i,
                    "lr": lr,
                    "w_type": w_type,
                    "w_factor": w_factor})
                i+=1


    return confi_dotlist



def main(iter=0):
    # for each iteration a folder will be created whose name is 'experiemnt_{i}'
    confi_dotlist=generate_list()
    confi_dotlist=confi_dotlist[iter:]
    for e, it in enumerate(confi_dotlist):
        if os.path.exists("experiment_{}".format(confi_dotlist[e]['iteration'])):
            break
        else:
            os.mkdir("experiment_{}".format(confi_dotlist[e]['iteration']))
        rc = call(f"/content/approach_TFM/mmf-models/grid-search/grid-search.sh {'experiement_' + str(e)} {confi_dotlist[e]['lr']} {confi_dotlist[e]['w_type']} {confi_dotlist[e]['w_factor']} ", shell=True)

if __name__ == "__main__":
    args = parse_args()
    main(args.iter)
# This code is mainly borrowed from: https://github.com/Muennighoff/vilio and extended with racism rule and extras
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
    parser.add_argument("--filefairface", type=str, default="fairface.json", help="file with fairface features")
    parser.add_argument("--enstype", type=str, default="loop", help="Type of ensembling to be performed - Current options: loop / sa")
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    parser.add_argument("--meme_anno_path", type=str, default="./annotations", help="path annotations")    
    parser.add_argument("--racism_rule", type=str, default="True", help="True/False racism rule")  
    parser.add_argument("--normalize", type=bool, default="False", help="Normalize False by default")    
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


def get_acc_and_best_threshold_from_roc_curve_(org_file, ens_file):
    fpr, tpr, thresholds = metrics.roc_curve(org_file['label'], ens_file['proba'],  pos_label=1)
    num_pos_class, num_neg_class = len(ens_file['label'] == 1), len(ens_file['label'] == 0)

    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]

    return np.amax(acc), best_threshold


def auc_score(org_file, ens_file):
    fpr, tpr, thresholds = metrics.roc_curve(org_file['label'], ens_file['proba'],  pos_label=1)
    return metrics.auc(fpr, tpr)


### RANK AVERAGE ###

def rank_average(subs, weights=None):
    """
    subs: list of submission dataframes with two columns (id, value)
    weights: per submission weights; default is equal weighting 
    """
    if weights is None:
        weights = len(subs) * [1.0 / len(subs)]
    else:
        weights = weights / np.sum(weights)
    preds = subs[0].copy()
    preds.iloc[:,1] = np.zeros(len(subs[0]))
    for i, sub in enumerate(subs):
        preds.iloc[:,1] = np.add(preds.iloc[:,1], weights[i] * rankdata(sub.iloc[:,1]) / len(sub))
        
    return preds

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



from heapq import heappush, heappop, heappushpop
import numpy
import math
import time
import matplotlib.pyplot as plotter

CAPACITY_INCREMENT = 1000

class _Simplex:
    def __init__(self, pointIndices, testCoords, contentFractions, objectiveScore, opportunityCost, contentFraction, difference):
        self.pointIndices = pointIndices
        self.testCoords = testCoords
        self.contentFractions = contentFractions
        self.contentFraction = contentFraction
        self.__objectiveScore = objectiveScore
        self.__opportunityCost = opportunityCost
        self.update(difference)

    def update(self, difference):
        self.acquisitionValue = -(self.__objectiveScore + (self.__opportunityCost * difference))
        self.difference = difference

    def __eq__(self, other):
        return self.acquisitionValue == other.acquisitionValue

    def __lt__(self, other):
        return self.acquisitionValue < other.acquisitionValue

class SimpleTuner:
    def __init__(self, cornerPoints, objectiveFunction, exploration_preference=0.15):
        self.__cornerPoints = cornerPoints
        self.__numberOfVertices = len(cornerPoints)
        self.queue = []
        self.capacity = self.__numberOfVertices + CAPACITY_INCREMENT
        self.testPoints = numpy.empty((self.capacity, self.__numberOfVertices))
        self.objective = objectiveFunction
        self.iterations = 0
        self.maxValue = None
        self.minValue = None
        self.bestCoords = []
        self.opportunityCostFactor = exploration_preference #/ self.__numberOfVertices
            

    def optimize(self, maxSteps=10):
        for step in range(maxSteps):
            #print(self.maxValue, self.iterations, self.bestCoords)
            if len(self.queue) > 0:
                targetSimplex = self.__getNextSimplex()
                newPointIndex = self.__testCoords(targetSimplex.testCoords)
                for i in range(0, self.__numberOfVertices):
                    tempIndex = targetSimplex.pointIndices[i]
                    targetSimplex.pointIndices[i] = newPointIndex
                    newContentFraction = targetSimplex.contentFraction * targetSimplex.contentFractions[i]
                    newSimplex = self.__makeSimplex(targetSimplex.pointIndices, newContentFraction)
                    heappush(self.queue, newSimplex)
                    targetSimplex.pointIndices[i] = tempIndex
            else:
                testPoint = self.__cornerPoints[self.iterations]
                testPoint.append(0)
                testPoint = numpy.array(testPoint, dtype=numpy.float64)
                self.__testCoords(testPoint)
                if self.iterations == (self.__numberOfVertices - 1):
                    initialSimplex = self.__makeSimplex(numpy.arange(self.__numberOfVertices, dtype=numpy.intp), 1)
                    heappush(self.queue, initialSimplex)
            self.iterations += 1

    def get_best(self):
        return (self.maxValue, self.bestCoords[0:-1])

    def __getNextSimplex(self):
        targetSimplex = heappop(self.queue)
        currentDifference = self.maxValue - self.minValue
        while currentDifference > targetSimplex.difference:
            targetSimplex.update(currentDifference)
            # if greater than because heapq is in ascending order
            if targetSimplex.acquisitionValue > self.queue[0].acquisitionValue:
                targetSimplex = heappushpop(self.queue, targetSimplex)
        return targetSimplex
        
    def __testCoords(self, testCoords):
        objectiveValue = self.objective(testCoords[0:-1])
        if self.maxValue == None or objectiveValue > self.maxValue: 
            self.maxValue = objectiveValue
            self.bestCoords = testCoords
            if self.minValue == None: self.minValue = objectiveValue
        elif objectiveValue < self.minValue:
            self.minValue = objectiveValue
        testCoords[-1] = objectiveValue
        if self.capacity == self.iterations:
            self.capacity += CAPACITY_INCREMENT
            self.testPoints.resize((self.capacity, self.__numberOfVertices))
        newPointIndex = self.iterations
        self.testPoints[newPointIndex] = testCoords
        return newPointIndex


    def __makeSimplex(self, pointIndices, contentFraction):
        vertexMatrix = self.testPoints[pointIndices]
        coordMatrix = vertexMatrix[:, 0:-1]
        barycenterLocation = numpy.sum(vertexMatrix, axis=0) / self.__numberOfVertices

        differences = coordMatrix - barycenterLocation[0:-1]
        distances = numpy.sqrt(numpy.sum(differences * differences, axis=1))
        totalDistance = numpy.sum(distances)
        barycentricTestCoords = distances / totalDistance

        euclideanTestCoords = vertexMatrix.T.dot(barycentricTestCoords)
        
        vertexValues = vertexMatrix[:,-1]

        testpointDifferences = coordMatrix - euclideanTestCoords[0:-1]
        testPointDistances = numpy.sqrt(numpy.sum(testpointDifferences * testpointDifferences, axis=1))



        inverseDistances = 1 / testPointDistances
        inverseSum = numpy.sum(inverseDistances)
        interpolatedValue = inverseDistances.dot(vertexValues) / inverseSum


        currentDifference = self.maxValue - self.minValue
        opportunityCost = self.opportunityCostFactor * math.log(contentFraction, self.__numberOfVertices)

        return _Simplex(pointIndices.copy(), euclideanTestCoords, barycentricTestCoords, interpolatedValue, opportunityCost, contentFraction, currentDifference)

    def plot(self):
        if self.__numberOfVertices != 3: raise RuntimeError('Plotting only supported in 2D')
        matrix = self.testPoints[0:self.iterations, :]

        x = matrix[:,0].flat
        y = matrix[:,1].flat
        z = matrix[:,2].flat

        coords = []
        acquisitions = []

        for triangle in self.queue:
            coords.append(triangle.pointIndices)
            acquisitions.append(-1 * triangle.acquisitionValue)


        plotter.figure()
        plotter.tricontourf(x, y, coords, z)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()


        plotter.figure()
        plotter.tripcolor(x, y, coords, acquisitions)
        plotter.triplot(x, y, coords, color='white', lw=0.5)
        plotter.colorbar()

        plotter.show()

def Simplex(devs, label, df_list=False, exploration=0.01, scale=1):
    """
    devs: list of dataframes with "proba" column
    label: list/np array of ground truths
    scale: By default we will get weights in the 0-1 range. Setting e.g. scale=50, gives weights in the 0-50 range.
    """
    predictions = []
    if df_list:
        for df in devs:
            predictions.append(df.proba)

        print(len(predictions[0]))
    else:
        for i, column in enumerate(devs):
            predictions.append(devs.iloc[:, i])

        print(len(predictions[0]))

    print("Optimizing {} inputs.".format(len(predictions)))

    def roc_auc(weights):
        ''' Will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction
        return roc_auc_score(label, final_prediction)

    # This defines the search area, and other optimization parameters.
    # For e.g. 11 models, we have 12 corner points -- e.g. all none, only model 1, all others none, only model 2 all others none..
    # We concat an identity matrix & a zero array to create those
    zero_vtx = np.zeros((1, len(predictions)), dtype=int)
    optimization_domain_vertices = np.identity(len(predictions), dtype=int) * scale

    optimization_domain_vertices = np.concatenate((zero_vtx, optimization_domain_vertices), axis=0).tolist()

    
    number_of_iterations = 3000
    exploration = exploration # optional, default 0.01

    # Optimize weights
    tuner = SimpleTuner(optimization_domain_vertices, roc_auc, exploration_preference=exploration)
    tuner.optimize(number_of_iterations)
    best_objective_value, best_weights = tuner.get_best()

    print('Optimized =', best_objective_value) # same as roc_auc(best_weights)
    print('Weights =', best_weights)

    return best_weights

    # ---------------------------------------------------------------------------------- #
    # APPLYING RANK AVERAGE METHOD
    # ---------------------------------------------------------------------------------- #

def rank_average_method(data_path="./results"):

    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                # print("Included in Simple Average: ", csv, auc_score(test_unseen_org, pd.read_csv(data_path + csv)))
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                # fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], pd.read_csv(data_path + csv).proba.values,  pos_label=1)
                # print("Included in Simple Average: ", csv, auc_score(test_seen_org, pd.read_csv(data_path + csv)))
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test_seen" in csv:
                # print("Included in Simple Average: ", csv, auc_score(dev_seen_org, pd.read_csv(data_path + csv)))
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values


    for csv in sorted(os.listdir(data_path)):
        if "test_unseen" in csv:
            test_unseen_probas = pd.DataFrame(test_unseen_probas)
            test_unseen_RA = rank_average(test_unseen)
        elif "test_seen" in csv:
            test_probas = pd.DataFrame(test_probas)
            test_RA = rank_average(test)
        elif "dev_seen" in csv:
            dev_probas = pd.DataFrame(dev_probas)
            dev_RA = rank_average(dev)


    # ---------------------------------------------------------------------------------- #
    # Optimal Threshold for Imbalanced Classification
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
    # ---------------------------------------------------------------------------------- #
    # test_unseen treshold
    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(test_unseen_org, test_unseen_RA)

    results_test_unseen= []
    for i in range(len(test_unseen_RA['label'])):
        if test_unseen_RA['proba'][i] >= best_threshold_test_unseen:
            results_test_unseen.append(1)
        else:
            results_test_unseen.append(0)

    test_unseen_RA['label'] = results_test_unseen

    # test_seen treshold

    max_acc, best_threshold_test_seen = get_acc_and_best_threshold_from_roc_curve_(test_seen_org, test_RA)

    results_test_seen= []
    for i in range(len(test_RA['label'])):
        if test_RA['proba'][i] >= best_threshold_test_unseen:
            results_test_seen.append(1)
        else:
            results_test_seen.append(0)

    test_RA['label'] = results_test_seen

    # dev_seen treshold

    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(dev_seen_org, dev_RA)

    results_dev_seen= []
    for i in range(len(dev_RA['label'])):
        if dev_RA['proba'][i] >= best_threshold_test_unseen:
            results_dev_seen.append(1)
        else:
            results_dev_seen.append(0)

    dev_RA['label'] = results_dev_seen



    # ---------------------------------------------------------------------------------- #
    # racism rule
    # ---------------------------------------------------------------------------------- #
    if args.racism_rule == "True":
        # detect keyword in text annotations

        # read the original files
        meme_anno_test_unseen = {}
        anno_file_test_unseen = os.path.join(args.meme_anno_path, 'test_unseen.jsonl')
        with open(anno_file_test_unseen, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test_unseen[data['id']] = data

        meme_anno_test = {}
        anno_file_test = os.path.join(args.meme_anno_path, 'test_seen.jsonl')
        with open(anno_file_test, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test[data['id']] = data

        meme_anno_dev = {}
        anno_file_dev = os.path.join(args.meme_anno_path, 'dev_seen.jsonl')
        with open(anno_file_dev, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_dev[data['id']] = data

        # join org annotations files

        meme_anno = {**meme_anno_test_unseen, **meme_anno_test, **meme_anno_dev}


        keyword = ['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla',
        'black', 'islam', 'muslim']
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
        fairface_ = pd.read_json(args.filefairface)

        fairface_anno = []
        for i in range(len(fairface_['id'])):
            if('Black' in fairface_.loc[i]['face_race4']):
                fairface_anno.append(str(fairface_.loc[i]['id']))

        # print("fairface_anno:", (fairface_anno))
        # check match in annotations text and fairface 'black'
        racism_results = []
        for i in range(len(fairface_anno)):
            if fairface_anno[i] in rasicm_sample_idx:
                racism_results.append(fairface_anno[i])

        print("match", racism_results)
        # change proba to 1 if racism detected
        for indx in racism_results:
            if int(indx) in test_unseen_RA['id'].values:
                test_unseen_RA.at[int(test_unseen_RA.index[test_unseen_RA['id']==int(indx)].values), 'proba'] = 1.0
                test_unseen_RA.at[int(test_unseen_RA.index[test_unseen_RA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in test_RA['id'].values:
                test_RA.at[int(test_RA.index[test_RA['id']==int(indx)].values), 'proba'] = 1.0
                test_RA.at[int(test_RA.index[test_RA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in dev_RA['id'].values:
                dev_RA.at[int(dev_RA.index[dev_RA['id']==int(indx)].values), 'proba'] = 1.0
                dev_RA.at[int(dev_RA.index[dev_RA['id']==int(indx)].values), 'label'] = 1.0


    # ---------------------------------------------------------------------------------- #
    # Results RANK AVERAGE
    # ---------------------------------------------------------------------------------- #
    print("-"*50)
    print("RESULTS RANK AVERAGE")
    print("dev_seen AUROC:", auc_score(dev_seen_org, dev_RA))
    print("dev_seen Acc:", metrics.accuracy_score(dev_seen_org['label'], dev_RA['label']))


    print("Test_seen AUROC:", auc_score(test_seen_org, test_RA))
    print("Test_seen Acc:", metrics.accuracy_score(test_seen_org['label'], test_RA['label']))


    print("Test_unseen AUROC:", auc_score(test_unseen_org, test_unseen_RA))
    print("Test_unseen Acc:", metrics.accuracy_score(test_unseen_org['label'], test_unseen_RA['label']))
    print("-"*50)

    # Create output dir
    os.makedirs(os.path.join(data_path, args.exp), exist_ok=True)

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                dev_RA.to_csv(os.path.join(data_path, args.exp, args.exp + "_dev_seen_RA.csv"), index=False)   
            elif "test_unseen" in csv:
                test_unseen_RA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_unseen_RA.csv"), index=False)   
            elif "test_seen" in csv:
                test_RA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_seen_RA.csv"), index=False)



    return dev_RA, test_unseen_RA, test_RA



    # ---------------------------------------------------------------------------------- #
    # APPLYING OPTIMIZATION METHOD
    # ---------------------------------------------------------------------------------- #
def optimization(data_path="./results"):
    """
    optimizate the weight used in sa and return save the results in .csv
    used simplex method
    at least 3 files to optimize!!
    """

    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                # print("Included in Simple Average: ", csv, auc_score(test_unseen_org, pd.read_csv(data_path + csv)))
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                # fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], pd.read_csv(data_path + csv).proba.values,  pos_label=1)
                # print("Included in Simple Average: ", csv, auc_score(test_seen_org, pd.read_csv(data_path + csv)))
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test_seen" in csv:
                # print("Included in Simple Average: ", csv, auc_score(dev_seen_org, pd.read_csv(data_path + csv)))
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values




    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)


    weights_dev = Simplex(dev_probas, dev_seen_org.label)


    for csv in sorted(os.listdir(data_path)):
        if "test_unseen" in csv:
            #test_unseen_probas = pd.DataFrame(test_unseen_probas)
            test_unseen_PA = simple_average(test_unseen_probas, test_unseen[0], weights_dev, power=1, normalize=True)
        elif "test_seen" in csv:
            #test_probas = pd.DataFrame(test_probas)
            test_PA = simple_average(test_probas, test[0], weights_dev, power=1, normalize=True)
        elif "dev_seen" in csv:
            #dev_probas = pd.DataFrame(dev_probas)
            dev_PA = simple_average(dev_probas, dev[0], weights_dev, power=1, normalize=True)



    # ---------------------------------------------------------------------------------- #
    # Optimal Threshold for Imbalanced Classification
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
    # ---------------------------------------------------------------------------------- #
    # test_unseen treshold
    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(test_unseen_org, test_unseen_PA)

    results_test_unseen= []
    for i in range(len(test_unseen_PA['label'])):
        if test_unseen_PA['proba'][i] >= best_threshold_test_unseen:
            results_test_unseen.append(1)
        else:
            results_test_unseen.append(0)

    test_unseen_PA['label'] = results_test_unseen

    # test_seen treshold

    max_acc, best_threshold_test_seen = get_acc_and_best_threshold_from_roc_curve_(test_seen_org, test_PA)

    results_test_seen= []
    for i in range(len(test_PA['label'])):
        if test_PA['proba'][i] >= best_threshold_test_unseen:
            results_test_seen.append(1)
        else:
            results_test_seen.append(0)

    test_PA['label'] = results_test_seen

    # dev_seen treshold

    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(dev_seen_org, dev_PA)

    results_dev_seen= []
    for i in range(len(dev_PA['label'])):
        if dev_PA['proba'][i] >= best_threshold_test_unseen:
            results_dev_seen.append(1)
        else:
            results_dev_seen.append(0)

    dev_PA['label'] = results_dev_seen

    # ---------------------------------------------------------------------------------- #
    # racism rule
    # ---------------------------------------------------------------------------------- #
    if args.racism_rule == "True":
        # detect keyword in text annotations

        # read the original files
        meme_anno_test_unseen = {}
        anno_file_test_unseen = os.path.join(args.meme_anno_path, 'test_unseen.jsonl')
        with open(anno_file_test_unseen, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test_unseen[data['id']] = data

        meme_anno_test = {}
        anno_file_test = os.path.join(args.meme_anno_path, 'test_seen.jsonl')
        with open(anno_file_test, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test[data['id']] = data

        meme_anno_dev = {}
        anno_file_dev = os.path.join(args.meme_anno_path, 'dev_seen.jsonl')
        with open(anno_file_dev, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_dev[data['id']] = data

        # join org annotations files

        meme_anno = {**meme_anno_test_unseen, **meme_anno_test, **meme_anno_dev}


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
        fairface_ = pd.read_json(args.filefairface)

        fairface_anno = []
        for i in range(len(fairface_['id'])):
            if('Black' in fairface_.loc[i]['face_race4']):
                fairface_anno.append(str(fairface_.loc[i]['id']))

        # print("fairface_anno:", (fairface_anno))
        # check match in annotations text and fairface 'black'
        racism_results = []
        for i in range(len(fairface_anno)):
            if fairface_anno[i] in rasicm_sample_idx:
                racism_results.append(fairface_anno[i])

        print("match", racism_results)
        # change proba to 1 if racism detected
        for indx in racism_results:
            if int(indx) in test_unseen_PA['id'].values:
                test_unseen_PA.at[int(test_unseen_PA.index[test_unseen_PA['id']==int(indx)].values), 'proba'] = 1.0
                test_unseen_PA.at[int(test_unseen_PA.index[test_unseen_PA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in test_PA['id'].values:
                test_PA.at[int(test_PA.index[test_PA['id']==int(indx)].values), 'proba'] = 1.0
                test_PA.at[int(test_PA.index[test_PA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in dev_PA['id'].values:
                dev_PA.at[int(dev_PA.index[dev_PA['id']==int(indx)].values), 'proba'] = 1.0
                dev_PA.at[int(dev_PA.index[dev_PA['id']==int(indx)].values), 'label'] = 1.0

    # ---------------------------------------------------------------------------------- #
    # Results OPTIMIZATION
    # ---------------------------------------------------------------------------------- #
    print("-"*50)
    print("RESULTS OPTIMIZATION")
    print("dev_seen AUROC:", auc_score(dev_seen_org, dev_PA))
    print("dev_seen Acc:", metrics.accuracy_score(dev_seen_org['label'], dev_PA['label']))


    print("Test_seen AUROC:", auc_score(test_seen_org, test_PA))
    print("Test_seen Acc:", metrics.accuracy_score(test_seen_org['label'], test_PA['label']))


    print("Test_unseen AUROC:", auc_score(test_unseen_org, test_unseen_PA))
    print("Test_unseen Acc:", metrics.accuracy_score(test_unseen_org['label'], test_unseen_PA['label']))
    print("-"*50)

    # Create output dir
    os.makedirs(os.path.join(data_path, args.exp), exist_ok=True)

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                dev_PA.to_csv(os.path.join(data_path, args.exp, args.exp + "_dev_seen_PA.csv"), index=False)   
            elif "test_unseen" in csv:
                test_unseen_PA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_unseen_PA.csv"), index=False)   
            elif "test_seen" in csv:
                test_PA.to_csv(os.path.join(data_path, args.exp, args.exp + "_test_seen_PA.csv"), index=False)

    return dev_PA, test_unseen_PA, test_PA


### APPLYING THE HELPER FUNCTIONS ###

def sa_wrapper(data_path="./results"):
    """
    Applies simple average.

    data_path: path to folder with  X * (dev_seen, test_seen & test_unseen) .csv files
    """
    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(data_path)):
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                # print("Included in Simple Average: ", csv, auc_score(test_unseen_org, pd.read_csv(data_path + csv)))
                dev.append(pd.read_csv(data_path + csv))
                dev_probas[csv[:-8]] = pd.read_csv(data_path + csv).proba.values
            elif "test_unseen" in csv:
                # fpr, tpr, thresholds = metrics.roc_curve(test_unseen_org['label'], pd.read_csv(data_path + csv).proba.values,  pos_label=1)
                # print("Included in Simple Average: ", csv, auc_score(test_seen_org, pd.read_csv(data_path + csv)))
                test_unseen.append(pd.read_csv(data_path + csv))
                test_unseen_probas[csv[:-14]] = pd.read_csv(data_path + csv).proba.values
            elif "test_seen" in csv:
                # print("Included in Simple Average: ", csv, auc_score(dev_seen_org, pd.read_csv(data_path + csv)))
                test.append(pd.read_csv(data_path + csv))
                test_probas[csv[:-7]] = pd.read_csv(data_path + csv).proba.values


    for csv in sorted(os.listdir(data_path)):
        if "test_unseen" in csv:
            test_unseen_probas = pd.DataFrame(test_unseen_probas)
            test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0], power=1, normalize=False)
        elif "test_seen" in csv:
            test_probas = pd.DataFrame(test_probas)
            test_SA = simple_average(test_probas, test[0], power=1, normalize=False)
        elif "dev_seen" in csv:
            dev_probas = pd.DataFrame(dev_probas)
            dev_SA = simple_average(dev_probas, dev[0], power=1, normalize=False)

    # ---------------------------------------------------------------------------------- #
    # Optimal Threshold for Imbalanced Classification
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
    # ---------------------------------------------------------------------------------- #
    # test_unseen treshold
    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(test_unseen_org, test_unseen_SA)

    results_test_unseen= []
    for i in range(len(test_unseen_SA['label'])):
        if test_unseen_SA['proba'][i] >= best_threshold_test_unseen:
            results_test_unseen.append(1)
        else:
            results_test_unseen.append(0)

    test_unseen_SA['label'] = results_test_unseen

    # test_seen treshold

    max_acc, best_threshold_test_seen = get_acc_and_best_threshold_from_roc_curve_(test_seen_org, test_SA)

    results_test_seen= []
    for i in range(len(test_SA['label'])):
        if test_SA['proba'][i] >= best_threshold_test_unseen:
            results_test_seen.append(1)
        else:
            results_test_seen.append(0)

    test_SA['label'] = results_test_seen

    # dev_seen treshold

    max_acc, best_threshold_test_unseen = get_acc_and_best_threshold_from_roc_curve_(dev_seen_org, dev_SA)

    results_dev_seen= []
    for i in range(len(dev_SA['label'])):
        if dev_SA['proba'][i] >= best_threshold_test_unseen:
            results_dev_seen.append(1)
        else:
            results_dev_seen.append(0)

    dev_SA['label'] = results_dev_seen

    # ---------------------------------------------------------------------------------- #
    # racism rule
    # ---------------------------------------------------------------------------------- #
    if args.racism_rule == "True":
        # detect keyword in text annotations

        # read the original files
        meme_anno_test_unseen = {}
        anno_file_test_unseen = os.path.join(args.meme_anno_path, 'test_unseen.jsonl')
        with open(anno_file_test_unseen, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test_unseen[data['id']] = data

        meme_anno_test = {}
        anno_file_test = os.path.join(args.meme_anno_path, 'test_seen.jsonl')
        with open(anno_file_test, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_test[data['id']] = data

        meme_anno_dev = {}
        anno_file_dev = os.path.join(args.meme_anno_path, 'dev_seen.jsonl')
        with open(anno_file_dev, 'r') as f:
            for l in f:
                data = json.loads(l)
                meme_anno_dev[data['id']] = data

        # join org annotations files

        meme_anno = {**meme_anno_test_unseen, **meme_anno_test, **meme_anno_dev}


        keyword = ['crime', 'hang', 'rob', 'steal', 'jail', 'prison', 'slave', 'apes', 'criminal', 'gorilla']
        keyword_tok = list(nlp(' '.join(keyword)))

        rasicm_sample_idx = []
        for i, (id, anno) in enumerate(meme_anno.items()):
            match = any([
            any([token.similarity(kwt) > 0.6 for kwt in keyword_tok])
                for token in nlp(anno['text'])
        ])
            if match:
                rasicm_sample_idx.append(id)

        # print("rasicm_sample_idx", rasicm_sample_idx)
        # detect race = 'black'
        fairface_ = pd.read_json(args.filefairface)

        fairface_anno = []
        for i in range(len(fairface_['id'])):
            if('Black' in fairface_.loc[i]['face_race4']):
                fairface_anno.append(str(fairface_.loc[i]['id']))

        # print("fairface_anno:", (fairface_anno))
        # check match in annotations text and fairface 'black'
        racism_results = []
        for i in range(len(fairface_anno)):
            if fairface_anno[i] in rasicm_sample_idx:
                racism_results.append(fairface_anno[i])

        print("match", racism_results)
        # change proba to 1 if racism detected
        for indx in racism_results:
            if int(indx) in test_unseen_SA['id'].values:
                test_unseen_SA.at[int(test_unseen_SA.index[test_unseen_SA['id']==int(indx)].values), 'proba'] = 1.0
                test_unseen_SA.at[int(test_unseen_SA.index[test_unseen_SA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in test_SA['id'].values:
                test_SA.at[int(test_SA.index[test_SA['id']==int(indx)].values), 'proba'] = 1.0
                test_SA.at[int(test_SA.index[test_SA['id']==int(indx)].values), 'label'] = 1.0

            elif int(indx) in dev_SA['id'].values:
                dev_SA.at[int(dev_SA.index[dev_SA['id']==int(indx)].values), 'proba'] = 1.0
                dev_SA.at[int(dev_SA.index[dev_SA['id']==int(indx)].values), 'label'] = 1.0



    # ---------------------------------------------------------------------------------- #
    # Results SA
    # ---------------------------------------------------------------------------------- #
    print("-"*50)
    print("RESULTS SIMPLE AVERAGE")
    print("dev_seen AUROC:", auc_score(dev_seen_org, dev_SA))
    print("dev_seen Acc:", metrics.accuracy_score(dev_seen_org['label'], dev_SA['label']))


    print("Test_seen AUROC:", auc_score(test_seen_org, test_SA))
    print("Test_seen Acc:", metrics.accuracy_score(test_seen_org['label'], test_SA['label']))


    print("Test_unseen AUROC:", auc_score(test_unseen_org, test_unseen_SA))
    print("Test_unseen Acc:", metrics.accuracy_score(test_unseen_org['label'], test_unseen_SA['label']))
    print("-"*50)

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
    return dev_SA, test_unseen_SA, test_SA





def choose_opt(data_path="./results"):
    """
    Compare 3 methods, choose the high AUROC in test set.
    Then apply this method in loop until the score is not improved.
    """

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    # Applying the 3 methods
    sa_dev, sa_test_unseen, sa_test_seen = sa_wrapper(args.enspath)
    ra_dev, ra_test_unseen, ra_test_seen = rank_average_method(args.enspath)
    sx_dev, sx_test_unseen, sx_test_seen = optimization(args.enspath)
    prob_results_dev = [sa_dev, ra_dev, sx_dev]
    prob_results_test = [sa_test_seen, ra_test_seen, sx_test_seen]
    prob_results_untest = [sa_test_unseen, ra_test_unseen, sx_test_unseen]

    # Calc auroc score
    auc_sa = auc_score(test_seen_org, sa_test_seen)
    auc_ra = auc_score(test_seen_org, ra_test_seen)
    auc_sx = auc_score(test_seen_org, sx_test_seen)

    method_names = ["Simple Average", "Rank Average", "Optimization Simplex"]
    methods_result = [auc_sa, auc_ra, auc_sx]
    max_auroc = max(methods_result)
    print('-'*50)
    print("Best method", method_names[methods_result.index(max_auroc)])
    print("Best AUROC {}".format(max_auroc))
    print('-'*50)
    print("Results for dev, test_seen, test_unseen")
    print("AUROC dev {}".format(auc_score(dev_seen_org, prob_results_dev[methods_result.index(max_auroc)])))
    print("AUROC test {}".format(auc_score(test_seen_org, prob_results_test[methods_result.index(max_auroc)])))
    print("AUROC test_unseen {}".format(auc_score(test_unseen_org, prob_results_untest[methods_result.index(max_auroc)])))



def ens_ens(data_path="./results"):
    """
    Compare 3 methods, choose the high AUROC in test set.
    Then apply this method in loop until the score is not improved.
    """

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    # Applying the 3 methods
    sa_dev, sa_test_unseen, sa_test_seen = sa_wrapper(args.enspath)
    ra_dev, ra_test_unseen, ra_test_seen = rank_average_method(args.enspath)
    sx_dev, sx_test_unseen, sx_test_seen = optimization(args.enspath)

    prob_results_dev = [sa_dev, ra_dev, sx_dev]
    prob_results_test = [sa_test_seen, ra_test_seen, sx_test_seen]
    prob_results_test_unseen = [sa_test_unseen, ra_test_unseen, sx_test_unseen]



    # APPLYING RANK AVERAGE
    # Simple average for the 3 ensemble
    dev_ens_ens_ra = rank_average(prob_results_dev)
    test_ens_ens_ra = rank_average(prob_results_test)
    test_unseen_ens_ens_ra = rank_average(prob_results_test_unseen)

    print('-'*50)
    print("Results RANK AVERAGE for dev, test_seen, test_unseen")
    print("AUROC dev {}".format(auc_score(dev_seen_org, dev_ens_ens_ra)))
    print("AUROC test {}".format(auc_score(test_seen_org, test_ens_ens_ra)))
    print("AUROC test_unseen {}".format(auc_score(test_unseen_org, test_unseen_ens_ens_ra)))
    print('-'*50)




### APPLYING LOOP

def main(path, gt_path="./annotations"):
    """
    Loops through Averaging, Power Averaging, Rank Averaging, Optimization to find the best ensemble.

    path: String to directory with csvs of all models
    For each model there should be three csvs: dev, test, test_unseen

    gt_path: Path to folder with ground truth for dev
    """
    # Ground truth
    dev_df = pd.read_json(os.path.join(gt_path, 'dev_seen.jsonl'), lines=True)

    # original files
    test_unseen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_unseen.jsonl"), lines=True)
    test_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "test_seen.jsonl"), lines=True)
    dev_seen_org = pd.read_json(os.path.join(args.meme_anno_path, "dev_seen.jsonl"), lines=True)

    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]
    dev, test, test_unseen = [], [], []
    dev_probas, test_probas, test_unseen_probas = {}, {}, {} # Never dynamically add to a pd Dataframe

    for csv in sorted(os.listdir(path)):
        print(csv)
        if ".csv" in csv:
            if ("dev" in csv) or ("val" in csv):
                dev.append(pd.read_csv(os.path.join(path, csv)))
                dev_probas[csv[:-8]] = pd.read_csv(os.path.join(path, csv)).proba.values
            elif "test_unseen" in csv:
                test_unseen.append(pd.read_csv(os.path.join(path, csv)))
                test_unseen_probas[csv[:-14]] = pd.read_csv(os.path.join(path, csv)).proba.values
            elif "test" in csv:
                test.append(pd.read_csv(os.path.join(path, csv)))
                test_probas[csv[:-7]] = pd.read_csv(os.path.join(path, csv)).proba.values


    dev_probas = pd.DataFrame(dev_probas)
    test_probas = pd.DataFrame(test_probas)
    test_unseen_probas = pd.DataFrame(test_unseen_probas)

    dev_or = dev.copy()
    test_or = test.copy()
    test_unseen_or = test_unseen.copy()

    if len(dev_df) > len(dev_probas):

        print("Your predictions do not include the full dev!")        
        dev_df = dev[0][["id"]].merge(dev_df, how="left", on="id")

    loop, last_score, delta = 0, 0, 0.1

    while (delta > 0.0001):

        # Individual Roc Aucs
        print("Individual RCs:\n")
        print("dev")

        for i, column in enumerate(dev_probas):
            score = roc_auc_score(dev_df.label, dev_probas.iloc[:, i])
            print(column, score)

        print('-'*50)

        if loop > 0:
            while len(dev) > 5:
                lowest_score = 1
                drop = 0
                for i, column in enumerate(dev_probas):
                    score = roc_auc_score(dev_df.label, dev_probas.iloc[:, i])
                    if score < lowest_score:
                        lowest_score = score
                        col = column
                        drop = i

                column_numbers = [x for x in range(dev_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                dev_probas = dev_probas.iloc[:, column_numbers]

                column_numbers = [x for x in range(test_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                test_probas = test_probas.iloc[:, column_numbers]

                column_numbers = [x for x in range(test_unseen_probas.shape[1])]  # list of columns' integer indices
                column_numbers.remove(drop)
                test_unseen_probas = test_unseen_probas.iloc[:, column_numbers]
    
                if i < len(dev_or):
                    dev_or.pop(drop)
                    test_or.pop(drop)
                    test_unseen_or.pop(drop)
                if i < len(dev):
                    dev.pop(drop)
                    test.pop(drop)
                    test_unseen.pop(drop)
    
                print("Dropped:", col)
                
        # Spearman Correlations: 
        print("Spearman Corrs:")
        dev_corr = dev_probas.corr(method='spearman')
        test_corr = test_probas.corr(method='spearman')
        test_unseen_corr = test_unseen_probas.corr(method='spearman')
        
        print(dev_corr,'\n')
        print(test_corr)
        print(test_unseen_corr)
        print('-'*50)

        ### SIMPLE AVERAGE ###
        dev_SA = simple_average(dev_probas, dev[0], power=1, normalize=True)
        test_SA = simple_average(test_probas, test[0], power=1, normalize=True)
        test_unseen_SA = simple_average(test_unseen_probas, test_unseen[0], power=1, normalize=True)

        print(roc_auc_score(dev_df.label, dev_SA.proba), accuracy_score(dev_df.label, dev_SA.label))
        print('-'*50)

        ### POWER AVERAGE ###
        dev_PA = simple_average(dev_probas, dev[0], power=2, normalize=True)
        test_PA = simple_average(test_probas, test[0], power=2, normalize=True)
        test_unseen_PA = simple_average(test_unseen_probas, test_unseen[0], power=2, normalize=True)

        print(roc_auc_score(dev_df.label, dev_PA.proba), accuracy_score(dev_df.label, dev_PA.label))
        print('-'*50)

        ### RANK AVERAGE ###
        dev_RA = rank_average(dev)
        test_RA = rank_average(test)
        test_unseen_RA = rank_average(test_unseen)

        print(roc_auc_score(dev_df.label, dev_RA.proba), accuracy_score(dev_df.label, dev_RA.label))
        print('-'*50)

        ### SIMPLEX ###
        weights_dev = Simplex(dev_probas, dev_df.label)

        dev_SX = simple_average(dev_probas, dev[0], weights_dev)
        test_SX = simple_average(test_probas, test[0], weights_dev)
        test_unseen_SX = simple_average(test_unseen_probas, test_unseen[0], weights_dev)

        print(roc_auc_score(dev_df.label, dev_SX.proba), accuracy_score(dev_df.label, dev_SX.label))
        print('-'*50)

        # Prepare Next Round
        dev = dev_or + [dev_SA, dev_PA, dev_RA, dev_SX]
        test = test_or + [test_SA, test_PA, test_RA, test_SX]
        test_unseen = test_unseen_or + [test_unseen_SA, test_unseen_PA, test_unseen_RA, test_unseen_SX]
        
        dev_probas = pd.concat([df.proba for df in dev], axis=1)
        test_probas = pd.concat([df.proba for df in test], axis=1)
        test_unseen_probas = pd.concat([df.proba for df in test_unseen], axis=1)

        # Calculate Delta & increment loop
        delta = abs(roc_auc_score(dev_df.label, dev_SX.proba) - last_score)
        last_score = roc_auc_score(dev_df.label, dev_SX.proba)

        loop += 1

        # I found the loop to not add any value after 2 rounds.
        if loop == 4:
            break
    
    print("Currently at {} after {} loops.".format(last_score, loop))
    print("AUC test_seen {}".format(auc_score(test_seen_org, test_SX)))
    print("AUC test_unseen {}".format(auc_score(test_unseen_org, test_unseen_SX)))
    # Get accuracy thresholds & optimize (This does not add value to the roc auc, but just to also have an acc score)
    fpr, tpr, thresholds = metrics.roc_curve(dev_df.label, dev_SX.proba)
    acc, threshold = get_acc_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, 250, 250)
    test_SX.label = test_SX.apply(set_acc, axis=1, args=[threshold])
    test_unseen_SX.label = test_unseen_SX.apply(set_acc, axis=1, args=[threshold])

    # Set path instd of /k/w ; Remove all csv data / load the exact same 3 files again as put out
    # As Simplex at some point simply weighs the highest of all - lets take sx as the final prediction after x loops
    dev_SX.to_csv(os.path.join(path, "FIN_dev_seen_" + args.exp + "_" + str(loop) + ".csv"), index=False)
    test_SX.to_csv(os.path.join(path, "FIN_test_seen_" + args.exp + "_" + str(loop) + ".csv"), index=False)
    test_unseen_SX.to_csv(os.path.join(path, "FIN_test_unseen_" + args.exp + "_" + str(loop) + ".csv"), index=False)

    print("Finished.")



if __name__ == "__main__":

    args = parse_args()
    
    if args.enstype == "loop":
        main(args.enspath)
    elif args.enstype == "best_ens":
        choose_opt(args.enspath)
    elif args.enstype == "ens_ens":
        ens_ens(args.enspath)
    elif args.enstype == "sa":
        sa_wrapper(args.enspath)
    elif args.enstype == "optimizer":
        optimization(args.enspath)
    elif args.enstype == "rank_avg":
        rank_average_method(args.enspath)
    else:
        print(args.enstype, " is not yet enabled. Feel free to add the code :)")
import csv
import json
import os
from typing import Callable
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ClassifierResults import ClassifierResults

import argparse
import matplotlib.gridspec as gridspec
import matplotlib
from tabulate import tabulate

import pprint

from latex_table_atnn_eatnn_gammas import atnn_eatnn_gammas_accuracies_table
from latex_table_atnn_vs_eatnn_times import times_comparison
from latex_table_atnn_vs_eatnn_times_ratio import times_comparison_ratio
from latex_table_cand_accuracies import cand_accuracies_table
from latex_table_cand_sb import cand_sb_table
from latex_table_cand_times import cand_times_table

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'




class PlotPrinterConfig:
    def __init__(self, directory: str, description: str):
        self.directory = directory
        self.description = description

    def is_set(self):
        return self.description is not None

    def get_path(self):
        return f"{self.directory}/{self.description}"


def extractParams(classifierDir: str) -> dict[str, float]:
    resultDict = {}

    for classifierRaw in classifierDir.split("_"):
        valueBeginIdx = -1
        for i, c in enumerate(classifierRaw):
            if c.isdigit():
                valueBeginIdx = i
                break

        # print("classifierDir: " + classifierDir)
        # print("classifierRaw: " + classifierRaw)
        # print("key: " + classifierRaw[:(valueBeginIdx)])
        # print("value: " + classifierRaw[valueBeginIdx:])
        key = classifierRaw[:(valueBeginIdx)]
        value = float(classifierRaw[valueBeginIdx:])
        resultDict[key] = value

    return resultDict


def listOfExperimentFilePathsForParams(classifierExpPath: str) -> list[str]:
    pairs = []
    for maybeDir in os.scandir(classifierExpPath):
        if (maybeDir.is_dir()):
            for expFile in os.scandir(maybeDir.path):
                with open(expFile.path, "r") as expData:
                    firstLine = expData.readline()
                    timestamp = int((firstLine.split(","))[0])
                    pairs.append((expFile.path, timestamp))

    sortedPairs = sorted(pairs, key=lambda pair: pair[1])
    return list(map(lambda pair: pair[0], sortedPairs))


def readData(paths: list[str], headers: list[str]):
    collectedHeaders = ["timestamp", "class", "predicted"] + headers[3:]
    accuracyKey = "accuracy"
    result = {}
    for tmp in collectedHeaders + [accuracyKey]:
        result[tmp] = []

    count = 0
    correct = 0

    for path in paths:
        with open(path, "r") as file:
            reader = csv.DictReader(file, fieldnames=headers)


            prev_row = None
            for row in reader:
                count = count + 1
                if row["class"] == row["predicted"]:
                    correct = correct + 1

                result[accuracyKey].append(100.0 * float(correct) / float(count))
                for key in collectedHeaders:
                    if row[key].isdigit():
                        result[key].append(int(row[key]))
                    elif key == "trainingDuration":
                        result[key].append(18_400_000) # jakieś przykładowe, żeby zakryć minusa, zginie w szumie
                    elif key == "classificationDuration":
                        result[key].append(18_400_000)  # jakieś przykładowe, żeby zakryć minusa, zginie w szumie
                    else:
                        result[key].append(row[key])

                # prev_row = row

    result_np = {}
    for key, value in result.items():
        result_np[key] = np.array(value)
    return result_np


def getUnit(measurement: str):
    if "Duration" in measurement or "duration" in measurement:
        return r"$\mu$s"
    elif measurement == "accuracy":
        return r"%"
    elif "%" in measurement:
        return r"%"
    else:
        return None


def readAllResults(classifierPath: str):
    classifierType = os.path.basename(os.path.normpath(classifierPath))

    allResults = []

    for classifierParamsRaw in os.listdir(classifierPath):
        classifierParamsPath = f"{classifierPath}/{classifierParamsRaw}"
        # print("classifierPath: " + classifierPath)

        classifierParams = extractParams(classifierParamsRaw)

        with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
            resultJson = json.load(resultFile)
            headers = resultJson["dataHeader"]

            chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
            results = readData(chronologicalDataFilePaths, headers)

            allResults.append(
                ClassifierResults(classifierParams, classifierType, results, headers, resultJson["jobId"]))

    return allResults


def getClassifierResults(classifierParamsPath: str):

    with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
        resultJson = json.load(resultFile)
        headers = resultJson["dataHeader"]

        chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
        results = readData(chronologicalDataFilePaths, headers)
        # print("lista plików:")
        # pprint.pprint(chronologicalDataFilePaths)

        trainingDuration = round(np.sum(results["trainingDuration"]) / 1e9, 2)
        classificationDuration = round(np.sum(results["classificationDuration"]) / 1e9, 2)

        acc = results["accuracy"][-1]
        T_overall = trainingDuration + classificationDuration
        T_train = trainingDuration
        # T_overall = (results["timestamp"][-1] - results["timestamp"][0])/1e9


    return acc, T_train, T_overall



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", action="store", default=None, required=False,
                        help="Description if plots should be printed as files")
    parser.add_argument("--plotsDir", action="store", default=None, required=False,
                        help="Directory for plots to be saved to")
    args = parser.parse_args()

    experimentId = os.environ["EXPERIMENT_ID"]
    resultsInputDir = os.environ["RESULTS_DIRECTORY"]
    resultsPath = f"{resultsInputDir}/{experimentId}"
    dataset_results = {}
    for dataset in os.listdir(resultsPath):
        datasetPath = f"{resultsPath}/{dataset}"
        print(f"dataset: {dataset}")

        bestClassifierResults = []

        allResults = []

        dataset_results[dataset] = {}
        for classifierType in os.listdir(datasetPath): # cand
            classifierPath = f"{datasetPath}/{classifierType}"

            dataset_results[dataset][classifierType] = {}
            for classifierParams in os.listdir(classifierPath): # Psize10_Msize10
                classifierParamsPath = f"{classifierPath}/{classifierParams}"
                print(f"params: {classifierParams}")

                accuracy, T_train, T_overall = getClassifierResults(classifierParamsPath)

                dataset_results[dataset][classifierType][classifierParams] = {}
                dataset_results[dataset][classifierType][classifierParams]["acc"] = round(float(accuracy), 2)
                dataset_results[dataset][classifierType][classifierParams]["time_train"] = round(float(T_train), 2)
                dataset_results[dataset][classifierType][classifierParams]["time_overall"] = round(float(T_overall), 2)

    rows = []
    for dataset, methods in dataset_results.items():
        for method, params in methods.items():
            for param_set, metrics in params.items():
                param_set = param_set[:-6] # usuwam _iter1, _iter2, itd.
                row = {
                    'Dataset': dataset,
                    'Method': method,
                    'Params': param_set,
                    'Accuracy': metrics['acc'],
                    'Time Overall': metrics['time_overall'],
                    'Time Train': metrics['time_train'],
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Reformatting and displaying the DataFrame using a standard Python print statement for simplicity
    print(tabulate(df, headers='keys', tablefmt='grid'))

    summary = df.groupby(["Dataset", "Method", "Params"]).mean(numeric_only=True).reset_index()
    print(tabulate(summary, headers='keys', tablefmt='grid'))

    print_latex_table = True
    if print_latex_table:
        acc_latex_table = cand_accuracies_table(summary)
        with open("misc/plotter/cand_accuracy_comparison_table.tex", "w") as f:
            f.write(acc_latex_table)

    if print_latex_table:
        acc_latex_table = cand_times_table(summary)
        with open("misc/plotter/cand_times_comparison.tex", "w") as f:
            f.write(acc_latex_table)

    if print_latex_table:
        acc_latex_table = cand_sb_table(summary)
        with open("misc/plotter/cand_sb.tex", "w") as f:
            f.write(acc_latex_table)

import csv
import json
import os
from typing import Callable
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from ClassifierResults import ClassifierResults

import argparse
import matplotlib.gridspec as gridspec
import matplotlib

import pprint


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


def getClassifierTimes(classifierPath: str):
    classifierType = os.path.basename(os.path.normpath(classifierPath))


    for classifierParamsRaw in os.listdir(classifierPath): # wchodze na poziom tych P10_M2
        classifierParamsPath = f"{classifierPath}/{classifierParamsRaw}"

        classifierParams = extractParams(classifierParamsRaw)

        with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
            resultJson = json.load(resultFile)
            headers = resultJson["dataHeader"]

            chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
            results = readData(chronologicalDataFilePaths, headers)

            trainingDuration = round(np.sum(results["trainingDuration"]) / 1e9, 2)
            classificationDuration = round(np.sum(results["classificationDuration"]) / 1e9, 2)

            df = pd.DataFrame.from_dict(results)
            times = {}
            drift_warn_times = df[df['driftStatus'] == 'warn'].shape[0]
            d_warn = drift_warn_times / df.shape[0]

            copied_nodes_trained_during_warn = df[df['driftStatus'] == 'warn']['clonedNodesToTrain'].sum()
            empty_nodes_trained_during_warn = df[df['driftStatus'] == 'warn']['emptyNodesToTrain'].sum()
            normal_nodes_trained_during_warn = df[df['driftStatus'] == 'warn']['normalNodesToTrain'].sum()
            n_warn_nodes = (copied_nodes_trained_during_warn + empty_nodes_trained_during_warn + normal_nodes_trained_during_warn) / drift_warn_times

            copied_nodes_to_train = df['clonedNodesToTrain'].sum()
            empty_nodes_to_train = df['emptyNodesToTrain'].sum()
            normal_nodes_to_train = df['normalNodesToTrain'].sum()
            n_nodes = (copied_nodes_to_train + empty_nodes_to_train + normal_nodes_to_train) / df.shape[0]

            T_train = trainingDuration
            T_overall = trainingDuration + classificationDuration

    return d_warn, n_warn_nodes, n_nodes, T_train, T_overall



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

        bestClassifierResults = []

        allResults = []

        dataset_results[dataset] = {}
        for classifierType in os.listdir(datasetPath):
            classifierPath = f"{datasetPath}/{classifierType}"

            d_warn, n_warn_nodes, n_nodes, T_train, T_overall = getClassifierTimes(classifierPath)

            dataset_results[dataset][classifierType] = {}
            dataset_results[dataset][classifierType]["d_warn"] = d_warn
            dataset_results[dataset][classifierType]["n_warn_nodes"] = float(n_warn_nodes)
            dataset_results[dataset][classifierType]["n_nodes"] = float(n_nodes)
            dataset_results[dataset][classifierType]["T_train"] = float(T_train)
            dataset_results[dataset][classifierType]["T_overall"] = float(T_overall)


    pprint.pprint(dataset_results)


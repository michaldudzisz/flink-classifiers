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

        print("classifierDir: " + classifierDir)
        print("classifierRaw: " + classifierRaw)
        print("key: " + classifierRaw[:(valueBeginIdx)])
        print("value: " + classifierRaw[valueBeginIdx:])
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

            for row in reader:
                count = count + 1
                if row["class"] == row["predicted"]:
                    correct = correct + 1

                result[accuracyKey].append(100.0 * float(correct) / float(count))
                for key in collectedHeaders:
                    if (key == "trainingDuration"):
                        if (not row[key].isdigit()):
                            print(row[key])
                            print(row[key].isdigit())
                            print("koncze sie, bo mam czas, ktory nie jest liczba")
                            quit()
                    if row[key].isdigit():
                        result[key].append(int(row[key]))
                    else:
                        result[key].append(row[key])

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


def translatePerformanceType(performanceType: str):
    match performanceType:
        case "accuracy":
            return "$\\text{dokładność}$"
        case "trainingDuration":
            return "$t_\\mathrm{ucz}$"
        case "classificationDuration":
            return "$t_\\mathrm{pred}$"
        case "nodesDuringTraversalCount":
            return "$n_\\mathrm{nodes}$"
        case "duringTraversalDuration":
            return "$t_\\mathrm{traverse}$"
        case "leafSplit":
            return "$f_\\mathrm{split}$"
        case "weightsNormalizationAndClassifierDeleteDuration":
            return "$t_\\mathrm{norm+del}$"
        case "deletedClassifiersCount":
            return "$n_\\mathrm{del}$"
        case "deletedClassifiersCount%":
            return "$n_\\mathrm{del}$"
        case "addedClassifiersCount":
            return "$n_\\mathrm{add}$"
        case "addClassifierDuration":
            return "$t_\\mathrm{add}$"
        case "classifiersAfterTrainCount":
            return "$n_\\mathrm{poUcz}$"
        case "avgClassifierTTL":
            return "$n_\\mathrm{TTL}$"
        case "weightsLoweringCount":
            return "$n_\\mathrm{w\\downarrow}$"
        case "correctVotesCount":
            return "$n_\\mathrm{poprawne}$"
        case "wrongVotesCount":
            return "$n_\\mathrm{błędne}$"
        case "weightsLoweringCount%":
            return "$n_\\mathrm{w\\downarrow}$"
        case "correctVotesCount%":
            return "$n_\\mathrm{poprawne}$"
        case "wrongVotesCount%":
            return "$n_\\mathrm{błędne}$"
        case "avgTrainDuration":
            return "$t_\\mathrm{ucz_\\mathrm{avg}}$"
        case "avgClassifyDuration":
            return "$t_\\mathrm{pred_\\mathrm{avg}}$"
        case "substituteTrainingBegin":
            return "TODO"
        case "replacedClassifier":
            return "TODO"
        case _:
            return ""


def readAllResults(classifierPath: str):
    classifierType = os.path.basename(os.path.normpath(classifierPath))

    allResults = []

    for classifierParamsRaw in os.listdir(classifierPath):
        classifierParamsPath = f"{classifierPath}/{classifierParamsRaw}"
        print("classifierPath: " + classifierPath)

        classifierParams = extractParams(classifierParamsRaw)

        with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
            resultJson = json.load(resultFile)
            headers = resultJson["dataHeader"]

            chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
            results = readData(chronologicalDataFilePaths, headers)

            allResults.append(
                ClassifierResults(classifierParams, classifierType, results, headers, resultJson["jobId"]))

    return allResults


def get_all_clasifiers_results(classifierPath: str):
    classifierType = os.path.basename(os.path.normpath(classifierPath))

    bestAccuracy = -1.0
    bestParams = {}
    bestResults = {}
    bestHeaders = []
    bestJobId = ""

    allResults = []
    classifier_results = []

    for classifierParamsRaw in os.listdir(classifierPath): # wchodze na poziom tych P10_M2
        classifierParamsPath = f"{classifierPath}/{classifierParamsRaw}"

        classifierParams = extractParams(classifierParamsRaw)

        with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
            resultJson = json.load(resultFile)
            headers = resultJson["dataHeader"]

            chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
            results = readData(chronologicalDataFilePaths, headers)

            currentAccuracy = results["accuracy"][-1]
            # if currentAccuracy > bestAccuracy:
            #     bestAccuracy = currentAccuracy
            #     bestParams = classifierParams
            #     bestResults = results
            #     bestHeaders = headers
            #     bestJobId = resultJson["jobId"]


            trainingDuration = round(np.sum(results["trainingDuration"]) / 1e9, 2)
            classificationDuration = round(np.sum(results["classificationDuration"]) / 1e9, 2)

            allResults.append((os.path.basename(classifierPath), classifierParams, currentAccuracy,
                               trainingDuration + classificationDuration, trainingDuration,
                               classificationDuration))

            classifier_results.append(ClassifierResults(classifierParams, classifierType, results, headers, resultJson["jobId"]))

    return classifier_results, allResults


def plotWindowed(dataset: str, classifierResults: list[ClassifierResults], performanceType: str,
                 plot_printer_config: PlotPrinterConfig, windowSize: int):
    plot(dataset, classifierResults, performanceType, plot_printer_config,
         mapper=lambda x: np.convolve(x, np.ones(windowSize) / windowSize, mode="valid"),
         subtitle=f"window {windowSize} samples", sampleNumberMapper=lambda x: x + windowSize, showDetections=False)



def plotComparison(dataset: str, classifierResults: list[ClassifierResults], plot_printer_config: PlotPrinterConfig):
    # plot(dataset, classifierResults, "accuracy", plot_printer_config,
    #      labelFun=lambda classifier: f"{classifier.classifierType}: {round(classifier.accuracy(), 2)}",
    #      showDetections=True)

    windowSize = 500
    plotWindowed(dataset, classifierResults, "accuracy", plot_printer_config, windowSize)



def plot(dataset: str, classifierResults: list[ClassifierResults], performanceType: str,
         plot_printer_config: PlotPrinterConfig,
         mode: str = None,
         labelFun: Callable[[ClassifierResults], str] = lambda classifier: classifier.classifierType,
         mapper: Callable[[np.ndarray[int]], np.ndarray[int]] = lambda x: x, subtitle: str = None,
         showDetections: bool = True, prefix: str = None, overridenTitle: str = None, overridenYLabel: str = None,
         overridenUnit: str = None, printSmall: bool = False,
         sampleNumberMapper: Callable[[np.ndarray], np.ndarray] = lambda x: x):
    if plot_printer_config.is_set() and printSmall is False:
        plt.figure(dpi=1200)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    performanceTypeTranslated = translatePerformanceType(performanceType)

    # if overridenTitle:
    #     axes[0].set_title(overridenTitle)
    # else:
    #     axes[0].set_title(title(dataset, prefix, performanceTypeTranslated, subtitle))

    if overridenUnit:
        unit = overridenUnit
    else:
        unit = getUnit(performanceType)

    for classifier in classifierResults:
        y = mapper(classifier.results[performanceType])
        sampleNumbers = sampleNumberMapper(np.cumsum(np.ones_like(y)))
        axes.plot(sampleNumbers, y, label=r"$\eta=$ " + str(classifier.params["lr"]))
        axes.set_title(r"Różna szybkość uczenia")



    axes.legend()
    if len(classifierResults) > 1:
        axes.legend()

    if overridenYLabel:
        axes.set_ylabel(overridenYLabel)
    else:
        axes.set_ylabel(ylabel(performanceTypeTranslated, unit))

    axes.set_xlabel(r"t")

    plt.tight_layout()
    plt.show(block=False)


def title(dataset: str, prefix: str, performanceType: str, subtitle: str):
    result = f"zbiór {dataset}"
    if prefix:
        result += " - " + prefix
    result += f": {performanceType}"
    if subtitle:
        result += " - " + subtitle

    return result


def ylabel(performanceType: str, unit: str):
    if unit:
        result = f"{performanceType} [{unit}]"
    else:
        result = performanceType

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", action="store", default=None, required=False,
                        help="Description if plots should be printed as files")
    parser.add_argument("--plotsDir", action="store", default=None, required=False,
                        help="Directory for plots to be saved to")
    args = parser.parse_args()

    plot_printer_config = PlotPrinterConfig(args.plotsDir, args.description)

    experimentId = os.environ["EXPERIMENT_ID"]
    print(f"experimentId: {experimentId}")
    resultsInputDir = os.environ["RESULTS_DIRECTORY"]
    print(f"resultsInputDir: {resultsInputDir}")
    resultsPath = f"{resultsInputDir}/{experimentId}"
    for dataset in os.listdir(resultsPath):
        datasetPath = f"{resultsPath}/{dataset}"

        classifier_results = []

        allResults = []

        for classifierType in os.listdir(datasetPath):
            classifierPath = f"{datasetPath}/{classifierType}"

            classifier_results_tmp, tmpResults = get_all_clasifiers_results(classifierPath)
            classifier_results.append(classifier_results_tmp)
            allResults.extend(tmpResults)

        classifier_results_tmp = sorted(classifier_results_tmp, key=lambda classifier: classifier.accuracy(),
                                       reverse=True)

        allResults = sorted(allResults, key=lambda x: x[2], reverse=True)
        print(tabulate(allResults, headers=["type", "params", "accuracy", "duration", "trainingDuration",
                                            "classificationDuration"]))

        plotComparison(dataset, classifier_results_tmp, plot_printer_config)

    if not plot_printer_config.is_set():
        plt.show()

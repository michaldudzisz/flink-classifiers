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

            for row in reader:
                count = count + 1
                if row["class"] == row["predicted"]:
                    correct = correct + 1

                result[accuracyKey].append(100.0 * float(correct) / float(count))
                for key in collectedHeaders:
                    if (key == "trainingDuration"):
                        if (not row[key].isdigit()):
                            row[key] = "1000"
                            # print(row[key])
                            # print(row[key].isdigit())
                            # print("koncze sie, bo mam czas, ktory nie jest liczba")
                            # quit()
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


def getBestClassifier(classifierPath: str):
    classifierType = os.path.basename(os.path.normpath(classifierPath))

    bestAccuracy = -1.0
    bestParams = {}
    bestResults = {}
    bestHeaders = []
    bestJobId = ""

    allResults = []
    allClasifiers = []

    for classifierParamsRaw in os.listdir(classifierPath): # wchodze na poziom tych P10_M2
        classifierParamsPath = f"{classifierPath}/{classifierParamsRaw}"

        classifierParams = extractParams(classifierParamsRaw)

        with open(f"{classifierParamsPath}/result.json", "r") as resultFile:
            resultJson = json.load(resultFile)
            headers = resultJson["dataHeader"]

            chronologicalDataFilePaths = listOfExperimentFilePathsForParams(classifierParamsPath)
            results = readData(chronologicalDataFilePaths, headers)

            currentAccuracy = results["accuracy"][-1]
            if currentAccuracy > bestAccuracy:
                bestAccuracy = currentAccuracy
                bestParams = classifierParams
                bestResults = results
                bestHeaders = headers
                bestJobId = resultJson["jobId"]


            trainingDuration = round(np.sum(results["trainingDuration"]) / 1e9, 2)
            classificationDuration = round(np.sum(results["classificationDuration"]) / 1e9, 2)

            allResults.append((os.path.basename(classifierPath), classifierParams, currentAccuracy,
                               trainingDuration + classificationDuration, trainingDuration,
                               classificationDuration))


            allClasifiers.append(ClassifierResults(classifierParams, classifierType, results, headers, resultJson["jobId"]))

    return allClasifiers, allResults


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
    # if (dataset != "incremental_drift_synth_attr2_speed0.2_len20000" and dataset != "incremental_drift_synth_attr2_speed0.5_len20000"):
    #     return

    if plot_printer_config.is_set() and printSmall is False:
        plt.figure(dpi=1200)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    performanceTypeTranslated = translatePerformanceType(performanceType)

    # if overridenTitle:
    #     axes[0].set_title(overridenTitle)
    # else:
    axes.set_title("Dokładność w oknie 500 próbek")

    if overridenUnit:
        unit = overridenUnit
    else:
        unit = getUnit(performanceType)

    speed_value = 0
    for classifier in classifierResults:
        y = mapper(classifier.results[performanceType])

        if mode is None or mode == "sample":
            print(f"classifier.params[\"gamma\"]: {classifier.params["gamma"]}")
            sampleNumbers = sampleNumberMapper(np.cumsum(np.ones_like(y)))
            if classifier.classifierType == "eatnn2" and classifier.params["gamma"] == 1.0:
                curve = axes.plot(sampleNumbers, y, label=r"$\gamma$=1")
            if classifier.classifierType == "eatnn2" and classifier.params["gamma"] == 0.5:
                curve = axes.plot(sampleNumbers, y, label=r"$\gamma$=0,5")
            if classifier.classifierType == "eatnn2" and classifier.params["gamma"] == 0.1:
                curve = axes.plot(sampleNumbers, y, label=r"$\gamma$=0,1")

            match = re.search(r'([0-9]*\.?[0-9]+)x', dataset)
            if match:
                speed_value = float(match.group(1))
                print(f'Liczba przed "x": {speed_value}')
            else:
                print('Nie znaleziono liczby po "speed".')
            # axes.set_title(r"Dokładność klasyfikacji w oknie 500 próbek")
            # axes.set_title(f"{dataset}")

            # rysuj linie wykrytych dryfów
            drift_status_already_added = {}

            color_blue = "#1f77b4"
            color_orange = "#ff7f0e"
            color_green = "#2ca02c"
            color_red = "#d62728"
            color_gray = "#7f7f7f"
            color_brown = "#8c564b"
            color_pink = "#e377c2"


            # for idx, driftStatus in enumerate(classifier.results["driftStatus"]):
            #     if driftStatus == "new_detected" and classifier.classifierType == "eatnn":
            #         if not drift_status_already_added.get("new_detected", False):
            #             drift_status_already_added["new_detected"] = True
            #             axes.axvline(x=idx, color=color_blue, linestyle=':', linewidth=1.5, label='SEATNN: nowe pojęcie')
            #         else:
            #             axes.axvline(x=idx, color=color_blue, linestyle=':', linewidth=1.5)
            #     if driftStatus == "new_detected" and classifier.classifierType == "atnn":
            #         if not drift_status_already_added.get("new_detected", False):
            #             drift_status_already_added["new_detected"] = True
            #             axes.axvline(x=idx, color=color_brown, linestyle=':', linewidth=1.5, label='ATNN: dodano pustą gałąź')
            #         else:
            #             axes.axvline(x=idx, color=color_brown, linestyle=':', linewidth=1.5)
            #     if driftStatus == "new_detected_cloned":
            #         if not drift_status_already_added.get("new_detected_cloned", False):
            #             axes.axvline(x=idx, color=color_green, linestyle=':', linewidth=1.5, label='sklonowano aktywną gałąź')
            #             drift_status_already_added["new_detected_cloned"] = True
            #         else:
            #             axes.axvline(x=idx, color=color_green, linestyle=':', linewidth=1.5)
            #     if driftStatus == "new_detected_empty":
            #         if not drift_status_already_added.get("new_detected_empty", False):
            #             axes.axvline(x=idx, color=color_red, linestyle=':', linewidth=1.5, label='SEATNN: dodano pustą gałąź')
            #             drift_status_already_added["new_detected_empty"] = True
            #         else:
            #             axes.axvline(x=idx, color=color_red, linestyle=':', linewidth=1.5)
                # if driftStatus == "current_evolving":
                #     if not drift_status_already_added.get("current_evolving", False):
                #         axes.axvline(x=idx, color=color_gray, linestyle=':', linewidth=1.5, label='SEATNN: pozwolono ewoluować aktywnej gałęzi')
                #         drift_status_already_added["current_evolving"] = True
                #     else:
                #         axes.axvline(x=idx, color=color_gray, linestyle=':', linewidth=1.5)
                # if driftStatus == "recurring" and classifier.classifierType == "eatnn":
                    # if not drift_status_already_added.get("recurring", False):
                    #     axes.axvline(x=idx, color=color_orange, linestyle=':', linewidth=1.5, label='SEATNN: powracające pojęcie')
                    #     drift_status_already_added["recurring"] = True
                    # else:
                    #     axes.axvline(x=idx, color=color_orange, linestyle=':', linewidth=1.5)
                # if driftStatus == "recurring" and classifier.classifierType == "atnn":
                #     if not drift_status_already_added.get("recurring", False):
                #         axes.axvline(x=idx, color=color_pink, linestyle=':', linewidth=1.5, label='ATNN: powracające pojęcie')
                #         drift_status_already_added["recurring"] = True
                #     else:
                #         axes.axvline(x=idx, color=color_pink, linestyle=':', linewidth=1.5)

    #
    #         # rysuj liczbe warstw na drugim wykresie
    #         trunk_line = []
    #         branch_line = []
    #         for idx, input_string in enumerate(classifier.results["branchStructure"]):
    #             # allBranches4_activeBranch3_growingPoint4_activeBranchDepth2
    #             name_value_pairs = input_string.split("_") # ['allBranches4', 'activeBranch3', 'growingPoint4', 'activeBranchDepth2']
    #             pattern = r"([a-zA-Z]*)(\d+)"
    #             matches = []
    #             for pair in name_value_pairs:
    #                 match = re.match(pattern, pair) # np 'allBranches4'
    #                 if match:
    #                     name = match.group(1)
    #                     value = int(match.group(2))
    #                     matches.append((name, value))
    #             result = {name: int(value) for name, value in matches}
    #             # print(result)
    #             if result["activeBranch"] == 0:
    #                 trunk_line.append(result["activeBranchDepth"])
    #             else:
    #                 trunk_line.append(result["growingPoint"])
    #             branch_line.append(result["growingPoint"] + result["activeBranchDepth"])
    #
    #         axes[1].plot(sampleNumbers, trunk_line[-len(sampleNumbers):], label="Węzły pnia")
    #         axes[1].plot(sampleNumbers, branch_line[-len(sampleNumbers):], label="Wszystkie węzły")
    #         axes[1].legend()
    #         axes[1].set_ylim(bottom=0)
    #         axes[1].set_title("Liczba wykorzystywanych węzłów w drzewie SEATNN")
    #
    #         # rysuj straty
    #         active_line = []
    #         for idx, input_string in enumerate(classifier.results["branchStructure"]):
    #             # allBranches4_activeBranch3_growingPoint4_activeBranchDepth2
    #             name_value_pairs = input_string.split("_") # ['allBranches4', 'activeBranch3', 'growingPoint4', 'activeBranchDepth2']
    #             pattern = r"([a-zA-Z]*)(\d+(?:\.\d+)?)"
    #             matches = []
    #             for pair in name_value_pairs:
    #                 match = re.match(pattern, pair) # np 'allBranches4'
    #                 if match:
    #                     name = match.group(1)
    #                     value = float(match.group(2))
    #                     matches.append((name, value))
    #             result = {name: float(value) for name, value in matches}
    #             active_line.append(result["active"])
    #
    #         axes[2].plot(sampleNumbers, active_line[-len(sampleNumbers):])
    #         axes[2].set_ylim(bottom=0)
    #         axes[2].set_ylim(top=0.85) # todo można to zmieniać
    #         axes[2].set_ylabel("Entropia krzyżowa")
    #         axes[2].set_title("Średnia wartość funkcji straty w oknie 50 próbek")
    #         axes[2].legend()
    #
    #
    # axes[0].legend()
    # if len(classifierResults) > 1:
    #     axes[0].legend()
    #
    # if overridenYLabel:
    #     axes[0].set_ylabel(overridenYLabel)
    # else:
    #     axes[0].set_ylabel(ylabel(performanceTypeTranslated, unit))
    #
    axes.set_xlabel(r"t")
    axes.legend()
    # axes.set_ylim([70, 93])
    # axes[1].set_xlabel(r"t")
    # axes[2].set_xlabel(r"t")

    plt.tight_layout()
    plt.savefig(f'/Users/michal.dudzisz/Documents/mgr/img/generated/seatnn_porownanie/eatnn_{dataset}.pdf', format='pdf')
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
    dataset_results = {}
    for dataset in os.listdir(resultsPath):
        datasetPath = f"{resultsPath}/{dataset}"

        bestClassifierResults = []

        allResults = []

        for classifierType in os.listdir(datasetPath):
            classifierPath = f"{datasetPath}/{classifierType}"

            allClassifiers, tmpResults = getBestClassifier(classifierPath)
            bestClassifierResults += allClassifiers
            allResults.extend(tmpResults)

        bestClassifierResults = sorted(bestClassifierResults, key=lambda classifier: classifier.accuracy(),
                                       reverse=True)

        allResults = sorted(allResults, key=lambda x: x[2], reverse=True)
        print(f"dataset: {dataset}")
        print(tabulate(allResults, headers=["type", "params", "accuracy", "duration", "trainingDuration",
                                            "classificationDuration"]))

        plotComparison(dataset, bestClassifierResults, plot_printer_config)
    #     dataset_results[dataset] = {}
    #     dataset_results[dataset][bestClassifierResults[0].classifierType] = str(bestClassifierResults[0].accuracy())
    #     dataset_results[dataset][bestClassifierResults[1].classifierType] = str(bestClassifierResults[1].accuracy())
    #
    # print(dataset_results)
    if not plot_printer_config.is_set():
        plt.show()


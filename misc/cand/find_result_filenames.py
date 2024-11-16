import os
from dataclasses import dataclass
import json


@dataclass
class ExperimentResultsMetadata:
    name: str
    dataset: str
    params: {}
    files: []
    file_headers: []
    exp_location: str


def extractParams(classifierDir: str) -> dict[str, float]:
    resultDict = {}

    for classifierRaw in classifierDir.split("_"):
        valueBeginIdx = -1
        for i, c in enumerate(classifierRaw):
            if c.isdigit():
                valueBeginIdx = i
                break

        key = classifierRaw[:(valueBeginIdx)]
        value = float(classifierRaw[valueBeginIdx:])
        resultDict[key] = value

    return resultDict

def read_csv_files_header(path_to_result_json_file: str):
    with open(path_to_result_json_file, "r") as resultFile:
        resultJson = json.load(resultFile)
        headers = resultJson["dataHeader"]
        return headers


def create_experiment_results_metadata(classifierPath: str, dataset: str, classifier_type: str):
    results = []
    for classifier_params_raw in os.listdir(classifierPath): # for example Psize10_Msize2
        classifier_params_path = f"{classifierPath}/{classifier_params_raw}"
        classifier_params = extractParams(classifier_params_raw)
        chronological_data_file_paths = list_of_experiment_file_paths_for_params(classifier_params_path)
        results.append(ExperimentResultsMetadata(
            name=f"{classifier_type}_{classifier_params_raw}",
            dataset=dataset,
            params=classifier_params,
            files=chronological_data_file_paths,
            file_headers=read_csv_files_header(f"{classifier_params_path}/result.json"),
            exp_location=classifier_params_path
        ))

    return results


def list_of_experiment_file_paths_for_params(classifierExpPath: str) -> list[str]:
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


def find_results_metadata(experiment_id: str) -> list[ExperimentResultsMetadata]:
    """
    Performs a lookup for experiment results files and its metadata.
    :param experiment_id:
    :return:
    """
    results_directory = "results"
    experiment_path = f"{results_directory}/{experiment_id}"

    results = []
    for dataset in os.listdir(experiment_path):
        dataset_path = f"{experiment_path}/{dataset}" # for example results/xxx/elec
        for classifier_type in os.listdir(dataset_path):
            classifier_type_path = f"{dataset_path}/{classifier_type}" # for example results/xxx/elec/cand
            results.extend(create_experiment_results_metadata(classifier_type_path, dataset, classifier_type))
    return results


if __name__ == "__main__":
    r = find_results_metadata("2024-11-13T15:16:39")
    print(r)

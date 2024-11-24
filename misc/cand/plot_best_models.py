import pandas as pd
import matplotlib.pyplot as plt
import os
from find_result_filenames import find_results_metadata


def get_neurons_from_name(name: str) -> int:
    # name: for example L1_N8_SGD_0.50000
    return int(name.split(sep="_")[1][1:])

def get_optimizer_from_name(name: str) -> str:
    # name: for example L1_N8_SGD_0.50000
    return name.split(sep="_")[2]

def get_lr_from_name(name: str) -> float:
    # name: for example L1_N8_SGD_0.50000
    return float(name.split(sep="_")[3])

def get_best_classifiers(data):
    # data['layers'] = data.apply(lambda row: get_layers_from_int_value(row['modelName']), axis=1)
    data['optimizer'] = data.apply(lambda row: get_optimizer_from_name(row['bestMLPName']), axis=1)
    data['neurons'] = data.apply(lambda row: get_neurons_from_name(row['bestMLPName']), axis=1)
    data['lr'] = data.apply(lambda row: get_lr_from_name(row['bestMLPName']), axis=1)
    data = data[['optimizer', 'neurons', 'lr']]
    data = data.reset_index(drop=True)
    return data


def make_charts(data):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    axs[0].scatter(range(0, len(data)), data['neurons'], marker='.', color='blue', linewidths=0.1)
    # axs[0].set_title("Wykres b w zależności od a")
    axs[0].set_xlabel("id")
    axs[0].set_ylabel("neurons")

    axs[1].scatter(range(0, len(data)), data['optimizer'], marker='.', color='green', linewidths=0.1)
    # axs[1].set_title("Wykres c w zależności od a")
    axs[1].set_xlabel("id")
    axs[1].set_ylabel("optimizer")

    axs[2].scatter(range(0, len(data)), data['lr'], marker='.', color='red', linewidths=0.1)
    # axs[2].set_title("Wykres d w zależności od a")
    axs[2].set_xlabel("id")
    axs[2].set_ylabel("lr")
    axs[2].set_yscale("log")

    # Wyświetlenie wykresów
    plt.tight_layout()
    plt.show()


def create_best_classifiers_file(data, filename):
    data.to_csv(filename, index=False)

def load_csv_files(file_list, headers: [str]):
    dataframes = [pd.read_csv(file, names=headers) for file in file_list]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def percentage_of_parameters_for_best_mlps(data: pd.DataFrame) -> dict:
    percentages = {}
    for col in data.columns:
        percentages[col] = data[col].value_counts(normalize=True) * 100
    return percentages

if __name__ == "__main__":
    try:
        experiment_id = os.environ["EXPERIMENT_ID"]
    except KeyError:
        experiment_id = "I don't exist"

    metadata = find_results_metadata(experiment_id)
    for experiment_metadata in metadata:
        data = load_csv_files(experiment_metadata.files, experiment_metadata.file_headers)
        best_classifiers = get_best_classifiers(data)
        # create_best_classifiers_file(data, f"{experiment_metadata.exp_location}/best_classifiers.csv")
        print(percentage_of_parameters_for_best_mlps(best_classifiers))
        make_charts(best_classifiers)

    # create_best_classifiers_file(best_classifiers, "cand_best_classifiers.csv")


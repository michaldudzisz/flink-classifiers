import pandas as pd
import matplotlib.pyplot as plt
import os
from find_result_filenames import find_results_metadata
import math

# public final static String USED_OPTIMIZER = "usedOptimizer";
# public final static String LAYER_SIZE = "hiddenLayerSize";
# public final static String LEARNING_RATE = "lr";
#
# public final static Long SGD = 0L;
# public final static Long ADAM = 1L;
#
# //    public final static Long N2E8 = 0L;
# //    public final static Long N2E9 = 1L;
# //    public final static Long N2E10 = 2L;
# //
# public final static Long LR5Em1 = 0L;
# public final static Long LR5Em2 = 1L;
# public final static Long LR5Em3 = 2L;
# public final static Long LR5Em4 = 3L;
# public final static Long LR5Em5 = 4L;
#

# def get_layers_from_int_value(name: str) -> str:
#     # name: for example L1_N8_SGD_0.50000
#     return name.split(sep="_")[0][1:]

def get_neurons_from_int_value(value: int) -> float:
    return 2 ** int(value)

def get_optimizer_from_int_value(value: int) -> str:
    __SGD = 0
    __ADAM = 1

    match int(value):
        case int(__SGD):
            return "SGD"
        case int(_ADAM):
            return "ADAM"
        case _:
            raise Exception(f"Not allowed value {value} in optimizer")


def get_lr_from_int_value(value: int) -> float:
    __LR5Em1 = 0
    __LR5Em2 = 1
    __LR5Em3 = 2
    __LR5Em4 = 3
    __LR5Em5 = 4

    if value not in [__LR5Em1, __LR5Em2, __LR5Em3, __LR5Em4, __LR5Em5]:
        raise Exception(f"learning rate {value} not allowed")

    return 5 * math.pow(10, -(value + 1))


def get_best_classifiers(data):
    # data['layers'] = data.apply(lambda row: get_layers_from_int_value(row['modelName']), axis=1)
    data['optimizer'] = data.apply(lambda row: get_optimizer_from_int_value(row['usedOptimizer']), axis=1)
    data['neurons'] = data.apply(lambda row: get_neurons_from_int_value(row['hiddenLayerSize']), axis=1)
    data['lr'] = data.apply(lambda row: get_lr_from_int_value(row['lr']), axis=1)
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


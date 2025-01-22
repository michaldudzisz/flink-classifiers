import pandas as pd
import matplotlib.pyplot as plt
import os
from find_result_filenames import find_results_metadata


def get_best_classifiers(data):
    data = data[['bestMLPName']]
    return data

def all_classifiers_losses(data):
    data = data[['MLPLosses']]

    # obtain all model names:
    first_row = data.iloc[0] # for example L1_N9_SGD_0.05000:0.20629471110616526;L1_N8_SGD_0.05000:0.21534860383321314
    pairs = first_row['MLPLosses'].split(";")
    model_names = []
    for pair in pairs:
        model_names.append(pair.split(":")[0]) # for example L1_N9_SGD_0.05000

    new_df = pd.DataFrame(columns=model_names)
    new_df[model_names] = data['MLPLosses'].str.split(';', expand=True)
    new_df = new_df.applymap(lambda x: float(x.split(':')[1]) if isinstance(x, str) else x)

    return new_df


def make_charts(best_classifiers, losses):
    cumulative_counts = pd.DataFrame(columns=best_classifiers['bestMLPName'].unique())

    for idx in range(len(best_classifiers)):
        current_counts = best_classifiers.loc[:idx, 'bestMLPName'].value_counts()
        cumulative_counts.loc[idx] = current_counts

    cumulative_counts.fillna(0, inplace=True)

    fig, axs = plt.subplots(2, 1, figsize=(10, 11))
    for column in cumulative_counts.columns:
        axs[0].plot(cumulative_counts.index, cumulative_counts[column], label=column, linewidth=1.5)

    axs[0].set_xlabel('Index of Row', fontsize=12)
    axs[0].set_ylabel('Cumulative Occurrences', fontsize=12)
    axs[0].set_title('Cumulative Occurrences of Model Names Over Time', fontsize=14)

    print("losses: ")
    print(losses)

    # losses = losses[["L1_N9_SGD_0.05000"]]
    for column in losses.columns:
        axs[1].plot(losses.index, losses[column], label=column)

    axs[1].set_title(f'Wykres dla kolumny')
    axs[1].set_xlabel('Indeks')
    axs[1].set_ylim((0, 8))
    # axs[1].set_ylabel('Wartość')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=6)

    plt.grid(alpha=0.4)
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
    # export EXPERIMENT_ID=2024-11-16T16:11:52
    try:
        experiment_id = os.environ["EXPERIMENT_ID"]
    except KeyError:
        experiment_id = "I don't exist"

    metadata = find_results_metadata(experiment_id)
    for experiment_metadata in metadata:
        print("loading csv data...")
        data = load_csv_files(experiment_metadata.files, experiment_metadata.file_headers)
        print("collecting losses...")
        losses = all_classifiers_losses(data)
        print("choosing best classifier...")
        best_classifiers = get_best_classifiers(data)
        print("drawing charts...")
        make_charts(best_classifiers, losses)

import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from find_result_filenames import find_results_metadata
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def get_best_classifiers(data):
    data = data[['bestMLPName']]
    return data

def split_row(row):
    split_data = {}
    for value in row:
        key, val = value.split(":")
        split_data[key] = float(val)
    return split_data

def all_classifiers_losses(data):
    data = data[['MLPLosses']]
    print(f"data: {data.iloc[2000]}")

    # obtain all model names:
    first_row = data.iloc[1] # for example L1_N9_SGD_0.05000:0.20629471110616526;L1_N8_SGD_0.05000:0.21534860383321314
    # print(f"first_row: {first_row}")
    pairs = first_row['MLPLosses'].split(";")
    # print(f"pairs: {pairs}")
    model_names = []
    for pair in pairs:
        model_names.append(pair.split(":")[0]) # for example L1_N9_SGD_0.05000

    print(f"model_names: {model_names}")
    print("\n\n")
    new_df = pd.DataFrame(columns=model_names)
    some_df = data['MLPLosses'].str.split(';', expand=True)
    new_df = pd.DataFrame([split_row(row) for _, row in some_df.iterrows()])
    # quit()
    #
    # new_df[model_names] = data['MLPLosses'].str.split(';', expand=True)
    # print(f"new_df_N9: {new_df["L1_N9_ADAM_0.00050"][2000]}")
    # print(f"new_df_N8: {new_df["L1_N8_ADAM_0.00050"][2000]}")
    # quit()
    # new_df = new_df.applymap(lambda x: float(x.split(':')[1]) if isinstance(x, str) else x)

    # print(f"new_df: {new_df.head(2)}")
    # quit()

    return new_df


def make_charts(best_classifiers, losses):
    nums = 29_999
    # nums = 3_999
    best_classifiers = best_classifiers.head(nums)
    losses = losses.head(nums)
    cumulative_counts = pd.DataFrame(columns=losses.columns)

    for idx in range(len(best_classifiers)):
        current_counts = best_classifiers.loc[:idx, 'bestMLPName'].value_counts()
        cumulative_counts.loc[idx] = current_counts

    cumulative_counts.fillna(0, inplace=True)

    # Zliczanie całkowitych wystąpień klasyfikatorów
    total_counts = cumulative_counts.iloc[-1].sort_values(ascending=False)
    classifiers = total_counts.index

    # Tworzenie kolorów od niebieskiego do żółtego
    color_gradient = [
        (0.0, "#173c87"),
        (0.05, "#3abd9e"),
        (0.3, "#edd78e"),
        (0.7, "#f5ecb8"),
        (1.0, "#f2b996"),
    ]

    sorted_columns = cumulative_counts.iloc[-1].sort_values(ascending=False).index.tolist()
    print(f"cumulative_counts last row: {cumulative_counts.iloc[-1]}")
    print(f"sorted_columns: {sorted_columns}")

    sorted_columns_to_draw = cumulative_counts.iloc[-1].sort_values(ascending=True).index.tolist()
    cumulative_counts = cumulative_counts[sorted_columns_to_draw]
    losses = losses[sorted_columns_to_draw]


    cmap = LinearSegmentedColormap.from_list("blue_to_yellow", color_gradient)
    norm = plt.Normalize(vmin=0, vmax=29)
    colors = {classifier: cmap(norm(idx)) for idx, classifier in enumerate(sorted_columns)}

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Wykres dla kumulatywnych wystąpień klasyfikatorów
    for column in cumulative_counts.columns:
        axs[0].plot(cumulative_counts.index, cumulative_counts[column], label=column, linewidth=1.5, color=colors[column])

    axs[0].axvline(x=5_000,  color='black', linestyle=':', linewidth=1)
    axs[0].axvline(x=10_000, color='black', linestyle=':', linewidth=1)
    axs[0].axvline(x=15_000, color='black', linestyle=':', linewidth=1)
    axs[0].axvline(x=20_000, color='black', linestyle=':', linewidth=1)
    axs[0].axvline(x=25_000, color='black', linestyle=':', linewidth=1)


    axs[0].set_xlabel(r'$t$', fontsize=12)
    axs[0].set_ylabel('Liczba wystąpień', fontsize=12)
    axs[0].set_title('Liczba wyboru danego klasyfikatora', fontsize=14)
    axs[0].set_xlim((0, 30_000))
    # axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=6)

    # print(f"losses_best: {losses["L1_N9_ADAM_0.00050"][2000]}")
    # print(f"losses_second: {losses["L1_N8_ADAM_0.00050"][2000]}")

    # Wykres dla strat klasyfikatorów
    for column in losses.columns:
        axs[1].plot(losses.index, losses[column], label=column, color=colors[column])

    axs[1].axvline(x=5_000,  color='black', linestyle=':', linewidth=1)
    axs[1].axvline(x=10_000, color='black', linestyle=':', linewidth=1)
    axs[1].axvline(x=15_000, color='black', linestyle=':', linewidth=1)
    axs[1].axvline(x=20_000, color='black', linestyle=':', linewidth=1)
    axs[1].axvline(x=25_000, color='black', linestyle=':', linewidth=1)

    axs[1].set_title(f'Przebieg estymowanej funkcji straty dla poszczególnych klasyfikatorów')
    axs[1].set_xlabel(r'$t$')
    axs[1].set_yscale('log')
    axs[1].set_ylim((0.18, 70))
    axs[1].set_xlim((0, 30_000))
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

import pandas as pd
import matplotlib.pyplot as plt

def plot_gradual():
    dataset_len=100_000
    df = pd.read_csv('./datasets/sea_grad.csv', nrows=dataset_len, sep="[,#]", engine="python")

    for i in [0, 9]:
        start_row = i * 10_000
        end_row = i * 10_000 + 10_000
        df_subset = df.iloc[start_row:end_row]

        masks = []
        num_classes = 10
        for k in range(0, num_classes):
            masks.append(df_subset['class'] == k)


        plt.figure(figsize=(5, 5))

        for k in range(0, num_classes):
            plt.scatter(df_subset.loc[masks[k], 'x1'], df_subset.loc[masks[k], 'x2'], label='class = ' + str(k), alpha=0.6)


        plt.xlabel('atrybut 1')
        plt.ylabel('atrybut 2')
        # plt.title('Wykres punktów z kolumn a, b oraz c')
        # plt.legend(loc="upper center", bbox_to_anchor=(0.0, -0.0), ncol=2)
        # plt.legend()

        plt.show()


def plot_abrupt():
    dataset_len=100_000
    df = pd.read_csv('./datasets/sea_abr.csv', nrows=dataset_len, sep="[,#]", engine="python")

    for i in [1, 3, 6, 9]:
        start_row = i * 10_000
        end_row = i * 10_000 + 10_000
        df_subset = df.iloc[start_row:end_row]

        num_classes = 2
        masks = []
        for k in range(0, num_classes):
            masks.append(df_subset['class'] == k)


        plt.figure(figsize=(5, 5))

        for k in range(0, num_classes):
            plt.scatter(df_subset.loc[masks[k], 'x1'], df_subset.loc[masks[k], 'x2'], label='class = ' + str(k), alpha=0.6)


        plt.xlabel('atrybut 1')
        plt.ylabel('atrybut 2')
        # plt.title('Wykres punktów z kolumn a, b oraz c')
        # plt.legend(loc="upper center", bbox_to_anchor=(0.0, -0.0), ncol=2)
        # plt.legend()

        plt.show()



if __name__ == "__main__":
    plot_abrupt()
    # plot_gradual()


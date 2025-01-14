import random
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

class SEA:

    def __init__(self, seed=42):
        self.bound = 10
        self.thresholds_percentages = [0.10, 0.16, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58] # gives 10 classes
        self.thresholds = list(map(lambda x: x * 2 * self.bound, self.thresholds_percentages))
        self.dataset_length = 60_000 # 100_000
        random.seed(seed)

    def generate_gradual(self) -> pd.DataFrame:
        dataset = {
            "x1": np.random.rand(self.dataset_length) * self.bound,
            "x2": np.random.rand(self.dataset_length) * self.bound,
            "x3": np.random.rand(self.dataset_length) * self.bound,
        }

        drift_start = 10_000
        drift_end = 60_000

        classes = []
        for i in range(0, drift_start):
            value = dataset["x1"][i] + dataset["x2"][i]
            classes.append(self.__determine_class(value, self.thresholds))


        temporary_thresholds = self.thresholds.copy()
        for i in range(drift_start, drift_end):
            temporary_threshold_percentages = list(map(lambda x: x + 0.2 * (i - drift_start) / (drift_end - drift_start), self.thresholds_percentages))
            temporary_thresholds = list(map(lambda x: x * 2 * self.bound, temporary_threshold_percentages))
            value = dataset["x1"][i] + dataset["x2"][i]
            classes.append(self.__determine_class(value, temporary_thresholds))

        for i in range(drift_end, self.dataset_length):
            value = dataset["x1"][i] + dataset["x2"][i]
            classes.append(self.__determine_class(value, temporary_thresholds))
            # print("dataset[\"x1\"][i]: " + str(dataset["x1"][i]))
            # print("dataset[\"x2\"][i]: " + str(dataset["x2"][i]))
            # print("value: " + str(value))
            # print("self.__determine_class(value, self.thresholds): " + str(self.__determine_class(value, temporary_thresholds)))
            # print("temporary_thresholds: " + str(temporary_thresholds))
            # quit()

        dataset["class"] = classes
        return pd.DataFrame(dataset)



    def generate_abrupt(self) -> pd.DataFrame:
        dataset = {
            "x1": np.random.rand(self.dataset_length) * self.bound,
            "x2": np.random.rand(self.dataset_length) * self.bound,
            "x3": np.random.rand(self.dataset_length) * self.bound,
        }

        # drift_points_percentages = [0.25, 0.50, 0.75] # four concepts
        # concept_thetas = [8, 9, 7, 9.5]

        drift_points_percentages = [0.25, 0.50, 0.75] # four concepts
        concept_thetas = [8, 9, 7, 9.5]

        classes = []
        for i in range(0, self.dataset_length):
            value = dataset["x1"][i] + dataset["x2"][i]

            current_concept = 0
            for drift_point in drift_points_percentages:
                current_percentage = i / self.dataset_length
                if current_percentage > drift_point:
                    current_concept = current_concept + 1


            current_theta = concept_thetas[current_concept]
            current_thresholds = [current_theta]
            classes.append(self.__determine_class(value, current_thresholds))

        dataset["class"] = classes
        return pd.DataFrame(dataset)



    def __determine_class(self, value, current_thresholds):
        max_class = 0
        for current_class, current_threshold in enumerate(current_thresholds):
            if value < current_threshold:
                return current_class
            else:
                max_class = max_class + 1

        # if not found earlier
        return max_class


def save_to_normal_csv(data, path_to_file):
    data.to_csv(path_to_file, index=False)

def save_to_flink_csv(data, path_to_file):
    # Normalizacja kolumn do zakresu (0, 1) z wykluczeniem ostatniej kolumny
    scaler = MinMaxScaler()
    df_to_normalize = data.iloc[:, :-1]
    df_normalized = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=df_to_normalize.columns)
    df_normalized[data.columns[-1]] = data[data.columns[-1]].astype(int).values

    with open(path_to_file, 'w') as file:
        file.write('x1,x2,x3,class\n')
        for index, row in data.iterrows():
            file.write(f"{row['x1']}#{row['x2']}#{row['x3']},{int(row['class'])}\n")


if __name__ == "__main__":
    sea = SEA()

    gradual_dataset = sea.generate_gradual()
    # save_to_normal_csv(gradual_dataset, './datasets/sea_grad_norm.csv')
    save_to_flink_csv(gradual_dataset, './datasets/sea_inc.csv')

    abrupt_dataset = sea.generate_abrupt()
    # save_to_normal_csv(abrupt_dataset, './datasets/sea_abr_norm.csv')
    save_to_flink_csv(abrupt_dataset, './datasets/sea_abr.csv')


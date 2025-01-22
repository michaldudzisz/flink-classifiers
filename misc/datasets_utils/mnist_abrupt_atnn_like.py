import random
import pandas as pd
import numpy as np
import csv
import numpy as np
import cv2
import math

# download MNIST for example from here: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
# we use dataset already converted to a csv file
# the format is: label, pix-11, pix-12, pix-13, ...
_MNIST_PATH = "datasets/mnist.csv"


class MNIST:

    _ATTRIBUTES = 28 * 28
    _DATASET_LEN = 60_000

    _DRIFT_PERIODS = [range(5_000, 10_000), range(15_000, 20_000), range(25_000, 30_000)]

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self._image_one_saved = False
        self._image_two_saved = False
        self._image_three_saved = False

    def generate(self, output_path):
        with (open(self.dataset_path, 'r') as csv_file, open(output_path, 'w') as output_csv_file):
            reader = csv.reader(csv_file)
            self._write_output_file_header(output_csv_file)
            for line_number, row in enumerate(reader):
                if line_number == 30_000:
                    break
                self._log_progress(line_number)
                if len(row) != self._ATTRIBUTES + 1: # attributes + one class label
                    raise Exception(f"Line {line_number} invalid, incorrect number of pixels.")
                label = row[0]
                values = np.array(row[1:], dtype=np.uint8)
                values  = values / 255 # normalize dataset

                if line_number in self._DRIFT_PERIODS[0] or line_number in self._DRIFT_PERIODS[1] or line_number in self._DRIFT_PERIODS[2]:
                    if label == "0":
                        label = "1"
                    elif label == "2":
                        label = "3"
                    elif label == "4":
                        label = "5"
                    elif label == "1":
                        label = "0"
                    elif label == "3":
                        label = "2"
                    elif label == "5":
                        label = "4"

                self._write_output_file_line(output_csv_file, values, label)

        with open(output_path.replace(".csv", ".txt"), "w") as f:
            f.write("0 0\n1 1\n2 2\n3 3\n4 4\n5 5\n6 6\n7 7\n8 8\n9 9")


    def _log_progress(self, line_number):
        if line_number % 1_000 == 0:
            print(f"Processed {line_number} lines.")

    def _write_output_file_header(self, file):
        file.write(f"x0")
        for i in range(1, self._ATTRIBUTES):
            file.write(f",x{i}")
        file.write(f",class\n")

    def _write_output_file_line(self, file, attributes, label):
        file.write(f"{attributes[0]}")
        for i in range(1, self._ATTRIBUTES):
            file.write(f"#{attributes[i]}")
        file.write(f",{label}\n")

def save_to_normal_csv(data, path_to_file):
    data.to_csv(path_to_file, index=False)

def save_to_flink_csv(data, path_to_file):
    with open(path_to_file, 'w') as file:
        file.write('x1,x2,x3,class\n')
        for index, row in data.iterrows():
            file.write(f"{row['x1']}#{row['x2']}#{row['x3']},{int(row['class'])}\n")


if __name__ == "__main__":
    # download MNIST for example from here: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
    # we use dataset already converted to a csv file
    # the format is: label, pix-11, pix-12, pix-13, ...
    # mnist = MNIST(_MNIST_PATH)
    # mnist.generate("./datasets/mnist_grad.csv", drift_start=0.50, drift_end=0.75, visualize=True)

    mnist = MNIST(_MNIST_PATH)
    mnist.generate("./datasets/mnist_abrupt_atnn_like.csv")

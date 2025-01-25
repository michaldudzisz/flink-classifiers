import random
import pandas as pd
import numpy as np
import csv
import numpy as np
import cv2
import math

# download MNIST for example from here: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
# we use dataset already converted to a csv file
# the format is: pix-11, pix-12, pix-13, label...
_MNIST_PATH = "datasets/mnist_fashion_train.csv"


class MNIST:

    _ATTRIBUTES = 28 * 28
    _DATASET_LEN = 60_000

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self._image_one_saved = False
        self._image_two_saved = False
        self._image_three_saved = False

    def generate(self, output_path, drift_start, drift_end=None, visualize=False, line_number_to_end_at=None):
        with open(self.dataset_path, 'r') as csv_file, open(output_path, 'w') as output_csv_file:
            reader = csv.reader(csv_file)
            self._write_output_file_header(output_csv_file)
            last_percentage = 0
            images_to_visualize = []
            for line_number, row in enumerate(reader):
                if len(row) != self._ATTRIBUTES + 1: # attributes + one class label
                    raise Exception(f"Line {line_number} invalid, incorrect number of pixels.")
                label = row[0]
                values = np.array(row[1:], dtype=np.uint8)
                values  = values / 255 # normalize dataset
                image = values.reshape((28, 28))
                current_percentage = self._get_current_percentage(line_number)
                angle = self._rotation_angle(drift_start=drift_start, percentage=current_percentage, drift_end=drift_end)
                self._log_progress(last_percentage, current_percentage, angle)
                last_percentage = current_percentage
                rotated_image = self._rotate_image(image, angle)
                self._save_to_visualize_if_applicable(visualize, images_to_visualize, rotated_image, label, current_percentage)
                rotated_list = rotated_image.flatten().tolist()
                self._write_output_file_line(output_csv_file, rotated_list, label)

                if line_number_to_end_at:
                    if line_number == line_number_to_end_at:
                        break

            if visualize:
                for image in images_to_visualize:
                    image = image # * 255
                    cv2.imshow(f"Image", image)
                    cv2.waitKey(0)


    def generate_atnn_like(self, output_path, drift_start_line_number, line_number_to_end_at=None):
        with open(self.dataset_path, 'r') as csv_file, open(output_path, 'w') as output_csv_file:
            reader = csv.reader(csv_file)
            self._write_output_file_header(output_csv_file)
            for line_number, row in enumerate(reader):
                if len(row) != self._ATTRIBUTES + 1: # attributes + one class label
                    raise Exception(f"Line {line_number} invalid, incorrect number of pixels.")
                label = row[0]
                values = np.array(row[1:], dtype=np.uint8)
                values  = values / 255 # normalize dataset

                if line_number >= drift_start_line_number:
                    if label == "0":
                        label = "1"
                    if label == "2":
                        label = "3"
                    if label == "4":
                        label = "5"

                self._write_output_file_line(output_csv_file, values, label)

                if line_number_to_end_at:
                    if line_number == line_number_to_end_at:
                        break


    def _rotate_image(self, image, angle):
        # angle - in degrees
        (h, w) = image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rotated

    def _get_current_percentage(self, index):
        return index / self._DATASET_LEN

    def _rotation_angle(self, drift_start, percentage, drift_end=None):
        angle = 0
        if drift_end is None:
            angle = 0 if percentage <= drift_start else 90
            return -angle

        if drift_end is not None:
            if percentage <= drift_start:
                angle = 0
            else: # drift_start < percentage <= drift_end:
                angle = (percentage - drift_start) / (drift_end - drift_start) * 90

        return -angle

    def _log_progress(self, last_percentage, current_percentage, angle):
        last_progress = math.floor(last_percentage * 100)
        current_progress = math.floor(current_percentage * 100)
        if current_progress > last_progress:
            print(f"{current_progress}%, a: {math.floor(angle)}")

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

    def _save_to_visualize_if_applicable(self, visualize, images, image, label, percentage):
        if not visualize:
            return

        desired_label = "8"

        if percentage < 0.1 and not self._image_one_saved:
            if label == desired_label:
                images.append(image)
                self._image_one_saved = True

        if 0.15 <= percentage <= 0.16 and not self._image_two_saved:
            if label == desired_label:
                images.append(image)
                self._image_two_saved = True

        if 0.28 <= percentage <= 0.29 and not self._image_three_saved:
            if label == desired_label:
                images.append(image)
                self._image_three_saved = True

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
    # mnist.generate("./datasets/mnist_grad_mniejszy.csv", drift_start=0.25, drift_end=0.30, visualize=False, line_number_to_end_at=24_000)
    # mnist.generate("./datasets/mnist_grad_powolny.csv", drift_start=0.10, drift_end=0.95, visualize=False, line_number_to_end_at=59_999)
    # mnist.generate_atnn_like("./datasets/mnist_grad_atnnowy_prosty.csv", drift_start_line_number=6_000, line_number_to_end_at=12_000)
    # mnist.generate("./datasets/mnist_grad_powolny_2x_szybszy.csv", drift_start=0.10, drift_end=0.45, visualize=False, line_number_to_end_at=29_999)
    # mnist.generate("./datasets/mnist_grad_powolny_4x_szybszy.csv", drift_start=0.10, drift_end=0.24, visualize=False, line_number_to_end_at=15_999)
    for drift_speed in [0.1, 0.5, 1, 2]:
        dataset_file = f"./datasets/fashion_inc_40k_{drift_speed}x.csv"
        mnist.generate(
            dataset_file,
            drift_start=15/60,
            drift_end=(15/60) + (15/60) / drift_speed,
            visualize=False,
            line_number_to_end_at=39_999
        )
        # encode classes
        with open(dataset_file.replace(".csv", ".txt"), "w") as f:
            for i in range(10):
                f.write(f"{i} {i}\n")

    # mnist.generate_atnn_like("./datasets/mnist_grad_atnnowy_prosty.csv", drift_start_line_number=6_000, line_number_to_end_at=12_000)

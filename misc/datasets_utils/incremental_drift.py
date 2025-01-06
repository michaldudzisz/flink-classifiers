import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def generate_incremental_drift_dataset(
        num_attributes=3,
        initial_means=None,
        std_dev=1.0,
        warmup_samples=1000,
        drift_samples=10000,
        drift_speed=0.1,
        output_file="incremental_drift_data.csv",
        plot_data=False,
        plot_data_static=False
):
    """
    Generates a dataset based on normal distributions to simulate incremental concept drift.

    Parameters:
        num_attributes (int): Number of attributes (x1, x2, x3, ...).
        initial_means (list or None): Initial mean values for each attribute. If None, defaults to 0 for all attributes.
        std_dev (float): Standard deviation for the normal distribution.
        drift_samples (int): Total number of samples to generate.
        drift_speed (float): Speed of change for the mean values over time.
        output_file (str): Path to save the dataset as a CSV file.
        plot_data (bool): Whether to plot the generated data.

    Returns:
        None
    """
    # Initialize parameters
    if initial_means is None:
        initial_means = [0] * num_attributes

    assert len(initial_means) == num_attributes, "Length of initial_means must match num_attributes"

    # Initialize dataset
    data = []
    samples = []
    labels = []

    # Current means of the distribution
    current_means = np.array(initial_means, dtype=float)

    # Generate warmup samples
    for i in range(warmup_samples):
        # Generate a sample from the current normal distribution
        sample = np.random.normal(loc=current_means, scale=std_dev, size=num_attributes)

        # Calculate the label (class)
        avg_sum = np.mean(current_means.sum())
        sample_sum = sample.sum()
        label = 1 if sample_sum > avg_sum else 0

        # flip label if needed
        k = 0.15
        flip_probability = 0.5 * (1 - np.minimum(1, (1.0 / (np.sqrt(num_attributes) * k * std_dev)) * np.abs(sample_sum - avg_sum)))
        if np.random.rand() < flip_probability:
            label = 1 - label

        # Append to dataset
        data.append(np.append(sample, label))
        samples.append(sample)
        labels.append(label)


    # Generate samples
    for i in range(drift_samples):
        # Generate a sample from the current normal distribution
        sample = np.random.normal(loc=current_means, scale=std_dev, size=num_attributes)

        # Calculate the label (class)
        avg_sum = np.mean(current_means.sum())
        sample_sum = sample.sum()
        label = 1 if sample_sum > avg_sum else 0

        # Append to dataset
        data.append(np.append(sample, label))
        samples.append(sample)
        labels.append(label)

        # Incrementally update the mean values
        current_means += drift_speed / drift_samples

    with open(output_file, "w") as f:
        for idx, sample in enumerate(samples):
            write_output_file_line(f, sample, labels[idx])

    # Convert dataset to a DataFrame
    column_names = [f"x{i+1}" for i in range(num_attributes)] + ["y"]
    df = pd.DataFrame(data, columns=column_names)

    # Save dataset to CSV with specified format
    # df.to_csv(output_file, sep='#', index=False, header=False, float_format='%.6f')

    # Save class encoding file
    with open(output_file.replace(".csv", ".txt"), "w") as f:
        f.write("0 0\n1 1")


    # Plot static data if requested
    if plot_data_static:
        plt.figure(figsize=(5, 5))
        plt.scatter(
            df["x1"],
            df["x2"],
            c=df["y"].map({0: 'blue', 1: 'red'}),
            s=5,
            alpha=0.5,
            label="xd"
        )
        class_0_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='klasa 0')
        class_1_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='klasa 1')
        # Dodanie legendy do wykresu
        plt.legend(handles=[class_0_legend, class_1_legend], loc='upper right')
        # plt.title(r"Rozkład próbek pojęcia $C_t$ w przestrzeni atrybutów $x_1$ i $x_2$")
        plt.xlabel(r"$x_1$")
        plt.ylabel(f"$x_2$")
        plt.tight_layout()
        plt.show()



    # Plot data if requested
    if plot_data:
        plt.figure(figsize=(12, 6))
        for i in range(num_attributes):
            plt.subplot(1, num_attributes, i + 1)
            plt.scatter(
                range(warmup_samples + drift_samples),
                df[f"x{i+1}"],
                c=df["y"].map({0: 'blue', 1: 'red'}),
                s=5,
                alpha=0.5,
                label=f"x{i+1}"
            )
            class_0_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='klasa 0')
            class_1_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='klasa 1')
            # Dodanie legendy do wykresu
            plt.legend(handles=[class_0_legend, class_1_legend], loc='upper right')
            plt.title(f"Attribute x{i+1}")
            plt.xlabel("Sample Index")
            plt.ylabel(f"x{i+1} Value")
        plt.tight_layout()
        plt.show()

def write_output_file_line(file, attributes, label):
    file.write(f"{attributes[0]}")
    for i in range(1, len(attributes)):
        file.write(f"#{attributes[i]}")
    file.write(f",{label}\n")

# Example usage
num_attributes = 2
std_dev = 0.2
warmup_samples = 2_000
drift_samples = 18_000
drift_speed = 10.0
generate_incremental_drift_dataset(
    num_attributes=num_attributes,
    std_dev=std_dev,
    warmup_samples=warmup_samples,
    drift_samples=drift_samples,
    drift_speed=drift_speed,
    output_file=f"./datasets/incremental_drift_synth_attr{num_attributes}_speed{drift_speed}_len{warmup_samples + drift_samples}.csv",
    plot_data_static=False,
    plot_data=False
)

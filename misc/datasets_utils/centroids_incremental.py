import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generate_uniform_drift_data(n_samples=20000, n_features=2, n_classes=10, warm_up=2000, drift_magnitude=0.0001,
                                std_dev=0.5, random_state=42, output_file="output.csv"):
    """
    Funkcja generująca dane z dryfem jednostajnym centroidów klas z modyfikowanym odchyleniem standardowym.

    Parametry:
    - n_samples: całkowita liczba próbek (łącznie warm-up i próbek z dryfem)
    - n_features: liczba cech
    - n_classes: liczba klas
    - warm_up: liczba próbek bez dryfu (warm-up)
    - drift_magnitude: wielkość dryfu centroidów na próbkę
    - std_dev: odchylenie standardowe próbek generowanych wokół centroidów
    - random_state: seed dla generatora losowego

    Zwraca:
    - data: macierz danych (n_samples x n_features)
    - labels: etykiety klas (n_samples,)
    """
    np.random.seed(random_state)

    samples_per_class = n_samples // n_classes
    start_centroids = np.random.uniform(-1, 1, size=(n_classes, n_features))
    centroids = start_centroids.copy()
    end_centroids = np.random.uniform(-1, 1, size=(n_classes, n_features))

    data = []
    labels = []

    drift_direction = np.random.uniform(-1, 1, size=(n_classes, n_features))
    # drift_direction /= np.linalg.norm(drift_direction)
    for i in range(n_samples):
        label = np.random.randint(0, n_classes)
        centroid = centroids[label]
        sample = np.random.normal(loc=centroid, scale=std_dev, size=n_features)

        data.append(sample)
        labels.append(label)

        if i >= warm_up:
            # centroids += drift_magnitude * drift_direction
            centroids += drift_magnitude * (end_centroids - start_centroids) / (n_samples - warm_up)
            # if i % 1000 == 0:
            #     print(drift_magnitude * (end_centroids - start_centroids) / (n_samples - warm_up))

    data = np.array(data)
    labels = np.array(labels)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)


    with open(output_file, "w") as f:
        for idx, sample in enumerate(data):
            write_output_file_line(f, sample, labels[idx])

    # Save class encoding file
    with open(output_file.replace(".csv", ".txt"), "w") as f:
        for i in range(n_classes):
            f.write(f"{i} {i}\n")



    return data, labels


def write_output_file_line(file, attributes, label):
    file.write(f"{attributes[0]}")
    for i in range(1, len(attributes)):
        file.write(f"#{attributes[i]}")
    file.write(f",{label}\n")



# Generowanie danych
n_attributes = 2
n_classes = 5
dmag = 1.0
std_dev=0.2
data, labels = generate_uniform_drift_data(
    warm_up=2000,
    n_samples=10000,
    n_features=n_attributes,
    n_classes=n_classes,
    drift_magnitude=dmag,
    std_dev=std_dev,
    output_file=f"./datasets/test_attr{n_attributes}_class{n_classes}_dmag{dmag}.csv",
)







plot_data = True
if plot_data:
    if data.shape[1] != 2:
        raise ValueError("Data must have 2 features for visualization.")
    # Wybór pierwszych 1000 próbek i ostatnich 1000 próbek
    data_first_1000 = data[:1000]
    labels_first_1000 = labels[:1000]

    data_last_1000 = data[-1000:]
    labels_last_1000 = labels[-1000:]

    # Wykres pierwszych 1000 próbek
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data_first_1000[:, 0], data_first_1000[:, 1], c=labels_first_1000, cmap='tab10', s=15, alpha=0.7)
    plt.title("Pierwsze 1000 próbek")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Wykres ostatnich 1000 próbek
    plt.subplot(1, 2, 2)
    plt.scatter(data_last_1000[:, 0], data_last_1000[:, 1], c=labels_last_1000, cmap='tab10', s=15, alpha=0.7)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title("Ostatnie 1000 próbek")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.tight_layout()
    plt.show()

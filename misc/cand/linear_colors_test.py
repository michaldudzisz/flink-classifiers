import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

num_lines = 30

# Tworzenie przykładowych danych
x = np.linspace(0, 10, 100)
y = [x+i for i in range(num_lines)]  # num_lines różnych linii

# Tworzenie skali kolorów od niebieskiego do żółtego
dark_color = "#216e23"
light_color = "#f7ebd0"

color_gradient = [
    (0.0, "#173c87"),
    (0.1, "#3abd9e"),
    (0.3, "#edd78e"),
    (0.7, "#f5ecb8"),
    (1.0, "#f2b996"),
]

cmap = LinearSegmentedColormap.from_list("blue_to_yellow", color_gradient)

# Normalizacja dla 5 linii (wartości od 0 do 1)
colors = [cmap(i / num_lines) for i in range(num_lines)]  # 4 to max indeks (5 linii -> 0, 0.25, 0.5, 0.75, 1.0)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    plt.plot(x, y[i], label=f'Line {i+1}', color=colors[i], linewidth=2)

plt.title("Example plot with LinearSegmentedColormap")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

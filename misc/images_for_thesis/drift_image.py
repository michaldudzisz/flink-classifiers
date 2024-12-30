import matplotlib.pyplot as plt
import numpy as np
import math

# Function to draw a drift chart
def draw_drift_chart(ax, x_positions, y_values, colors, title):
    ax.scatter(x_positions, y_values, c=colors, s=200)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.set_yticks([])
    ax.set_xlabel("Czas")
    ax.set_ylabel("Wartość średnia")
    ax.annotate("", xy=(15, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))  # X-axis arrow
    ax.annotate("", xy=(0, 5), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))  # Y-axis arrow
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Keep left and bottom spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# Set up the figure with subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)


blue = "#057df5"
red = "#f21b34"

# Define the data for each type of drift
x_sudden = np.arange(1, 15)
y_sudden = [1] * (int(len(x_sudden)/2)) + [4] * int(len(x_sudden)/2)
colors_sudden = [blue] * (int(len(x_sudden)/2)) + [red] * int(len(x_sudden)/2)
x_sudden_full = x_sudden
y_sudden_full = y_sudden




x_gradual = np.arange(1, 15)
y_gradual = [1, 1, 1, 4, 1, 1, 4, 1, 4, 4, 1, 4, 4, 4]
colors_gradual = [blue] * 3 + [red] + [blue] * 2 + [red] + [blue] + [red] * 2 + [blue] + [red] * 3





x_incremental_1 = list(np.arange(1, 3))
y_incremental_1 = [1] * len(x_incremental_1)

x_incremental_2 = list(np.arange(3, 13))
x_incremental_idxes = np.arange(0, 10)
y_incremental_2 = list(map(lambda x: math.sin((x/20 - 0.25)*2*math.pi)*1.5 + 2.5, x_incremental_idxes))

x_incremental_3 = list(np.arange(13, 15))
y_incremental_3 = [4] * len(x_incremental_3)
colors_incremental = [
    blue,
    blue,
    blue,
    "#0595f5",
    "#2fddf7",
    "#2ff7d3",
    "#39f72f",
    "#a7f72f",
    "#eff551",
    "#f7e02f",
    "#f7932f",
    red,
    red,
    red
]

x_incremental = x_incremental_1 + x_incremental_2 + x_incremental_3
y_incremental = y_incremental_1 + y_incremental_2 + y_incremental_3



# Define the data for each type of drift
x_reoccurring = np.arange(1, 15)
y_reoccurring = [1] * 4 + [4] * 6 + [1] * 4
colors_reoccurring = [blue] * 4 + [red] * 6 + [blue] * 4



font = {'family' : 'serif',
        'size'   : 12,
        'serif':  'cmr10'
        }

plt.rc('font', **font)



# Draw each drift type as a separate subplot
draw_drift_chart(axs[0], x_sudden_full, y_sudden_full, colors_sudden, "Nagły dryf pojęć")
draw_drift_chart(axs[1], x_gradual, y_gradual, colors_gradual, "Stopniowy dryf pojęć")
draw_drift_chart(axs[2], x_incremental, y_incremental, colors_incremental, "Inkrementalny dryf pojęć")
draw_drift_chart(axs[3], x_reoccurring, y_reoccurring, colors_reoccurring, "Powracające pojęcie")

# Show the plot
plt.show()

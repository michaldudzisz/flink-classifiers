import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


file_path1 = './datasets/incremental_drift_synth_attr2_speed0.5_len20000.csv'
file_path2 = './datasets/incremental_drift_synth_attr2_speed2.0_len20000.csv'

df1 = pd.read_csv(file_path1, sep='#|,', engine='python', names=['x1', 'x2', 'y'])
df2 = pd.read_csv(file_path2, sep='#|,', engine='python', names=['x1', 'x2', 'y'])


fig, axes = plt.subplots(2, 2, figsize=(8, 8))

fig.text(0.5, 0.96, r'dryf inkrementalny, $\mu_\text{końc} = 0,5$', ha='center', va='center', fontsize=12)
fig.text(0.5, 0.48, r'dryf inkrementalny, $\mu_\text{końc} = 2$', ha='center', va='center', fontsize=12)
# Wykresy dla df1
axes[0, 0].scatter(df1.index, df1['x1'],
                   c=df1["y"].map({0: 'blue', 1: 'red'}),
                   s=2,
                   alpha=0.5,
                   label=r"$x_1$")
# axes[0, 0].set_title(r"Rozkład $x_1$")
axes[0, 0].set_xlabel(r'$t$')
axes[0, 0].set_ylabel(r'$x_1$')
axes[0, 0].set_ylim([-0.7, 2.5])

axes[0, 1].scatter(df1.index, df1['x2'],
                   c=df1["y"].map({0: 'blue', 1: 'red'}),
                   s=2,
                   alpha=0.5,
                   label=r"$x_2$ vs Index")
# axes[0, 1].set_title('DF1: x2 vs Index')
axes[0, 1].set_xlabel(r'$t$')
axes[0, 1].set_ylabel(r'$x_2$')
axes[0, 1].set_ylim([-0.7, 2.5])

# Wykresy dla df2
axes[1, 0].scatter(df2.index, df2['x1'],
                   c=df2["y"].map({0: 'blue', 1: 'red'}),
                   s=2,
                   alpha=0.5,
                   label=r"$x_1$ vs Index")
# axes[1, 0].set_title('DF2: x1 vs Index')
axes[1, 0].set_xlabel(r'$t$')
axes[1, 0].set_ylabel(r'$x_1$')
axes[1, 0].set_ylim([-0.7, 2.5])

axes[1, 1].scatter(df2.index, df2['x2'],
                   c=df2["y"].map({0: 'blue', 1: 'red'}),
                   s=2,
                   alpha=0.5,
                   label=r"$x_2$")
# axes[1, 1].set_title('DF2: x2 vs Index')
axes[1, 1].set_xlabel(r'$t$')
axes[1, 1].set_ylabel(r"$x_2$")
axes[1, 1].set_ylim([-0.7, 2.5])

class_0_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='klasa 0')
class_1_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='klasa 1')
# Dodanie legendy do wykresu
plt.legend(handles=[class_0_legend, class_1_legend], loc='lower right')

plt.subplots_adjust(top=0.5)

plt.tight_layout()
plt.show()
#
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.scatter(
#         range(warmup_samples + drift_samples),
#         df[f"x{i+1}"],
#         c=df["y"].map({0: 'blue', 1: 'red'}),
#         s=5,
#         alpha=0.5,
#         label=f"x{i+1}"
#     )
#     class_0_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='klasa 0')
#     class_1_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='klasa 1')
#     # Dodanie legendy do wykresu
#     plt.legend(handles=[class_0_legend, class_1_legend], loc='upper right')
#     plt.title(f"Attribute x{i+1}")
#     plt.xlabel("Sample Index")
#     plt.ylabel(f"x{i+1} Value")
# plt.tight_layout()
# plt.show()
#

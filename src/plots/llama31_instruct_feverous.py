import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize


set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
        1.00e-05,
        5.00e-06,
        1.00e-06,
        5.00e-07,
        1.00e-07,
    ],
    "example_num": [
        30,
        30,
        30,
        30,
        30,
        90,
        90,
        90,
        90,
        90,
        150,
        150,
        150,
        150,
        150,
        300,
        300,
        300,
        300,
        300,
        600,
        600,
        600,
        600,
        600,
        1500,
        1500,
        1500,
        1500,
        1500,
    ],
    "performance": [
        54.63243874,
        69.35822637,
        58.6697783,
        5.670945158,
        0.186697783,
        55.3092182,
        71.10851809,
        70.54842474,
        69.05484247,
        0.7934655776,
        53.4655776,
        64.48074679,
        73.23220537,
        69.54492415,
        7.677946324,
        49.91831972,
        70.80513419,
        71.24854142,
        68.68144691,
        63.59393232,
        66.88448075,
        74.02567095,
        74.56242707,
        73.81563594,
        68.05134189,
        71.80863477,
        71.69194866,
        74.63243874,
        74.53908985,
        70.50175029,
    ],
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

markers = ["o", "s", "D", "^", "*"]

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df["learning_rate"].unique()):
    subset = df[df["learning_rate"] == lr]
    plt.plot(
        subset["example_num"],
        subset["performance"],
        marker=markers[i],
        markersize=10,
        label=f"{lr:.1e}",
    )

    if lr == 1.00e-06:
        plt.text(
            subset["example_num"].iloc[-1],
            subset["performance"].iloc[-1] + 1,
            f"{subset['performance'].iloc[-1]:.2f}",
            fontsize=NumberFontSize,
            ha="center",
        )
# plt.xlabel('Steps')x f

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_feverous.pdf")

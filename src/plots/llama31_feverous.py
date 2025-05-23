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
        41.19019837,
        42.61376896,
        6.721120187,
        0.5834305718,
        0,
        49.00816803,
        46.76779463,
        50.36172695,
        22.73045508,
        0,
        53.02217036,
        57.59626604,
        49.52158693,
        54.81913652,
        0.02333722287,
        51.80863477,
        47.11785298,
        48.68144691,
        54.00233372,
        8.541423571,
        50.82847141,
        67.72462077,
        67.72462077,
        63.47724621,
        48.23803967,
        62.31038506,
        71.85530922,
        70.9218203,
        63.29054842,
        58.85647608,
    ],
}

markers = ["o", "s", "D", "^", "*"]
# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

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

    if lr == 5.00e-06:
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
plt.savefig("../figures/llama31_feverous.pdf")

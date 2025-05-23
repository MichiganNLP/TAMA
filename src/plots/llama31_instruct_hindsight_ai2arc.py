import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import (
    set_parameters,
    NumberFontSize,
    plot_details,
    FigSize,
    markers,
    MARKSIZE,
)


set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00e-05] * 6
    + [5.00e-06] * 6
    + [1.00e-06] * 6
    + [5.00e-07] * 6
    + [1.00e-07] * 6,
    "epochs": [1, 2, 3, 4, 5, 6] * 5,
    "performance": [
        73.72,
        70.31,
        69.54,
        66.38,
        66.3,
        67.15,
        79.35,
        79.01,
        78.16,
        77.99,
        78.33,
        78.41,
        82.08,
        81.23,
        81.74,
        81.66,
        81.74,
        81.48,
        81.06,
        81.48,
        81.31,
        81.31,
        81.06,
        81.4,
        80.72,
        80.89,
        80.89,
        80.97,
        80.89,
        80.97,
    ],
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df["learning_rate"].unique()):
    subset = df[df["learning_rate"] == lr]
    plt.plot(
        subset["epochs"],
        subset["performance"],
        marker=markers[i],
        markersize=MARKSIZE,
        label=f"{lr:.1e}",
    )

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_hindsight_ai2arc.pdf")

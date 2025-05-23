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
        27.01,
        28.79,
        32.14,
        25.89,
        29.91,
        28.79,
        28.35,
        29.46,
        28.79,
        28.12,
        27.68,
        28.35,
        31.47,
        31.92,
        32.81,
        32.14,
        32.59,
        32.14,
        31.7,
        31.92,
        31.7,
        31.7,
        31.7,
        31.7,
        32.37,
        32.14,
        31.47,
        31.7,
        31.7,
        31.47,
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
plt.savefig("../figures/llama31_instruct_hindsight_gpqa.pdf")

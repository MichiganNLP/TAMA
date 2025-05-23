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
        15.55,
        18.72,
        17.58,
        17.94,
        19.05,
        18.98,
        19.49,
        20.49,
        21.80,
        23.07,
        23.16,
        22.35,
        27.81,
        28.60,
        29.07,
        28.95,
        27.89,
        27.95,
        23.25,
        23.64,
        23.28,
        22.93,
        22.49,
        22.33,
        19.89,
        19.94,
        20.16,
        20.40,
        20.19,
        20.14,
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
plt.savefig("../figures/llama31_instruct_hindsight_s2.pdf")

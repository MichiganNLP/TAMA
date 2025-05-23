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
        26.38,
        26.69,
        26.01,
        25.03,
        24.12,
        24.5,
        34.7,
        35.52,
        33,
        33.93,
        33.68,
        33.76,
        26.13,
        31.84,
        32.52,
        33.52,
        34.65,
        34.67,
        21.88,
        25.35,
        27.16,
        28.42,
        28.82,
        28.93,
        22.02,
        22.05,
        21.99,
        21.98,
        22.09,
        22.13,
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
plt.savefig("../figures/llama31_instruct_hindsight_mmlu_pro.pdf")

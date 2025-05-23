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
        61.25,
        62.18,
        62,
        62.05,
        62.04,
        61.31,
        61.71,
        62,
        61.99,
        61.98,
        62.85,
        62.39,
        62.07,
        62.03,
        62.11,
        57.84,
        60.61,
        62.49,
        62.26,
        62.12,
        53.97,
        63.08,
        62.68,
        62.48,
        62.07,
        52.29,
        61.64,
        64.63,
        62.98,
        62.09,
    ],
}

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
        markersize=MARKSIZE,
        label=f"{lr:.1e}",
    )


plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_mmlu.pdf")

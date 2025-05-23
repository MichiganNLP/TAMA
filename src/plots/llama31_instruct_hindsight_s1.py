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
        0.001851166235,
        0.003702332469,
        0.01295816364,
        0.3332099223,
        0.1851166235,
        0.394298408,
        6.003332099,
        0.4924102184,
        1.432802666,
        0.875601629,
        1.360607183,
        0.9922251018,
        64.54091077,
        65.05553499,
        52.56201407,
        46.7160311,
        45.44798223,
        43.80970011,
        64.83154387,
        65.21658645,
        65.32765642,
        65.20177712,
        65.24990744,
        65.12958164,
        18.41355054,
        20.24805628,
        20.44242873,
        20.68122917,
        20.66086635,
        20.74046649,
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
plt.savefig("../figures/llama31_instruct_hindsight_s1.pdf")

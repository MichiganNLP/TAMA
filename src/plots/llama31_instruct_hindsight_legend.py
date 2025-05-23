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
        35.85,
        35.25,
        37.05,
        38.97,
        38.25,
        38.13,
        38.01,
        44.12,
        45.56,
        45.56,
        45.68,
        45.56,
        75.66,
        74.70,
        75.06,
        72.42,
        74.10,
        73.98,
        77.46,
        78.66,
        77.22,
        77.46,
        76.62,
        76.98,
        80.70,
        81.06,
        80.22,
        80.70,
        80.58,
        80.10,
    ],
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

FigSize = (20, 8)
# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df["learning_rate"].unique()):
    subset = df[df["learning_rate"] == lr]
    plt.plot([], [], marker=markers[i], markersize=MARKSIZE, label=f"{lr:.1e}")


plot_details(df=df)
plt.gca().set_axis_off()

plt.legend(ncol=5)

# Display the plot
plt.savefig("../figures/llama31_instruct_hindsight_legends.pdf")


# # Example data for the legend
# labels = ['Class A', 'Class B', 'Class C']
# colors = ['red', 'green', 'blue']

# # Create an invisible plot
# fig, ax = plt.subplots()

# # Add dummy points with labels for the legend
# for label, color in zip(labels, colors):
#     ax.plot([], [], label=label, color=color)

# # Display only the legend
# ax.legend()

# # Hide the axes
# ax.axis('off')

# # Show the plot

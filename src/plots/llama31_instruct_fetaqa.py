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
        16.90968812,
        8.644346023,
        23.72080884,
        16.47135193,
        14.8697894,
        31.50614845,
        31.97094418,
        14.73879426,
        21.52546577,
        15.18747714,
        33.72104556,
        33.99273832,
        28.59787509,
        18.57094492,
        16.16356253,
        32.67385335,
        36.31916413,
        32.63765587,
        28.32618489,
        23.39735071,
        33.71476053,
        36.05466798,
        35.50259953,
        31.20397263,
        18.80990432,
        34.31515858,
        36.25676218,
        35.22894017,
        34.20154977,
        27.1473162,
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
plt.savefig("../figures/llama31_instruct_fetaqa.pdf")

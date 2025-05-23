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
        553.2832567,
        611.7336625,
        648.1961717,
        661.8553976,
        674.3793091,
        677.1597913,
        641.3599561,
        727.3013852,
        727.6073145,
        740.5969001,
        746.6749825,
        747.1483644,
        769.4338198,
        781.311316,
        796.541163,
        795.2967795,
        793.0492035,
        789.6560831,
        747.1347742,
        773.5034595,
        783.194248,
        779.9832575,
        782.840531,
        784.9473574,
        629.1384836,
        637.7781263,
        669.2377702,
        681.6687433,
        688.6979985,
        689.611277,
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
plt.savefig("../figures/llama31_instruct_hindsight_all_table_tasks.pdf")

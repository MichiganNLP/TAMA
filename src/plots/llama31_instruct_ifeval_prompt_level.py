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
        62.10720887,
        67.65249538,
        70.7948244,
        70.7948244,
        71.34935305,
        40.29574861,
        65.43438078,
        71.71903882,
        71.71903882,
        70.60998152,
        29.57486137,
        58.04066543,
        70.60998152,
        71.16451017,
        70.97966728,
        22.36598891,
        51.94085028,
        70.60998152,
        70.97966728,
        71.53419593,
        20.88724584,
        57.48613678,
        70.97966728,
        71.16451017,
        69.50092421,
        21.44177449,
        35.85951941,
        65.06469501,
        70.97966728,
        72.08872458,
    ],
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Plot the results
plt.figure(figsize=FigSize)

for lr in df["learning_rate"].unique():
    subset = df[df["learning_rate"] == lr]
    plt.plot(
        subset["example_num"], subset["performance"], marker="o", label=f"{lr:.1e}"
    )

# plt.xlabel('Steps')x f

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_ifeval_prompt_level.pdf")

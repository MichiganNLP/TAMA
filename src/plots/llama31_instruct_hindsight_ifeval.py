import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize, markers, MARKSIZE



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05] * 6 + [5.00E-06] * 6 + [1.00E-06] * 6 + [5.00E-07] * 6 + [1.00E-07] * 6,
    "epochs": [1, 2, 3, 4, 5, 6] * 5,
    "performance": [35.85,35.25,37.05,38.97,38.25,38.13,38.01,44.12,45.56,45.56,45.68,45.56,75.66,74.70,75.06,72.42,74.10,73.98,77.46,78.66,77.22,77.46,76.62,76.98,80.70,81.06,80.22,80.70,80.58,80.10]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['epochs'], subset['performance'], marker=markers[i], markersize=MARKSIZE, label=f'{lr:.1e}')

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_hindsight_ifeval.pdf")

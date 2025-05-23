import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize, markers, MARKSIZE



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05] * 6 + [5.00E-06] * 6 + [1.00E-06] * 6 + [5.00E-07] * 6 + [1.00E-07] * 6,
    "epochs": [1, 2, 3, 4, 5, 6] * 5,
    "performance": [59.09,57.46,55.39,55.43,55.06,55.28,65.30,65.67,64.64,64.95,64.86,64.66,66.09,66.99,67.11,67.48,67.37,67.29,65.80,66.58,66.81,66.88,67.06,66.98,65.92,65.68,65.82,65.80,65.77,65.80]
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
plt.savefig("../figures/llama31_instruct_hindsight_mmlu.pdf")

import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [66.32,66.69,66.1,66.08,66.05,64.62,68,66.11,66.08,66.07,65.19,67.21,66.42,66.16,66.21,59.55,67.15,66.61,66.04,66.11,59.78,66.29,66.65,66.2,66.04,54.91,64.82,66.9,66.36,65.95]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

markers = ['o', 's', 'D', '^', '*']

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['example_num'], subset['performance'], marker=markers[i], markersize=10, label=f'{lr:.1e}')


plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_mmlu.pdf")

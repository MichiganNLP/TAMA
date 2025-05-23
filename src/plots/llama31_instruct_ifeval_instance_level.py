import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [72.90167866,76.85851319,79.3764988,79.13669065,79.73621103,51.43884892,74.82014388,80.21582734,79.49640288,78.77697842,39.80815348,68.34532374,78.65707434,79.01678657,78.89688249,36.33093525,63.42925659,78.65707434,79.13669065,79.61630695,32.85371703,67.0263789,78.77697842,79.49640288,78.17745803,31.05515588,47.12230216,73.86091127,78.41726619,79.97601918]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

markers = ['o', 's', 'D', '^', '*']

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['example_num'], subset['performance'], marker=markers[i], markersize=10, label=f'{lr:.1e}')

# plt.xlabel('Steps')x f

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_ifeval_instance_level.pdf")

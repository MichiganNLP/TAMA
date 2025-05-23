import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize, markers, MARKSIZE



set_parameters()

# Data: learning rate, number of tuning steps, performance


# llama31 tabfact
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [54.83214649,59.23781204,40.01878081,0,0,57.37538149,67.54049613,64.7468503,63.02527584,0,51.42812427,66.4762501,69.4890054,65.09116519,0,51.20901479,70.96799437,68.40128336,63.27568667,48.95531732,68.46388606,71.63314813,72.72087018,71.94616167,66.73448627,70.10720714,73.09648642,74.34071524,72.93997965,66.85969168]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['example_num'], subset['performance'], marker=markers[i], markersize=MARKSIZE, label=f'{lr:.1e}')

    if lr == 5.00E-06:
        plt.text(subset['example_num'].iloc[-1], subset['performance'].iloc[-1] + 1, f"{subset['performance'].iloc[-1]:.2f}", fontsize=NumberFontSize, ha='center')
# plt.xlabel('Steps')
# plt.ylabel('Acc')

plot_details(df=df)

# Display the plot
plt.savefig("../figures/llama31_instruct_tabfact.pdf")

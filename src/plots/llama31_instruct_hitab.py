import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [28.09343434,49.49494949,35.29040404,2.777777778,0,36.93181818,57.63888889,54.22979798,44.19191919,0.06313131313,36.80555556,54.54545455,58.01767677,53.40909091,1.957070707,36.23737374,57.26010101,59.53282828,55.99747475,36.23737374,53.15656566,64.14141414,62.81565657,61.17424242,45.2020202,48.80050505,60.29040404,66.28787879,62.43686869,52.46212121]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

markers = ['o', 's', 'D', '^', '*']

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['example_num'], subset['performance'], marker=markers[i], markersize=10, label=f'{lr:.1e}')

    if lr == 1.00E-06:
        plt.text(subset['example_num'].iloc[-1], subset['performance'].iloc[-1] + 1, f"{subset['performance'].iloc[-1]:.2f}", fontsize=NumberFontSize, ha='center')
# plt.xlabel('Steps')x f

plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_instruct_hitab.pdf")

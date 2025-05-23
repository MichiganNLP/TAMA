import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize, markers, MARKSIZE



set_parameters()

# Data: learning rate, number of tuning steps, performance


# llama31 tabfact
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [53.25925346,52.71930511,10.14946396,0.3130135378,0,50.38735425,50.07434072,53.00884263,31.40308318,0,51.50637765,56.78065576,53.23577745,52.64105173,0,50.16041944,50.16041944,52.52367165,56.23288207,15.01682448,65.85022302,69.16034118,67.23530793,64.46513812,51.78808983,64.52774082,71.10102512,71.02277173,69.56725878,57.93880585]
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
plt.savefig("../figures/llama31_tabfact.pdf")

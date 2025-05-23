import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize



set_parameters()

# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07,1.00E-05,5.00E-06,1.00E-06,5.00E-07,1.00E-07],
    "example_num": [30,30,30,30,30,90,90,90,90,90,150,150,150,150,150,300,300,300,300,300,600,600,600,600,600,1500,1500,1500,1500,1500],
    "performance": [35.85131894,34.17266187,30.69544365,32.25419664,31.05515588,24.2206235,28.53717026,35.13189448,32.73381295,29.97601918,21.82254197,25.7793765,34.29256595,33.33333333,32.37410072,23.86091127,27.69784173,32.13429257,33.69304556,32.13429257,26.61870504,29.97601918,33.09352518,31.77458034,32.85371703,23.26139089,29.73621103,36.33093525,29.97601918,35.13189448]
}

markers = ['o', 's', 'D', '^', '*']
# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Plot the results
plt.figure(figsize=FigSize)

for i, lr in enumerate(df['learning_rate'].unique()):
    subset = df[df['learning_rate'] == lr]
    plt.plot(subset['example_num'], subset['performance'], marker=markers[i], markersize=10, label=f'{lr:.1e}')


plot_details(df=df)
# Display the plot
plt.savefig("../figures/llama31_ifeval_instance_level.pdf")

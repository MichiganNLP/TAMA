import matplotlib.pyplot as plt
import pandas as pd
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize, MARKSIZE



# Data: learning rate, number of tuning steps, performance

# llama31 feverous
data = {
    "learning_rate": [1.00E-06] * 6 + [5.00E-07] * 6,
    "epochs": [1,2,3,4,5,6] * 2,
    "hitab_performance": [62.75252525,66.03535354,64.52020202,65.34090909,64.77272727,64.58333333,57.89141414,61.80555556,62.81565657,62.24747475,61.61616162,61.36363636],
    "fetaqa_performance": [31.96905707,35.13878133,36.73397102,34.73414757,35.69333245,34.7266535,29.19074837,32.33208479,33.8563837,33.0701473,34.52604149,34.74192451],
    "tabfact_performance": [70.19328586,73.69121214,73.42515064,72.91650364,72.97910635,72.87737695,68.84732765,70.37326864,70.16198451,71.49229204,71.7270522,71.86790829],
    "feverous_performance": [72.5554259,73.41890315,74.4924154,73.69894982,73.4655776,73.23220537,71.24854142,71.57526254,73.69894982,73.6756126,73.60560093,73.74562427],
    "ifeval_performance": [75.05995204,77.57793765,74.94004796,73.98081535,74.58033573,74.2206235,77.8177458,78.29736211,78.65707434,78.41726619,77.45803357,79.3764988],
    "mmlu_performance": [66.43,66.62,66.9,66.88,66.98,66.92,66.04,66.31,66.44,66.53,66.41,66.55]
}

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

for lr in df['learning_rate'].unique():
    set_parameters()
    # Plot the results
    plt.figure(figsize=FigSize)
    subset = df[df['learning_rate'] == lr]
    plt.plot([], [], marker='o', markersize=MARKSIZE, label=f'HiTab')
    plt.plot([], [], marker='s', markersize=MARKSIZE, label=f'FeTaQA')
    plt.plot([], [], marker='^', markersize=MARKSIZE, label=f'TabFact')
    plt.plot([], [], marker='v', markersize=MARKSIZE, label=f'FEVEROUS')
    plt.plot([], [], marker='D', markersize=MARKSIZE, label=f'IFEval')
    plt.plot([], [], marker='*', markersize=MARKSIZE, label=f'MMLU')

    # plt.xlabel('Steps')x f

    plot_details(df=df)
    plt.gca().set_axis_off()
    plt.legend(ncol=3)
    # Display the plot
    plt.savefig(f"../figures/llama31_instruct_lr_{lr}_legend.pdf")
    plt.close()

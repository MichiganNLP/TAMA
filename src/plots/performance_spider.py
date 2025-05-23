import numpy as np
import matplotlib.pyplot as plt
from math import pi
from plots.utils.utils import set_parameters, NumberFontSize, plot_details, FigSize


set_parameters()

# Step 1: Define the models and their performance on 5 benchmarks
models = ["TableLlama", "TableLLM", "TableBenchLLM", "TAMA"]  # Names of the models
benchmarks = ["Table-Syn", "IFEval", "MMLU", "MMLU-Pro", "AI2ARC", "GPQA"]  # Benchmarks

# Performance data for each model on the benchmarks (example data)
performance = {
    "TableLlama": [0, 25.78, 30.27, 12.33, 30.89, 23.44],
    "TableLLM": [18.4, 30.46, 35.9, 15.36, 34.81, 24.11],
    "TableBenchLLM": [9, 32.85, 52.67, 17.84, 53.5, 27.01],
    # "L-Instruct": [53.6,79.62,66.04,22.1,80.89,32.14],
    "TAMA (ours)": [64.93, 74.7, 66.99, 31.84, 81.23, 31.92],
}

# Number of variables (benchmarks)
num_benchmarks = len(benchmarks)

# Step 2: Create a radar plot (spider plot) for each model
angles = [n / float(num_benchmarks) * 2 * pi for n in range(num_benchmarks)]
angles += angles[:1]  # Complete the circle by repeating the first angle

fig, ax = plt.subplots(figsize=(15, 8), subplot_kw=dict(polar=True))

# Step 3: Plot each model's performance
for model, perf in performance.items():
    perf += perf[:1]  # Repeat the first value to close the radar chart
    ax.plot(
        angles, perf, linewidth=2, linestyle="solid", label=model
    )  # Draw the radar chart
    ax.fill(angles, perf, alpha=0.25)  # Fill the area under the plot


# Step 4: Customize the plot
ax.set_xticks(angles[:-1])

ax.set_xticklabels(benchmarks, size=20)
ax.tick_params(axis="x", pad=20)

# # Step 1: Adjust the padding for specific tick labels (e.g., Benchmark 1 and Benchmark 3)
tick_labels = ax.get_xticklabels()

# Apply padding to specific labels (e.g., "MMLU-Pro" and "Table-Syn")
for label in tick_labels:
    if label.get_text() == "MMLU-Pro":
        label.set_position(
            (label.get_position()[0], label.get_position()[1] - 0.15)
        )  # Move down
    elif label.get_text() == "Table-Syn":
        label.set_position(
            (label.get_position()[0], label.get_position()[1] - 0.15)
        )  # Move up


# # You can modify the specific ticks by index
# # For example, to set more padding for 'Benchmark 1' (index 0) and 'Benchmark 3' (index 2):
# tick_labels[0].set_position((tick_labels[0].get_position()[0] * 20, tick_labels[0].get_position()[1] * 20))  # Adjust padding for Benchmark 1
# tick_labels[2].set_position((tick_labels[2].get_position()[0] * 20, tick_labels[2].get_position()[1] * 20))  # Adjust padding for Benchmark 3


# Set the range for the radar chart (optional, depending on your data scale)
ax.set_ylim(0, 100)  # Assuming performance scores are between 0 and 1

# Add a title and legend
# plt.legend(loc='upper right', bbox_to_anchor=(2.3, 1.1))

# Show the plot
plt.savefig("../figures/performance_spider.pdf")

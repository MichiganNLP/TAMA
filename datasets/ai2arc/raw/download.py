from datasets import load_dataset

dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")

dataset["test"].to_csv("test.csv")  # For the training set

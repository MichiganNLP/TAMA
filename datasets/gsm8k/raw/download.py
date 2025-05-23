from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", 'main')

dataset["train"].to_csv("train.csv")  # For the training set
dataset["test"].to_csv("test.csv")

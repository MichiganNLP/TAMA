from datasets import load_dataset

dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")

dataset["train"].to_csv("test.csv")  # For the training set
# dataset["test"].to_csv("test.csv")

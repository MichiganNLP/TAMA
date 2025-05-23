import os
import json
from loguru import logger
from typing import List
import csv
import pandas as pd


root_dir = "../datasets/gsm8k"


def process() -> List:

    # Read the CSV file, ensuring proper parsing of multiline answers
    df = pd.read_csv(f"{root_dir}/raw/test.csv", quoting=csv.QUOTE_ALL)

    processed = []
    for _, row in df.iterrows():
        instruction = "You are helpful agent."
        input = row["question"]
        full_output = row["answer"]
        output = row["answer"].split("####")[-1].strip()
        processed.append(
            {
                "instruction": instruction,
                "input": input,
                "full_output": full_output,
                "output": output,
            }
        )

    if not os.path.exists(f"{root_dir}/processed"):
        os.makedirs(f"{root_dir}/processed")
    with open(f"{root_dir}/processed/test.json", "w") as f:
        json.dump(processed, f, indent=4)

    return processed


if __name__ == "__main__":

    process()

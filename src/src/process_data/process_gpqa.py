import os
import json
from loguru import logger
from typing import List
import csv
import pandas as pd
import random


random.seed(42)
root_dir = "../datasets/gpqa"


def process() -> List:

    # Read the CSV file, ensuring proper parsing of multiline answers
    df = pd.read_csv(f"{root_dir}/raw/test.csv", quoting=csv.QUOTE_ALL)

    selected_columns = df[
        [
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "High-level domain",
            "Subdomain",
        ]
    ]

    # process it into the format of question, A, B, C, D, correct answer
    processed = []
    for i in range(len(df)):
        ele = {}
        correct_answer = random.choice(["A", "B", "C", "D"])
        ele[correct_answer] = df.iloc[i]["Correct Answer"]
        all_choices = ["A", "B", "C", "D"]
        all_choices.remove(correct_answer)
        for choice, text in zip(
            all_choices,
            df.iloc[i][
                ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
            ].to_list(),
        ):
            ele[choice] = text
        question = df.iloc[i]["Question"]
        processed.append(
            {
                "question": question,
                "A": ele["A"],
                "B": ele["B"],
                "C": ele["C"],
                "D": ele["D"],
                "correct_answer": correct_answer,
                "high-level-domain": df.iloc[i]["High-level domain"],
                "subdomain": df.iloc[i]["Subdomain"],
            }
        )

    if not os.path.exists(f"{root_dir}/processed"):
        os.makedirs(f"{root_dir}/processed")

    pd.DataFrame(processed).to_csv(
        f"{root_dir}/processed/test.csv", index=False, header=False
    )


if __name__ == "__main__":

    process()

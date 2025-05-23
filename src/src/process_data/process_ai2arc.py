
import os
import json
from loguru import logger
from typing import List
import csv
import pandas as pd
import random
from datasets import load_dataset

import ast

random.seed(42)
root_dir = "../datasets/ai2arc"

def process() -> List:
    
    # # Read the CSV file, ensuring proper parsing of multiline answers
    # df = pd.read_csv(f"{root_dir}/raw/test.csv", quoting=csv.QUOTE_ALL)
    
    dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge')
    # process it into the format of question, A, B, C, D, correct answer
    processed = []

    # get the max number of choices
    max_choice_num = max(len(ele['choices']['label']) for ele in dataset['test'])

    for ele in dataset['test']:
        processed_ele = {}
        question = ele["question"]
        choices = ele["choices"]
        answer = ele["answerKey"]
        for text, la in zip(choices['text'] + ["NA"] * (max_choice_num - len(choices['text'])), ['A', 'B', 'C', 'D', 'E']):
            processed_ele[la] = text

        processed_ele['question'] = question
        processed_ele['answer'] = answer
        processed.append(processed_ele)
    
    if not os.path.exists(f"{root_dir}/processed"):
        os.makedirs(f"{root_dir}/processed")

    pd.DataFrame(processed).to_csv(f"{root_dir}/processed/test.csv", index=False)



if __name__ == "__main__":
    
    process()

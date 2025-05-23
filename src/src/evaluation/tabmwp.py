
import json
from loguru import logger
import re
from typing import List, Optional, Tuple
import argparse
from src.evaluation.utils.tabmwp_utils import grade_answer
import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info

import ast

def load_string_as_dict(input_string):
    # Convert the string into a Python dictionary
    return ast.literal_eval(input_string)

def eval() -> None:
    files = find_files(root_dir=root_dir, name="tabmwp_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        
        predictions = [line["predict"] for line in data]

        with open("/code/tablellm/datasets/tama_instruct/tabmwp/raw/problems_test.json", 'r') as f:
            gold_data = json.load(f)
        references = [line["answer"] for _, line in gold_data.items()]
        num_correct = 0
        for pred, gold in zip(predictions, references):
                if grade_answer(str(pred), str(gold)):
                    num_correct += 1
        acc = 100 * num_correct / len(predictions)
        # print(f"wikitqa, acc, {100 * num_correct / len(preds)}")

        logger.info(folder_name, steps)
        logger.info(f"TabMWP: {acc}")

        print(f"{extract_info(folder_name)}, {steps}, {acc}")
        
if __name__ == "__main__":
    eval()
    




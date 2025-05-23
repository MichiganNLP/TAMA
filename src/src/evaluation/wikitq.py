import json
from loguru import logger
import re
from typing import List, Optional, Tuple
import argparse
from src.evaluation.utils.wikitqa_utils import to_value_list, check_denotation
import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info


def eval() -> None:
    files = find_files(root_dir=root_dir, name="wikitq_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        
        predictions = [line["predict"] for line in data]
  
        with open("/code/tablellm/datasets/tama_instruct/wikitq/raw/test.json", 'r') as f:
            gold_data = json.load(f)
        references = [line["answers"] for line in gold_data]
        num_correct = 0
        for pred, gold in zip(predictions, references):
            try:
                if pred:
                    pred = to_value_list([pred])
                    gold = to_value_list(gold)
                    if check_denotation(gold, pred):
                        num_correct += 1
            except Exception as e:
                # if any of the above failed, we classify it as incorrect
                print(e.text)
        acc = 100 * num_correct / len(predictions)
        # print(f"wikitqa, acc, {100 * num_correct / len(preds)}")

        logger.info(folder_name, steps)
        logger.info(f"WikiTQ: {acc}")

        print(f"{extract_info(folder_name)}, {steps}, {acc}")
        
if __name__ == "__main__":
    eval()
    

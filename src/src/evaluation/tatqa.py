
from src.evaluation.utils.tatqa_utils import _answer_to_bags


import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info

def eval() -> None:
    files = find_files(root_dir=root_dir, name="tatqa_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        
        predictions = [line["predict"] for line in data]
        references = [line["label"] for line in data]
        correct_num = 0
        for pred, gold in zip(predictions, references):
            pred_bags = _answer_to_bags(str(pred))
            gold_bags = _answer_to_bags(gold)
            if set(pred_bags[0]) == set(gold_bags[0]) and len(pred_bags[0]) == len(gold_bags[0]):
                correct_num += 1
        
        results = correct_num / len(references) * 100
        logger.info(folder_name, steps)
        logger.info(f"TATQA: {results}")

        print(f"{extract_info(folder_name)}, {steps}, {results}")
        
if __name__ == "__main__":
    eval()
    


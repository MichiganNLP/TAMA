import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info

import re
import json

def extract_json_answer(text):
    # Regular expression to match the {"answer": "X"} pattern
    match = re.search(r'\{\s*".*?"\s*:\s*".*?"\s*\}', text)
    
    if match:
        # Extract the matched string
        json_str = match.group(0)
        
        # Optionally, convert the string to a dictionary
        
        return json_str
    else:
        return None
    
def eval() -> None:
    files = find_files(root_dir=root_dir, name="s1_test_gpt_500")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        predictions = [line["predict"] for line in data]
        references = [line["label"] for line in data]
        
        num_correct = 0
        for pred, gold in zip(predictions, references):
            try:
                pred = json.loads(extract_json_answer(pred))
                gold = json.loads(gold)
                assert len(list(gold.values())) == 1
                gold_value = list(gold.values())[0]
                pred_value = list(pred.values())[0]
                if pred_value == gold_value:
                    num_correct += 1
            except Exception:
                continue
        acc = 100 * num_correct / len(predictions)
        logger.info(folder_name, steps)
        logger.info(f"S1: {acc}")

        print(f"{extract_info(folder_name)}, {steps}, {acc}")
        
if __name__ == "__main__":
    eval()
    
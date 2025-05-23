import json
import random
import os
from typing import List

random.seed(42)

root_dir = "../datasets/tama_instruct/hitab"


def process(name: str = "hitab", outfilename: str = "hitab_test", infilename: str = "hitab_test") -> None:
    
    with open(f"{root_dir}/raw/{infilename}.json", 'r') as f:
        data = json.load(f)
    processed = []
    for itm in data:
        question = itm["question"]
        instruction = itm["instruction"]
        output = itm["output"]
        input_seg = itm["input_seg"]
        
        input = f"{input_seg}\n\n{question}"
        
        processed.append({
            "input": input,
            "instruction": instruction,
            "output": output
        })
    if "train" in outfilename:
        processed = random.sample(processed, k=200)
    if not os.path.exists(f"{root_dir}/processed"):
        os.makedirs(f"{root_dir}/processed")
    with open(f"{root_dir}/processed/{outfilename}.json", 'w') as f:
        json.dump(processed, f, indent=4)
        

if __name__ == "__main__":
    process()
    process(outfilename="train", infilename="hitab_train_7417")
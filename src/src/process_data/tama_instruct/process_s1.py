""" 
Process the first synthesized dataset
"""
import json
from collections import defaultdict
import random

random.seed(42)


def process() -> None:
    with open("../datasets/tama_instruct/s1/raw/test_All.jsonl", "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    edata = []
    for i, ele in enumerate(data):
        ele["id"] = i
        edata.append(ele)
    data = edata
    categorized_data = defaultdict(list)

    for ele in data:
        categorized_data[ele["dataset"]].append(ele)

    selected_test = []
    for k, v in categorized_data.items():
        # if k == "WikiTest":
        #     continue

        # if len(v) < 50:
        #     selected_test.extend(v)
        # else:
        #     selected_test.extend(random.sample(v, k=50))
        selected_test.extend(v)

    # process the selected test file
    processed = []
    for ele in selected_test:
        task = ele["task"]
        dataset = ele["dataset"]
        original_id = ele["id"]
        instruction = "You are a helpful assistant that specializes in tables."
        input = ele["prompt"]
        output = ele["completion"]
        processed.append(
            {
                "task": task,
                "dataset": dataset,
                "original_id": original_id,
                "instruction": instruction,
                "input": input,
                "output": output,
            }
        )

    processed = random.sample(processed, k=5000)

    with open(
        "../datasets/tama_instruct/s1/processed/test_all_gpt_5000.json", "w"
    ) as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    process()

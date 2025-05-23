import os
import json
from loguru import logger
from typing import List

root_dir = "../datasets/"


def process(name: str, dataset_dir: str, filename: str) -> List:

    with open(f"{dataset_dir}/{filename}", "r") as f:
        data = json.load(f)
    processed = []
    for itm in data:
        question = itm["question"]
        instruction = itm["instruction"]
        output = itm["output"]
        input_seg = itm["input_seg"]

        input = f"{input_seg}\n\n{question}"

        processed.append({"input": input, "instruction": instruction, "output": output})
    if not os.path.exists(f"{root_dir}/{name}/processed"):
        os.makedirs(f"{root_dir}/{name}/processed")
    with open(f"{root_dir}/{name}/processed/{filename}", "w") as f:
        json.dump(processed, f, indent=4)

    return processed


if __name__ == "__main__":
    to_process = [
        ("fetaqa", "fetaqa_test.json"),
        ("fetaqa", "fetaqa_train_7325.json"),
        ("feverous", "feverous_eval.json"),
        ("hitab", "hitab_test.json"),
        ("hitab", "hitab_train_7417.json"),
        ("hybridqa", "hybridqa_eval.json"),
        ("kvret", "kvret_test.json"),
        ("tabfact", "tabfact_test.json"),
        ("tabfact", "tabfact_train_92283.json"),
        ("totto", "totto_eval.json"),
        ("wikisql", "wikisql_test.json"),
        ("wikitq", "wikitq_test.json"),
    ]
    all_train_data = []
    for (name, filename) in to_process:
        logger.info(f"Processing {name}, {filename}")
        dataset_dir = f"{root_dir}/{name}/raw"
        if "train" in filename:
            train_data = process(name, dataset_dir, filename)
            for itm in train_data:
                itm["domain"] = name
                all_train_data.append(itm)
        else:
            process(name, dataset_dir, filename)

    if not os.path.exists(f"{root_dir}/combined_3/"):
        os.makedirs(f"{root_dir}/combined_3/")

    with open(f"{root_dir}/combined_3/train.json", "w") as f:
        json.dump(all_train_data, f, indent=4)

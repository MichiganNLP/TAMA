""" 
For some datasets, we modify them based on the version processed by UnifiedSKG:
https://drive.google.com/drive/folders/1GXigUv3MU-Sh4XiY6Wz3xVeNT_s0SuON
"""

import json
import re
import random
from loguru import logger
from tqdm import tqdm

random.seed(42)

root_dir = "../datasets/tama_instruct/seq2seq_unifiedskg/"


def replace_row_patterns(text):
    # Regular expression to match "row X :" where X is any number
    result = re.sub(r"row \d+ :", r" | [SEP] | ", text)
    return result


def process(name: str) -> None:

    with open(f"{root_dir}/end2end_data/{name}", "r") as f:
        data = json.load(f)

    if "train" in name:
        data = random.sample(data, k=200)

    instruction = None
    if "feverous" in name:
        instruction = "This is a table fact verification task. The goal of this task is to distinguish whether the given table supports the statement, or refutes the statement, or there is not enough info."
    elif "hybridqa" in name:
        instruction = "This is a hybrid question answering task. The goal of this task is to answer the question given tables and passages."
    elif "kvret" in name:
        instruction = "This is a dialogue response generation task grounded on tables. The goal of this task is to generate response based on the given dialogue history and the given table. The dialogues are grounded through underlying tables and span three distinct tasks in the in-car personal assistant space: calendar scheduling, weather information retrieval, and point-of-interest navigation."
    elif "totto" in name:
        instruction = "This is a highlighted cells description task. The goal of this task is to generate the language description given table cells."
    elif "wikisql" in name:
        instruction = "This is a table QA task. The goal of this task is to answer the question given the table."
    elif "wikitq" in name:
        instruction = "This is a table QA task. The goal of this task is to answer the question given the table."
    else:
        raise NotImplementedError
    processed = []
    for itm in tqdm(iter(data)):
        table_content = itm["struct_in"]
        text = itm["text_in"]
        if "feverous" in name:
            text = f"The statement is: <{text}>. Is it entailed or refuted by the table above? If you think the current information can not provide enough evidence for determining it, please choose 'not enough info', otherwise please choose the answer from 'supports' or 'refutes'."
        elif "hybridqa" in name:
            text = f"The passage may also provide related context. You can refer to both the passages and the table when you answer the question. passages: {itm['passage']} |  The question: {text}"
        elif "kvret" in name:
            text = f"The dialogue history is: <{text}>. Please generate the response based on the given table and the given dialogue history."
        elif "totto" in name:
            text = "Please generate one natural language description to describe the given highlighted table cells."
        elif "wikisql" in name:
            pass
        elif "wikitq" in name:
            pass
        else:
            raise NotImplementedError
        output = itm["seq_out"]
        table_content = replace_row_patterns(table_content)
        processed.append(
            {
                "input": f"{table_content}\n\n{text}",
                "instruction": instruction,
                "output": output,
            }
        )

    with open(f"{root_dir}/processed/{name}", "w") as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    for name in [
        # "compwebq_train.json",
        # "compwebq_test.json",
        # "dart_train.json",
        # "dart_test.json",
        "feverous_train.json",
        "hybridqa_train.json",
        "kvret_train.json",
        # "mmqa_train.json",
        # "mmqa_eval.json",
        # "sqa_train.json",
        # "sqa_test.json",
        "totto_train.json",
        "wikisql_train.json",
        "wikitq_train.json",
    ]:
        logger.info(f"Processing {name} file.")
        process(name)

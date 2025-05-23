import json
import os
from loguru import logger
import random

from src.process_data.tama_instruct.utils.utils import read_tsv_as_dict_list

random.seed(42)


def process(split: str) -> None:
    logger.info("Processing infotabs")
    instruction = "Please select the relationship between the table and the hypothesis, you may choose one from entail, contradict, neutral."

    root_dir = "../datasets/tama_instruct/infotabs"
    if split == "test":
        data = read_tsv_as_dict_list(
            f"{root_dir}/raw/infotabs/data/maindata/infotabs_test_alpha1.tsv"
        )
        data += read_tsv_as_dict_list(
            f"{root_dir}/raw/infotabs/data/maindata/infotabs_test_alpha2.tsv"
        )
        data += read_tsv_as_dict_list(
            f"{root_dir}/raw/infotabs/data/maindata/infotabs_test_alpha3.tsv"
        )
    elif split == "train":
        data = read_tsv_as_dict_list(
            f"{root_dir}/raw/infotabs/data/maindata/infotabs_train.tsv"
        )
    else:
        raise NotImplementedError

    fns = os.listdir(f"{root_dir}/raw/infotabs/data/tables/html/")
    html_tables = dict()
    for fn in fns:
        with open(f"{root_dir}/raw/infotabs/data/tables/html/{fn}", "r") as f:
            table_content = f.read()
        html_tables[fn.split(".html")[0]] = json.dumps(table_content)

    processed = []
    for itm in data:
        table_id = itm["table_id"]
        html_table = html_tables[table_id]
        hypothesis = itm["hypothesis"]
        output = itm["label"]
        if output == "E":
            output = "entail"
        elif output == "N":
            output = "neutral"
        elif output == "C":
            output = "contradict"
        else:
            logger.error(f"Output {output} not supported.")
            raise ValueError(f"Output {output} not supported.")
        input = f"Table: {html_table}\n\nHypothesis: {hypothesis}"
        processed.append(
            {"input": input, "output": output, "instruction": instruction,}
        )

    if split == "train":
        processed = random.sample(processed, k=200)

    processed = random.sample(processed, k=500)

    with open(f"{root_dir}/processed/{split}_gpt.json", "w") as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    # process("train")
    process("test")

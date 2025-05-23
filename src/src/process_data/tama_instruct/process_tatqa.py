import json
from loguru import logger
from typing import List
import random

random.seed(42)

def process(split: str) -> None:
    logger.info("Processing tatqa")
    root_dir = "../datasets/tama_instruct/tatqa"

    if split == "test":
        with open(f"{root_dir}/raw/TAT-QA/dataset_raw/tatqa_dataset_test_gold.json", 'r') as f:
            data = json.load(f)
    elif split == "train":
        with open(f"{root_dir}/raw/TAT-QA/dataset_raw/tatqa_dataset_train.json", 'r') as f:
            data = json.load(f)

    processed = []
    for itm in data:
        table_content = json.dumps(itm["table"]["table"])
        paragraphs = [ele["text"] for ele in itm["paragraphs"]]

        for question in itm["questions"]:
            question_text = question["question"]
            answer = question["answer"]
            if isinstance(answer, List):
                answer = ", ".join(answer)
            else:
                answer = str(answer)

            processed.append({
                "input": f"""\
Table:
```
{table_content}
```

Relevant Paragraphs:
```
{"\n".join(paragraphs)}
```

Question:
{question_text}
""",
            "instruction": "Please answer the question given the table and the relevant paragraphs.",
            "output": answer
            })

    if split == "train":
        processed = random.sample(processed, k=200)
    
    with open(f"{root_dir}/processed/{split}.json", 'w') as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    process("train")
    process("test")

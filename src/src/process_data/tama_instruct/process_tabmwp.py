import json
import random
from loguru import logger


random.seed(42)

def process(split: str) -> None:
    logger.info("Processing TabMWP")
    root_dir = "../datasets/tama_instruct/tabmwp"

    if split == "test":
        with open(f"{root_dir}/raw/PromptPG/data/tabmwp/problems_test.json", 'r') as f:
            data = json.load(f)
    elif split == "train":
        with open(f"{root_dir}/raw/PromptPG/data/tabmwp/problems_train.json", 'r') as f:
            data = json.load(f)
        
    processed = []
    for itm in list(data.values()):
        title = itm["table_title"]
        table_content = itm["table"]
        question = itm["question"]
        choices = itm["choices"]
        answer = itm["answer"]
        choice_str = ""
        if choices:
            choice_str = f"Please choose from the choices:\n{', '.join(choices)}"
        processed.append({
            "instruction": "Please answer the question given the table in the format of {'result': 'Your Answer'}.",
            "input": f"""\
Table Title:
{title}

Table content:
```
{table_content}
```

Question:

{question}

{choice_str}
""",
            "output": answer
        })
    
    if split == "train":
        processed = random.sample(processed, k=200)

    with open(f"{root_dir}/processed/{split}_json.json", 'w') as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    # process("train")
    process("test")

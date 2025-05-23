import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info


def eval() -> None:
    files = find_files(root_dir=root_dir, name="fetaqa_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, "r") as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        predictions = [line["predict"] for line in data]
        references = [line["label"] for line in data]
        sacrebleu = evaluate.load("sacrebleu")

        results = sacrebleu.compute(predictions=predictions, references=references)[
            "score"
        ]
        logger.info(folder_name, steps)
        logger.info(f"FeTaQA: {results}")

        print(f"{extract_info(folder_name)}, {steps}, {results}")


if __name__ == "__main__":
    eval()

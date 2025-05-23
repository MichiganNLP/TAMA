import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info


def eval() -> None:
    files = find_files(root_dir=root_dir, name="ait-qa_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, "r") as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        predictions = [line["predict"] for line in data]
        references = [line["label"] for line in data]
        num_correct = 0
        if "original" in folder_name:
            for pred, gold in zip(predictions, references):
                if gold in pred:
                    num_correct += 1
        for pred, gold in zip(predictions, references):
            if pred == gold:
                num_correct += 1

        results = 100 * num_correct / len(references)
        logger.info(folder_name, steps)
        logger.info(f"AITQA: {results}")

        print(f"{extract_info(folder_name)}, {steps}, {results}")


if __name__ == "__main__":
    eval()

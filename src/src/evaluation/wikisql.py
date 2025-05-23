import json
import re
from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info
from src.evaluation.utils.hitab_utils import evaluate


# Script from https://github.com/xlang-ai/UnifiedSKG/blob/main/metrics/unified/evaluator.py#L18
def maybe_normalize_float(span):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def eval_ex_match(pred, gold_result):
    pred = [span.strip() for span in pred.split(', ')]
    gold_result = [span.strip() for span in gold_result.split(', ')]

    clean_float = True  # TODO
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]

    return sorted(pred) == sorted(gold_result)


def eval() -> None:
    files = find_files(root_dir=root_dir, name="wikisql_test")
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

            if "original" in folder_name:
                if gold in pred:
                    num_correct += 1
            else:
                if eval_ex_match(pred, gold):
                    num_correct += 1
        
        result = 100 * num_correct / len(predictions)
        logger.info(folder_name, steps)
        logger.info(f"WikiSQL: {result}")

        print(f"{extract_info(folder_name)}, {steps}, {result}")
        
if __name__ == "__main__":
    eval()
    
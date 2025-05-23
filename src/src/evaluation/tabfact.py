import json
from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info

def evaluate() -> None:
    files = find_files(root_dir=root_dir, name="tabfact_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        correct_num = 0
        # if "original" in folder_name:
        possible_labels = list(set(itm['label'] for itm in data))
        for line in data:
            gold = line["label"]
            predict =  line["predict"]

            # if "original" in folder_name:
            if gold in predict and all(label not in predict for label in [x for x in possible_labels if x != gold]):
                correct_num += 1
            # else:
            #     if predict.startswith(gold):
            #         correct_num += 1
        logger.info(folder_name, steps)
        logger.info(f"TabFact: {correct_num / len(data) * 100}")
        
        print(f"{extract_info(folder_name)}, {steps}, {correct_num / len(data) * 100}")

if __name__ == "__main__":
    evaluate()
    
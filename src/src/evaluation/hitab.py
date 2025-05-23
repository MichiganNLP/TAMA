import json

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info
from src.evaluation.utils.hitab_utils import evaluate

def eval() -> None:
    files = find_files(root_dir=root_dir, name="hitab_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        
        pred_list = []
        gold_list = []

        for i in range(len(data)):
            if len(data[i]["predict"].strip("</s>").split(">, <")) > 1:
                instance_pred_list = data[i]["predict"].strip("</s>").split(">, <")
                pred_list.append(instance_pred_list)
                gold_list.append(data[i]["label"].strip("</s>").split(">, <"))
            else:
                pred_list.append(data[i]["predict"].strip("</s>"))
                gold_list.append(data[i]["label"].strip("</s>"))
        
        result = evaluate(gold_list, pred_list)["exact_match"]
        logger.info(folder_name, steps)
        logger.info(f"HiTab: {result}")

        print(f"{extract_info(folder_name)}, {steps}, {result}")
        
if __name__ == "__main__":
    eval()
    
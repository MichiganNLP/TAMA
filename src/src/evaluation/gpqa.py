import json
from loguru import logger
from src.evaluation.utils.utils import find_eval_files, root_dir, extract_info

def evaluate() -> None:
    files = find_eval_files(root_dir=root_dir, name="gpqa")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        
        result = float(data[0].strip().split(": ")[-1])
        logger.info(folder_name, steps)
        logger.info(f"GPQA: {result}")

        print(f"{extract_info(folder_name)}, {steps}, {result}")
        
if __name__ == "__main__":
    evaluate()
    
import json
import evaluate

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info

def eval() -> None:
    files = find_files(root_dir=root_dir, name="totto_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        predictions = [line["predict"] for line in data]

        with open("/code/tablellm/datasets/tama_instruct/seq2seq_unifiedskg/end2end_data/totto_eval.json", 'r') as f:
            gold_data = json.load(f)

        references = []
        max_ref_num = max(len(line["final_sentences"]) for line in gold_data)
        for line in gold_data:
            single_ref = line["final_sentences"]
            single_ref.extend([None for _ in range(max_ref_num - len(line["final_sentences"]))])
            references.append(single_ref)
        sacrebleu = evaluate.load('sacrebleu')
        results = sacrebleu.compute(predictions=predictions, references=references)["score"]
        logger.info(folder_name, steps)
        logger.info(f"ToTTo: {results}")

        print(f"{extract_info(folder_name)}, {steps}, {results}")
        
if __name__ == "__main__":
    eval()
    
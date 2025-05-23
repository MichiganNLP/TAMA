import os
import json
from typing import Dict, List

home_dir = "/code/"
# home_dir = "/home/dnaihao/dnaihao-scratch/"
# root_dir = f"{home_dir}/tablellm/saves/llama3.1-8b/full/sft"
# root_dir = f"{home_dir}/tablellm/saves/llama3.1-8b-instruct/full/sft"
# root_dir = "/code/tablellm/saves/CodeLlama-13b-Instruct/full/sft"
# root_dir = "/code/tablellm/saves/tablellama/full/sft"
# root_dir = "/code/tablellm/saves/longlora/full/sft"
# root_dir = "/code/tablellm/saves/tablebenchllm/full/sft"
# root_dir = "/home/dnaihao/dnaihao-scratch/tablellm/saves/tablellm/full/sft"
# root_dir = "/code/tablellm/saves/gpt-3.5-turbo"
# root_dir = "/code/tablellm/saves/gpt-4-turbo"
# root_dir = "/code/tablellm/saves/tablellm/full/sft/"
# root_dir = "/code/tablellm/saves/tablellama"
# root_dir = "/code/table-sft/saves/mistral-finetune"
# root_dir = "/code/table-sft/saves/olmo-finetune"
# root_dir = "/code/table-sft/saves/0-allenai-OLMo-7B-0724-Instruct-hf-none-0.0e-0"
# root_dir = "/code/tablellm/saves/llama2-7b-instruct/full/sft/num_exploration"
# root_dir = "/code/tablellm/saves/qwen-7b-instruct/full/sft/num_exploration"
# root_dir = "/code/tablellm/saves/mistral-7b-instruct/full/sft/num_exploration"
# root_dir = "/code/tablellm/saves/phi-7b-instruct/full/sft/num_exploration"
# root_dir = "/code/tablellm/saves/llama3.1-8b-instruct/lora/sft/num_exploration"
# root_dir = "/code/tablellm/saves/llama3.1-8b-instruct/qlora/sft/num_exploration"
# root_dir = "/code/table-sft/saves/mistral_nemo-finetune"
# root_dir = "/code/table-sft/src/0-mistralai-Mistral-Nemo-Instruct-2407-none-0.0e-0"
# root_dir = "/code/table-sft/saves/phi-finetune"
# root_dir = "/code/table-sft/saves/mistral-finetune"
# root_dir = "/code/table-sft/saves/0-microsoft-Phi-3-small-8k-instruct-none-0.0e-0"
# root_dir = "/code/table-sft/saves/0-mistralai-Mistral-7B-Instruct-v0.3-none-0.0e-0"
root_dir = "/code/table-sft/saves/phi-mini-finetune"
# root_dir = "/code/table-sft/saves/0-microsoft-Phi-3-mini-4k-instruct-none-0.0e-0"


temperature = 0.01
top_p = 0.95

def find_files(root_dir: str, name: str) -> Dict:
    """ 
    Given the root dir, search for all the predictions correponding to the dataset name
    Return a dictionary of the folder name and the corresponding filepath
    """
    folders = os.listdir(root_dir)
    pred_files = dict()

    if "gpt-3.5" in root_dir or "gpt-4" in root_dir or root_dir.endswith("tablellm") or root_dir.endswith("tablellama") \
        or "none-0.0e-0" in root_dir:
        folders = os.listdir(f"{root_dir}/evaluations/")
        for folder in folders:
            if name == folder:
                pred_files[f"original_{folder}_0"] = {
                    "path": f"{root_dir}/evaluations/{folder}/generated_predictions.jsonl",
                    "steps": 0
                }
        return pred_files

    for folder in folders:
        # if "original-model" not in folder:
        #     continue
        # if not any(x in name for x in ["tablellama", "tablegpt_large", "tablellm"]) or ("5.0e-7" not in folder):
        #     continue
        # for step in [18060]:
        if "5.0e-7" not in folder:
            continue
        if "tablegpt" not in folder:
            continue
        # for step in [13602]:
        # if "train_2600" not in folder:
        #     continue
        # for step in [146, 292, 438, 584, 730, 876]:
        dirs = os.listdir(f"{root_dir}/{folder}")
        step = 0
        # data_folder_name = f"{name}-{temperature}-{top_p}"
        data_folder_name = name
        step = max([int(x.split("-")[-1]) for x in dirs if "checkpoint" in x])
        if os.path.exists(f"{root_dir}/{folder}/checkpoint-{step}/evaluations/{data_folder_name}/generated_predictions.jsonl"):
        # if os.path.exists(f"{root_dir}/{folder}/evaluations/{data_folder_name}/generated_predictions.jsonl"):
            # also read in the number of steps
            if os.path.exists(f"{root_dir}/{folder}/trainer_log.jsonl"):
                with open(f"{root_dir}/{folder}/trainer_log.jsonl", 'r') as f:
                    trainer_log = f.readlines()
                trainer_log = [json.loads(line) for line in trainer_log]
                steps = trainer_log[0]["total_steps"]
            else:
                steps = 0
            steps = step
            
            pred_files[f"{folder}_{step}"] = {
                # "path": f"{root_dir}/{folder}/evaluations/{data_folder_name}/generated_predictions.jsonl",
                "path": f"{root_dir}/{folder}/checkpoint-{step}/evaluations/{data_folder_name}/generated_predictions.jsonl",
                "steps": steps
            }
            
    return pred_files


def find_eval_files(root_dir: str, name: str) -> Dict:
    """ 
    Given the root dir, search for all the predictions correponding to the dataset name
    Return a dictionary of the folder name and the corresponding filepath
    """
    folders = os.listdir(root_dir)
    pred_files = dict()
    for folder in folders:
        # if "original-model" not in folder:
        #     continue
        # for step in [0]:
        step = 0
        if os.path.exists(f"{root_dir}/{folder}/evaluations/{name}/results.log"):

            # also read in the number of steps
            with open(f"{root_dir}/{folder}/trainer_log.jsonl", 'r') as f:
                trainer_log = f.readlines()
            trainer_log = [json.loads(line) for line in trainer_log]
            steps = trainer_log[0]["total_steps"]
            
            steps = step
            pred_files[f"{folder}_{step}"] = {
                "path": f"{root_dir}/{folder}/evaluations/{name}/results.log",
                "steps": steps
            }
            
    return pred_files

def extract_info(name: str) -> str:
    """ 
    Given the name of 17-meta-llama-Meta-Llama-3.1-8B-combined_3_100-1.0e-6, 
    extract the information of experiment_id, model name, training dataset name, lr
    """
    spans = name.split("-")
    experiment_id = spans[0]
    model_name = None
    if "Meta-Llama-3.1" in name:
        model_name = "Meta-Llama-3.1"
    elif "CodeLlama-13b" in name:
        model_name = "CodeLlama-13b"
    elif "TableLlama" in name:
        model_name = "TableLlama"
    elif "longlora" in name:
        model_name = "longlora"
    elif "TableLLM" in name:
        model_name = "TableLLM"
    elif "olmo" in name:
        model_name = "olmo"
    elif "tablellama_train" in name:
        model_name = "Mistral-tablellama"
    elif "tablellm_train" in name:
        model_name = "Mistral-tablellm"
    elif "tablegpt_large" in name:
        model_name = "tablegpt"
    elif "tablebench_train" in name:
        model_name = "tablebench"
    elif "Llama-2" in name:
        model_name = "Llama-2"
    elif "Qwen2.5" in name:
        model_name = "qwen-2.5"
    elif "Mistral-7B" in name:
        model_name = "mistral-7b"
    elif "mistral" in name:
        model_name = "mistral-7b"
    elif "Phi-3" in name:
        model_name = "phi-3"
    if "original" in name:
        return ", ".join(["0", "original-model", "NA", "NA"])
    
    assert model_name
    train_data_name = spans[-3]
    lr = "-".join(spans[-2:]).split("_")[0]
    return ", ".join([experiment_id, model_name, train_data_name, lr])
    

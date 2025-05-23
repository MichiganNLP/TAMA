import json
import os
import subprocess

from loguru import logger
from src.evaluation.utils.utils import find_files, root_dir, extract_info, home_dir
from src.evaluation.utils.hitab_utils import evaluate


def eval() -> None:
    with open(
        f"{home_dir}/tablellm/datasets/instruction_following/processed/test.json", "r"
    ) as f:
        original_data = json.load(f)
    prompts = [line["input"] for line in original_data]
    files = find_files(root_dir=root_dir, name="instruction_following_test")
    for folder_name, ele in files.items():
        file_path = ele["path"]
        steps = ele["steps"]
        with open(file_path, "r") as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]

        processed = []
        for line, prompt in zip(data, prompts):
            response = line["predict"]
            processed.append({"prompt": prompt, "response": response})

        with open(f"{os.path.dirname(file_path)}/response_data.jsonl", "w") as f:
            f.write("\n".join([json.dumps(line) for line in processed]))

        p = subprocess.run(
            [
                "python",
                "-m",
                "src.evaluation.instruction_following_eval.evaluation_main",
                f"--input_data={home_dir}/tablellm/datasets/instruction_following/processed/input_data_test.jsonl",
                f"--input_response_data={os.path.dirname(file_path)}/response_data.jsonl",
                f"--output_dir={os.path.dirname(file_path)}",
            ],
            capture_output=True,
            text=True,
        )
        anchor_line_idx = -100
        for idx, line in enumerate(p.stdout.split("\n")):
            if "eval_results_strict.jsonl Accuracy Scores:" in line:
                anchor_line_idx = idx + 1
            if idx == anchor_line_idx:
                assert "prompt-level: " in line
                prompt_level_score = float(line.split("prompt-level: ")[-1]) * 100
            if idx == anchor_line_idx + 1:
                assert "instruction-level: " in line
                instance_level_score = (
                    float(line.split("instruction-level: ")[-1]) * 100
                )

        logger.info(folder_name, steps)
        logger.info(
            f"IFEval strict: prompt-level score: {prompt_level_score}, instance-level score: {instance_level_score}"
        )

        print(
            f"{extract_info(folder_name)}, {steps}, {prompt_level_score}, {instance_level_score}"
        )


if __name__ == "__main__":
    eval()

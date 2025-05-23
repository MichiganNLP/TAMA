import json
from sklearn.model_selection import train_test_split


with open("../datasets/instruction_following/input_data.jsonl", 'r') as f:
    data = f.readlines()
    
data = [json.loads(line) for line in data]
original_data_dict = {ele["key"]: ele for ele in data}
processed = []
for ele in data:
    prompt = ele["prompt"]
    processed.append({
        "instruction": "You are a helpful assistant",
        "input": prompt,
        "output": "dummy place holder",
        "key": ele["key"]
    })



with open("../datasets/instruction_following/processed/test.json", 'w') as f:
    json.dump(processed, f, indent=4)

test_original = [original_data_dict[ele["key"]] for ele in processed]
with open("../datasets/instruction_following/processed/input_data_test.jsonl", 'w') as f:
    f.write("\n".join(json.dumps(ele) for ele in test_original))


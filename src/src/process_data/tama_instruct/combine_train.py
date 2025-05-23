""" 
Combined the selected training set together
"""

import json
import os

root_dir = "../datasets/tama_instruct"
all_train_data = []

for name in ["ait-qa", "fetaqa", "feverous", "hitab", "hybridqa", "infotabs", "kvret", "tabfact", "tabmwp", "tatqa", "totto", "wikisql", "wikitq"]:
    
    if os.path.exists(f"{root_dir}/{name}/processed/train.json"):
        with open(f"{root_dir}/{name}/processed/train.json", 'r') as f:
            train_data = json.load(f)
    elif os.path.exists(f"{root_dir}/{name}/processed/{name}_train.json"):
        with open(f"{root_dir}/{name}/processed/{name}_train.json", 'r') as f:
            train_data = json.load(f) 
    else:
        raise RuntimeError

    all_train_data.extend(train_data)

    if os.path.exists(f"{root_dir}/{name}/processed/test.json"):
        with open(f"{root_dir}/{name}/processed/test.json", 'r') as f:
            test_data = json.load(f)
    elif os.path.exists(f"{root_dir}/{name}/processed/{name}_test.json"):
        with open(f"{root_dir}/{name}/processed/{name}_test.json", 'r') as f:
            test_data = json.load(f) 
    elif os.path.exists(f"{root_dir}/{name}/processed/{name}_eval.json"):
        with open(f"{root_dir}/{name}/processed/{name}_eval.json", 'r') as f:
            test_data = json.load(f) 
    else:
        raise RuntimeError

    with open(f"{root_dir}/combined_200/test/{name}_test.json", 'w') as f:
        json.dump(test_data, f, indent=4)

with open(f"{root_dir}/combined_200/train/train.json", 'w') as f:
    json.dump(all_train_data, f, indent=4)



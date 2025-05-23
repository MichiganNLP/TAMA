import json

def process() -> None:

    with open("../datasets/tama_instruct/s2/raw/TableBench_DP.jsonl", 'r') as f:
        data = f.readlines()
    
    data = [json.loads(ele) for ele in data]
    processed = []
    for ele in data:
        processed.append({
            "instruction": "You are a helpful assistant that specializes in tables.",
            "input": ele["instruction"],
            "output": ele["answer"],
            "type": ele["qsubtype"]
        })
    with open("../datasets/tama_instruct/s2/processed/test.json", 'w') as f:
        json.dump(processed, f, indent=4)


if __name__ == "__main__":
    process()
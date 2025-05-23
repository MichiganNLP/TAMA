import json
import random
from loguru import logger
from sklearn.model_selection import train_test_split 

random.seed(42)
root_dir = "../datasets/tama_instruct/ait-qa/"
path_to_raw = f"{root_dir}/raw/AITQA/raw_data"

def process() -> None:
    with open(f"{path_to_raw}/aitqa_questions.jsonl", 'r') as f:
        data = f.readlines()
    
    all_tables = dict()
    with open(f"{path_to_raw}/aitqa_tables.jsonl", 'r') as f:
        tables = f.readlines()
    tables = [json.loads(line) for line in tables]
    all_tables = {table["id"]: table for table in tables}

    # Process tables
    # We follow the original AITQA paper on this:
    # Row headers as Table cells in a new column: Row headers are added as the first column of the
    # table as regular body cells. We use a dummy text header as the column header for this new
    # column.
    # Header hierarchies as flat headers: Header hierarchies are flattened by concatenating parent
    # header text with children text.

    processed_tables = dict()
    for k, v in all_tables.items():
        col_headers = v["column_header"]
        row_headers = v["row_header"]
        row_datas = v["data"]
        if row_headers != []:
            table_content = "header" + " | " + " | ".join([" ".join(col_l) for col_l in col_headers])
            for i in range(min(len(row_headers), len(row_datas))):
                row_header = row_headers[i]
                row_data = row_datas[i]
                table_content += " | [SEP] | "
                table_content += "".join(row_header) + " | "
                table_content += " | ".join(row_data)
            if len(row_headers) > len(row_datas):
                table_content += " | [SEP] | ".join(["".join(row_header) for row_header in row_headers[len(row_datas):]])
            elif len(row_datas) > len(row_headers):
                for row_data in row_datas[len(row_headers):]:
                    table_content += " | [SEP] | "
                    table_content += " | | "
                    table_content += " | ".join(row_data)
        else:
            table_content = "header" + " | " + " | ".join([" ".join(col_l) for col_l in col_headers])
            for row_data in row_datas:
                table_content += " | [SEP] | "
                table_content += " | ".join(row_data)
        processed_tables[k] = table_content
    

    data = [json.loads(line) for line in data]
    processed = []
    for ele in data:
        table_id = ele["table_id"]
        question = ele["question"]
        answers = ele["answers"]
        assert len(answers) == 1
        processed.append({
            "instruction": "Please answer the question given the table content",
            "input": f"Table: {processed_tables[table_id]}\n\nQuestion: {question}",
            "output": "".join(answers),
            "table_id": table_id,
        })
    train, test = train_test_split(processed, random_state=42, train_size=200)
    logger.info(f"Test tables: {len(set([ele["table_id"] for ele in test]))}")
    with open(f"{root_dir}/processed/train.json", 'w') as f:
        json.dump(train, f, indent=4)
    with open(f"{root_dir}/processed/test.json", 'w') as f:
        json.dump(test, f, indent=4)



if __name__ == "__main__":
    process()

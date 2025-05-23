""" 
Select small amount of training set from the original complete training set
"""
import json
import os
import random
from loguru import logger
from typing import List

random.seed(42)


def select_data(path_to_folder: str, path_to_train: str, num_per_domain: int) -> List:

    assert path_to_train.endswith(".json")

    with open(path_to_train, "r") as f:
        data = json.load(f)

    # Check the domains / original datasets
    domains = list(set(itm["domain"] for itm in data))

    all_selected_data = []
    for domain in domains:
        logger.info(f"Processing {domain}")
        domain_data = [itm for itm in data if itm["domain"] == domain]
        assert len(domain_data) >= num_per_domain
        selected_data = random.sample(domain_data, k=num_per_domain)
        all_selected_data.extend(selected_data)

        if num_per_domain == 500:
            if not os.path.exists(
                f"{path_to_folder}/{num_per_domain}_per_domain/{domain}"
            ):
                os.makedirs(f"{path_to_folder}/{num_per_domain}_per_domain/{domain}")

            with open(
                f"{path_to_folder}/{num_per_domain}_per_domain/{domain}/train.json", "w"
            ) as f:
                json.dump(selected_data, f, indent=4)

    if not os.path.exists(f"{path_to_folder}/{num_per_domain}_per_domain/"):
        os.makedirs(f"{path_to_folder}/{num_per_domain}_per_domain/")

    with open(f"{path_to_folder}/{num_per_domain}_per_domain/train.json", "w") as f:
        json.dump(all_selected_data, f, indent=4)

    return all_selected_data, data


if __name__ == "__main__":
    path_to_folder = "../datasets/combined_3/"
    path_to_train = f"{path_to_folder}/train.json"
    all_selected_data = []
    for num_per_domain in [10, 30, 50, 100, 200, 500]:
        logger.info(f"Selecting {num_per_domain} per domain")
        selected_data, all_data = select_data(
            path_to_folder, path_to_train, num_per_domain=num_per_domain
        )
        all_selected_data.extend(selected_data)

    all_selected_data = {ele["input"] + ele["output"]: ele for ele in all_selected_data}
    domains = list(set(itm["domain"] for itm in all_data))
    # we need to create a dev set for every domain
    remaining_itms = []
    for itm in all_data:
        # first we rule out the ones that have been selected
        if itm["input"] + itm["output"] in all_selected_data:
            continue
        remaining_itms.append(itm)

    if not os.path.exists(f"{path_to_folder}/dev/"):
        os.makedirs(f"{path_to_folder}/dev/")
    for domain in domains:
        domain_data = [ele for ele in remaining_itms if ele["domain"] == domain]
        dev_set = random.sample(domain_data, k=1000)

        with open(f"{path_to_folder}/dev/{domain}.json", "w") as f:
            json.dump(dev_set, f, indent=4)

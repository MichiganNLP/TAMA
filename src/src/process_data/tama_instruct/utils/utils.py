
import csv
def read_tsv_as_dict_list(file_path):
    """
    Read a TSV file and return its contents as a list of dictionaries.

    :param file_path: Path to the TSV file
    :return: List of dictionaries where each dictionary represents a row
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        # Use csv.DictReader to read the TSV file
        # Specify the delimiter as '\t' for TSV files
        reader = csv.DictReader(file, delimiter='\t')
        data = [row for row in reader]
    return data



import pandas as pd
import requests
import os


def read_csv_to_dataframe(file_path):
    """
    Reads a CSV file into a DataFrame. Returns None if the file does not exist.

    Parameters:
    - file_path (str): The local file path to read the CSV from.

    Returns:
    - pd.DataFrame or None: The DataFrame created from the CSV file,
    or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    data = pd.read_csv(file_path, sep=',')
    return data
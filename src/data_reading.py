import pandas as pd
import requests
import os

def data_reading(url, file_path):
    """
    Downloads content from the given URL, saves it to the specified file path,
    and reads the CSV into a DataFrame. Returns None if the file does not exist.

    Parameters:
    - url (str): The URL to download the content from.
    - file_path (str): The local file path to save the downloaded content.

    Returns:
    - pd.DataFrame or None: The DataFrame created from the downloaded CSV file,
      or None if the file does not exist.
    """
    try:
        request = requests.get(url)
        request.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return None

    with open(file_path, 'wb') as f:
        f.write(request.content)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    data = pd.read_csv(file_path, sep=',')
    return data

# Example usage:
# url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
# file_path = '../data/raw/bank-full.csv'
# result_df = data_reading(url, file_path)
# if result_df is not None:
#     print(result_df.head())
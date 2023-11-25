import pandas as pd
import requests
import warnings


def data_reading(url, file_path):
    """
    Downloads content from the given URL, saves it to the specified file path, and reads it into a DataFrame.

    Parameters:
    - url (str): The URL to download the content from.
    - file_path (str): The local file path to save the downloaded content.

    Returns:
    - pd.DataFrame: The DataFrame created from the downloaded CSV file.
    """
    request = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(request.content)

    data = pd.read_csv(file_path, sep=',')
    return data 
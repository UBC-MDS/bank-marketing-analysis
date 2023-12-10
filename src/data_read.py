import pandas as pd
import requests
import os

def download_file(url, file_path):
    """
    Downloads content from the given URL and saves it to the specified file path.

    Parameters:
    - url (str): The URL to download the content from.
    - file_path (str): The local file path to save the downloaded content.

    Returns:
    - bool: True if download is successful, False otherwise.
    """
    try:
        request = requests.get(url)
        request.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return False

    with open(file_path, 'wb') as f:
        f.write(request.content)

    return True

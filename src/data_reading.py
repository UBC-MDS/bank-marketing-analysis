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
    # Download the file
    request = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(request.content)

    # Read the CSV into a DataFrame
    data = pd.read_csv(file_path, sep=',')

    return data

def main():
    # Set display options and random state
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:.3f}'.format
    RANDOM_STATE = 522
    warnings.filterwarnings("ignore")

    # Define the URL and file path
    url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
    file_path = '../data/raw/bank-full.csv'

    # Download and read the CSV into a DataFrame
    bank_data = download_and_read_csv(url, file_path)

    # Use the bank_data DataFrame as needed
    print(bank_data.head())  # Example: Display the first few rows of the DataFrame

if __name__ == "__main__":
    main()
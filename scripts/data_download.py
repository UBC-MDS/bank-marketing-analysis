import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_reading import data_reading

@click.command()
@click.option('--url', type=str, default='https://archive.ics.uci.edu/static/public/222/data.csv')
@click.option('--save_path', type=str, default='data/raw')

def main(url, save_path):
    """
    Downloads a data from the specified url and saves it so a specified save_path.

    Parameters:
    - url (str): The URL of the data file.
    - save_path (str): The local path where the data should be extracted.

    Example:
    To run the script, use the following command in the terminal:
    ```
    python scripts/data_download.py --url='https://archive.ics.uci.edu/static/public/222/data.csv' --save_path='data/raw'
    ```
    """
    save_path = os.path.join(save_path, 'bank-full.csv')
    try:
        data_reading(url, save_path)
    except:
        os.makedirs(save_path)
        data_reading(url, save_path)

if __name__ == '__main__':
    main()
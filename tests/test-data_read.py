import pandas as pd
import pytest
import sys
import os

# Import the functions to be tested
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_reading import download_file, read_csv_to_dataframe

# Test download_file function
def test_download_file():
    test_url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
    test_file_path = 'test_data.csv'

    result = download_file(test_url, test_file_path)

    assert result is True
    assert os.path.exists(test_file_path)

    os.remove(test_file_path)

def test_read_csv_to_dataframe():
    test_data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    test_file_path = 'test_data.csv'

    pd.DataFrame(test_data).to_csv(test_file_path, index=False)

    result_df = read_csv_to_dataframe(test_file_path)

    assert isinstance(result_df, pd.DataFrame)
    pd.testing.assert_frame_equal(result_df, pd.DataFrame(test_data))

    os.remove(test_file_path)

# Test read_csv_to_dataframe returns None if the file does not exist
def test_read_csv_to_dataframe_returns_none_if_file_not_exists():
    non_existing_file_path = 'non_existing_file.csv'

    result_df = read_csv_to_dataframe(non_existing_file_path)

    assert result_df is None
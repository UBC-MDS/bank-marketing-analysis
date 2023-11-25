import pandas as pd
import numpy as np
import pytest
import sys
import os


# Import the re_sample function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.resample import re_sample


def test_data_reading():
    test_url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
    test_file_path = 'test_data.csv'

    result_df = data_reading(test_url, test_file_path)

    assert isinstance(result_df, pd.DataFrame)

    assert os.path.exists(test_file_path)

    os.remove(test_file_path)

def test_data_reading_returns_none_if_file_not_exists():
    non_existing_file_path = 'non_existing_file.csv'

    result_df = data_reading('https://example.com/non_existing_data.csv', non_existing_file_path)

    assert result_df is None

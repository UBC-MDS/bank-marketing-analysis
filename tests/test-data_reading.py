import pandas as pd
import numpy as np
import pytest
import sys
import os
import unittest


# Import the re_sample function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.resample import re_sample

class TestDataReading(unittest.TestCase):
    def setUp(self):
        # Define test URL and file path
        self.test_url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
        self.test_file_path = 'test_data.csv'

    def tearDown(self):
        # Clean up test file after the test
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_data_reading(self):
        # Call the function to be tested
        result_df = data_reading(self.test_url, self.test_file_path)

        # Check if the result is a Pandas DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)

        # Check if the test file has been created
        self.assertTrue(os.path.exists(self.test_file_path))

        # Check if the DataFrame has some expected structure/content
        # You may customize this part based on your specific use case
        self.assertFalse(result_df.empty)
        self.assertIn('column_name', result_df.columns)
        self.assertEqual(len(result_df), expected_row_count)

if __name__ == '__main__':
    unittest.main()
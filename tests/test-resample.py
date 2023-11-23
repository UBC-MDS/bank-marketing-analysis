import pandas as pd
import numpy as np
import pytest
import sys
import os

# Import the re_sample function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.resample import re_sample

# Test data
X_sample = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
y_sample = pd.Series([0, 1, 0, 1])

# Test for each resampling method
@pytest.mark.parametrize("method", [
    'random_over_sample',
    'SMOTE',
    'ADASYN',
    'BorderlineSMOTE',
    'KMeansSMOTE',
    'ClusterCentroids',
    'random_under_sample'
])
def test_re_sample_methods(method):
    X_resampled, y_resampled = re_sample(X_sample, y_sample, func=method)
    assert X_resampled is not None and y_resampled is not None


# Test for 'None' functionality
def test_re_sample_none():
    result = re_sample(X_sample, y_sample, func=None)
    assert result is None


# Test for unsupported func value
def test_re_sample_unsupported():
    result = re_sample(X_sample, y_sample, func='unsupported_method')
    assert result is None


# Test for different data types
def test_re_sample_data_types():
    X_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_array = np.array([0, 1, 0, 1])
    result = re_sample(X_array, y_array, func='random_over_sample')
    assert isinstance(result[0], np.ndarray) and isinstance(result[1], np.ndarray)


# Test for random state
def test_re_sample_random_state():
    X_resampled_1, y_resampled_1 = re_sample(X_sample, y_sample, func='random_over_sample', random_state=42)
    X_resampled_2, y_resampled_2 = re_sample(X_sample, y_sample, func='random_over_sample', random_state=42)
    np.testing.assert_array_equal(X_resampled_1, X_resampled_2)
    np.testing.assert_array_equal(y_resampled_1, y_resampled_2)
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import sys

# Import the model_report function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_report import model_report

# Setup Test Data
X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize different models for testing
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train, y_train)

# Tests for expected inputs
def test_model_report_with_random_forest():
    """ Test if the function correctly generates a report using a RandomForest model. """
    report, _ = model_report(random_forest_model, X_test, y_test, 'RandomForest')
    assert isinstance(report, pd.DataFrame), "Report should be a DataFrame for RandomForest"

def test_model_report_with_gradient_boosting():
    """ Test if the function correctly generates a report using a GradientBoosting model. """
    report, _ = model_report(gradient_boosting_model, X_test, y_test, 'GradientBoosting')
    assert isinstance(report, pd.DataFrame), "Report should be a DataFrame for GradientBoosting"

# Tests for edge cases
def test_model_report_empty_input():
    """ Test how the function handles empty datasets. """
    with pytest.raises(ValueError):
        model_report(random_forest_model, np.array([]), np.array([]), 'EmptyData')

def test_model_report_large_input():
    """ Test how the function handles an extremely large dataset. """
    large_X, large_y = np.random.rand(10000, 10), np.random.randint(0, 2, 10000)
    report, _ = model_report(random_forest_model, large_X, large_y, 'LargeData')
    assert isinstance(report, pd.DataFrame), "Report should be a DataFrame for large dataset"
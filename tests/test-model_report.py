
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Import the model_report function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_report import model_report

# Setup Test Data
X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def test_model_report_valid_input():
    """ Test if the function correctly generates a report for valid input data and a valid model. """
    report = model_report(model, X_test, y_test, 'RandomForest')
    assert isinstance(report, pd.DataFrame), "Report should be a DataFrame"

def test_model_report_invalid_input():
    """ Test how the function handles invalid inputs. """
    with pytest.raises(ValueError):
        model_report(model, 'invalid_data', y_test, 'RandomForest')

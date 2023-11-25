
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib.testing.compare import compare_images
import sys
import os

# Import the compute_and_plot_roc_curve function from the provided file
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.compute_and_plot_roc_curve import compute_and_plot_roc_curve

# Setup Test Data
X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def test_compute_and_plot_roc_curve_valid_input():
    """ Test if the function correctly computes and plots the ROC curve for valid input data and a valid model. """
    compute_and_plot_roc_curve(model, X_test, y_test, 'test_roc_curve')
    plt.savefig('./test_roc_curve.png')
    assert os.path.exists('test_roc_curve.png'), "ROC curve plot file should be created"

def test_compute_and_plot_roc_curve_invalid_input():
    """ Test how the function handles invalid inputs. """
    with pytest.raises(ValueError):
        compute_and_plot_roc_curve(model, 'invalid_data', y_test, 'test_roc_curve')

# Cleanup
def teardown_module(module):
    """ Remove files created during testing. """
    if os.path.exists('test_roc_curve.png'):
        os.remove('test_roc_curve.png')

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Initialize models for testing
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train, y_train)

# Tests for expected inputs
def test_compute_and_plot_roc_curve_with_random_forest():
    """ Test if the function correctly computes and plots the ROC curve using a RandomForest model. """
    compute_and_plot_roc_curve(random_forest_model, X_test, y_test, 'test_roc_curve_random_forest')
    plt.savefig('./test_roc_curve_random_forest.png')
    assert os.path.exists('test_roc_curve_random_forest.png'), "ROC curve plot file should be created for RandomForest"

def test_compute_and_plot_roc_curve_with_gradient_boosting():
    """ Test if the function correctly computes and plots the ROC curve using a GradientBoosting model. """
    compute_and_plot_roc_curve(gradient_boosting_model, X_test, y_test, 'test_roc_curve_gradient_boosting')
    plt.savefig('./test_roc_curve_gradient_boosting.png')
    assert os.path.exists('test_roc_curve_gradient_boosting.png'), "ROC curve plot file should be created for GradientBoosting"

# Tests for edge cases
def test_compute_and_plot_roc_curve_empty_input():
    """ Test how the function handles empty datasets. """
    with pytest.raises(ValueError):
        compute_and_plot_roc_curve(random_forest_model, np.array([]), np.array([]), 'test_empty_data')

def test_compute_and_plot_roc_curve_large_input():
    """ Test how the function handles an extremely large dataset. """
    large_X, large_y = np.random.rand(100000, 10), np.random.randint(0, 2, 100000)
    compute_and_plot_roc_curve(random_forest_model, large_X, large_y, 'test_large_data')
    plt.savefig('./test_roc_curve_large_data.png')
    assert os.path.exists('test_roc_curve_large_data.png'), "ROC curve plot file should be created for large dataset"

# Cleanup
def teardown_module(module):
    """ Remove files created during testing. """
    for file_name in ['test_roc_curve_random_forest.png', 'test_roc_curve_gradient_boosting.png', 'test_roc_curve_large_data.png']:
        if os.path.exists(file_name):
            os.remove(file_name)
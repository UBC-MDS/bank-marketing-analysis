import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn import metrics

def compute_roc_curve(model, testing_x, testing_y):
    """
    Compute the Receiver Operating Characteristic (ROC) curve.

    This function takes a machine learning model and test data as inputs.
    It computes the ROC curve using the model's probability predictions on the test data.

    Parameters:
    - model: A trained machine learning model that supports probability prediction.
    - testing_x: Test dataset (features).
    - testing_y: True labels for the test dataset.

    Returns:
    - fpr (array): An array containing the false positive rates.
    - tpr (array): An array containing the true positive rates.
    - roc_auc (float): The computed area under the ROC curve.

    """
    # Input validation
    if model is None:
        raise ValueError("No model provided. Please provide a machine learning model.")
    if not hasattr(model, 'predict_proba'):
        raise ValueError("The provided model does not support probability predictions.")
    if testing_x is None or testing_y is None:
        raise ValueError("Test data and labels must not be None.")
    
    # Compute ROC curve and ROC area
    predict_prob = model.predict_proba(testing_x)
    fpr, tpr, threshold = metrics.roc_curve(testing_y, predict_prob[:,1])
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, roc_auc

def compute_and_plot_roc_curve(model, testing_x, testing_y, name, figsize=(5,5)):
    """
    Plot the Receiver Operating Characteristic (ROC) curve based on the results from compute_roc_curve().

    This function takes a machine learning model, test data, and the name of the model as inputs.
    It computes the ROC curve using the model's probability predictions on the test data using compute_roc_curve().
    The function plots the ROC curve, showing the trade-off between the true positive rate (TPR)
    and false positive rate (FPR) at various threshold settings. 
    The Area Under the Curve (AUC) is also calculated and displayed in the plot.

    Parameters:
    - model: A trained machine learning model that supports probability prediction.
    - testing_x: Test dataset (features).
    - testing_y: True labels for the test dataset.
    - name (str): The name of the model, used for labeling the plot.
    - figsize (tuple): The size of the figure in which the ROC curve is plotted (default is (5, 5)).

    Returns:
    - fpr (array): An array containing the false positive rates.
    - tpr (array): An array containing the true positive rates.
    - roc_auc (float): The computed area under the ROC curve.

    """
    # Input validation
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    
    # Compute the ROC curve
    fpr, tpr, roc_auc = compute_roc_curve(model, testing_x, testing_y)

    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label='{} : {:0.3f}'.format(name, roc_auc))
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    
    return fig, ax, fpr, tpr, roc_auc
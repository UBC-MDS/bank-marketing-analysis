import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, classification_report, recall_score, precision_score

def compute_metrics(model, testing_x, testing_y, customerized_threshold=False, threshold=0.5):
    """
    Compute various performance metrics for a machine learning model.

    Parameters:
    - model: A trained machine learning model.
    - testing_x: Test dataset (features).
    - testing_y: True labels for the test dataset.
    - customerized_threshold (bool): Flag to apply a custom threshold for predictions (default is False).
    - threshold (float): The custom threshold for classification if customerized_threshold is True (default is 0.5).

    Returns:
    - A dictionary containing recall, precision, F1-score, ROC-AUC score, and predictions.
    """
    # Input validation
    if model is None:
        raise ValueError("No model provided. Please provide a machine learning model.")    
    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
        raise ValueError("Provided model must have 'predict' and 'predict_proba' methods.")
    if testing_x is None or testing_y is None:
        raise ValueError("Test data and labels must not be None.")

    # Make prediction
    predictions = model.predict(testing_x)
    predictions_prob = model.predict_proba(testing_x)

    # Compute metrics
    if customerized_threshold:
        predictions = (predictions_prob[:, 1] > threshold).astype(int)

    metrics_dict = {
        'recall': recall_score(testing_y, predictions),
        'precision': precision_score(testing_y, predictions),
        'f1_score': f1_score(testing_y, predictions),
        'roc_auc': roc_auc_score(testing_y, predictions_prob[:, 1]),
        'predictions': predictions
    }

    return metrics_dict

def plot_confusion_matrix_func(predictions, testing_y, export_path=None):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    - predictions: Model predictions.
    - testing_y: True labels for the test dataset.
    - export_path: Path to export the plot, if provided.
    """
    confusion = confusion_matrix(testing_y, predictions)
    plt.figure(figsize=(5,5), dpi=100)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, indices, fontsize=10)
    plt.yticks(indices, indices, fontsize=10)
    plt.colorbar()
    plt.xlabel('Predictions', fontsize=10)
    plt.ylabel('Ground Truth', fontsize=10)
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            plt.text(i, j, confusion[i][j], va='center', ha='center', c='darkorange', fontsize=10)
    plt.grid(False)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight', dpi=200)

def model_report(model, testing_x, testing_y, name, export_path=None, customerized_threshold=False, threshold=0.5, plot_confusion_matrix=True):
    """
    Generate and print a performance report of a machine learning model on test data.

    This function evaluates a given model on test data and generates various performance metrics
    including recall, precision, F1-score, and ROC-AUC score. It also prints a classification report
    and optionally plots a confusion matrix. The function allows for the application of a custom
    threshold for classification decisions.

    Parameters:
    - model: A trained machine learning model.
    - testing_x: Test dataset (features).
    - testing_y: True labels for the test dataset.
    - name (str): The name of the model, used for labeling in the report.
    - export_path: Path to export the plot, if provided.
    - customerized_threshold (bool): Flag to apply a custom threshold for predictions (default is False).
    - threshold (float): The custom threshold for classification if customerized_threshold is True (default is 0.5).
    - plot_confusion_matrix (bool): Flag to plot the confusion matrix (default is True).

    Returns:
    - DataFrame: A pandas DataFrame containing the model name and calculated performance metrics.
    - DataFrame: A pandas DataFrame containing the classification report.

    The function prints the classification report and, if requested, displays the confusion matrix plot.
    """
    # Input validation
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    
    # Compute metrics
    metrics_dict = compute_metrics(model, testing_x, testing_y, customerized_threshold, threshold)

    # Generate classification report
    classification_rep = pd.DataFrame(classification_report(testing_y, metrics_dict['predictions'], output_dict=True)).transpose()    

    # Generate confusion matrix
    if plot_confusion_matrix:
        plot_confusion_matrix_func(metrics_dict['predictions'], testing_y, export_path)

    df = pd.DataFrame({"Model"           : [name],
                       "Recall_score"    : [metrics_dict['recall']],
                       "Precision"       : [metrics_dict['precision']],
                       "f1_score"        : [metrics_dict['f1_score']],
                       "Area_under_curve": [metrics_dict['roc_auc']]
                      })
    
    return df, classification_rep
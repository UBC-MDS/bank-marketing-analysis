import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score, roc_auc_score, classification_report, recall_score, precision_score

def model_report(model, testing_x, testing_y, name, export_path, customerized_threshold=False, threshold=0.5, plot_confusion_matrix = True) :
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
    - customerized_threshold (bool): Flag to apply a custom threshold for predictions (default is False).
    - threshold (float): The custom threshold for classification if customerized_threshold is True (default is 0.5).
    - plot_confusion_matrix (bool): Flag to plot the confusion matrix (default is True).

    Returns:
    - DataFrame: A pandas DataFrame containing the model name and calculated performance metrics.

    The function prints the classification report and, if requested, displays the confusion matrix plot.
    """

    predictions  = model.predict(testing_x)
    predictions_prob = model.predict_proba(testing_x)
    
    if customerized_threshold:
        predictions = []
        for pred in predictions_prob[:,1]:  
            predictions.append(1) if pred > threshold else predictions.append(0) 
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions_prob[:, 1])
    f1score      = f1_score(testing_y,predictions) 
    
    # classification_report
    classification_rep = pd.DataFrame(classification_report(testing_y, predictions, output_dict=True)).transpose()    

    # customered_confusion_matrix
    if plot_confusion_matrix:   
        fact = testing_y
        classes = list(set(fact))
        classes.sort()
        confusion = confusion_matrix(predictions, testing_y)
        plt.figure(figsize=(5,5), dpi=100)
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        plt.xticks(indices, indices, fontsize=10)
        plt.yticks(indices, indices, fontsize=10)
        plt.colorbar()
        plt.xlabel('Predictions',fontsize=10)
        plt.ylabel('Ground Turth',fontsize=10)
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index],va = 'center',ha = 'center',c='darkorange',fontsize=10)
        plt.grid(False)

    if export_path:
            plt.savefig(export_path, bbox_inches='tight', dpi=200)


    df = pd.DataFrame({"Model"           : [name],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve": [roc_auc]
                      })
    
    return df, classification_rep
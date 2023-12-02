import click
import os
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint, uniform
from sklearn import metrics
from joblib import dump
import dataframe_image as dfi
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.compute_and_plot_roc_curve import compute_and_plot_roc_curve
from src.model_report import model_report


@click.command()
@click.option('--resampled_training_data', type=str, default='data/processed/X_train_resmp.csv')
@click.option('--resampled_training_response', type=str, default='data/processed/y_train_resmp.csv')
@click.option('--test_data', type=str, default='data/processed/X_test.csv')
@click.option('--test_response', type=str, default='data/processed/y_test.csv')
@click.option('--preprocessor_pipe', type=str, default='results/models/bank_preprocessor.pickle')
@click.option('--save_pipelines_to', type=str, default='results/models')
@click.option('--save_plot_to', type=str, default='results/figures')
@click.option('--seed', type=int, default=522)

def main(resampled_training_data, resampled_training_response, test_data, test_response, preprocessor_pipe, save_pipelines_to, save_plot_to, seed):
    '''
    Fits various classifiers to the training data, performs hyperparameter tuning,
    and saves the pipeline objects, hyperparameters, and evaluation results.

    Parameters:
    - resampled_training_data (str): Path to the CSV file containing resampled training data.
    - resampled_training_response (str): Path to the CSV file containing resampled training response.
    - test_data (str): Path to the CSV file containing test data.
    - test_response (str): Path to the CSV file containing test response.
    - preprocessor_pipe (str): Path to the preprocessor pipeline pickle file.
    - save_pipelines_to (str): Directory to save the trained model pipelines.
    - save_plot_to (str): Directory to save the evaluation plots.
    - seed (int): Random seed for reproducibility.

    Returns:
    None

    This function performs the following steps:
    1. Reads the data from CSV files.
    2. Loads a preprocessor pipeline from a pickle file.
    3. Defines a set of classifiers for Decision Tree, KNN, Naive Bayes, and Logistic Regression.
    4. Performs hyperparameter tuning using RandomizedSearchCV for Logistic Regression and KNN.
    5. Saves the best models, hyperparameters, and evaluation results to pickle files.
    6. Generates and saves evaluation plots, including ROC curves and classification reports.

    Note:
    - The function assumes the existence of utility functions, such as `compute_and_plot_roc_curve` and `model_report`.
    - Make sure to run this script as a standalone program to execute the main function.

    Example:
    ```bash
    python scripts/fit_bank_classifier.py \
        --resampled_training_data='data/processed/X_train_resmp.csv' \
        --resampled_training_response='data/processed/y_train_resmp.csv' \
        --test_data='data/processed/X_test.csv' \
        --test_response='data/processed/y_test.csv' \
        --preprocessor_pipe='results/models/bank_preprocessor.pickle' \
        --save_pipelines_to='results/models' \
        --save_plot_to='results/figures' \
        --seed=522
    ```
    '''

    RANDOM_STATE = seed

    X_tr = pd.read_csv(resampled_training_data, sep=",", index_col=0)    
    y_tr = pd.read_csv(resampled_training_response, sep=",", index_col=0)
    X_test = pd.read_csv(test_data, sep=",", index_col=0)    
    y_test = pd.read_csv(test_response, sep=",", index_col=0)

    preprocessor = pickle.load(open(preprocessor_pipe, "rb"))


    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    }

    classification_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Logistic Regression: Tuning
  
    param_dist = {
        "logisticregression__C": loguniform(1e-3, 1e3)
        }


    pipe = make_pipeline(
        preprocessor,
        models['Logistic Regression']  
        )

    random_search = RandomizedSearchCV(pipe, 
                                    param_dist, 
                                    n_iter=100, 
                                    n_jobs=-1, 
                                    cv=5,
                                    scoring=classification_metrics,
                                    refit='roc_auc',
                                    return_train_score=True,
                                    random_state=RANDOM_STATE
                                    )
    random_search.fit(X_tr, y_tr)

    with open(os.path.join(save_pipelines_to, "logistic_random_search.pickle"), 'wb') as f:
        pickle.dump(random_search, f)

    random_search.best_params_

    random_search.best_score_


    params_df = pd.DataFrame(list(random_search.best_params_.items()), columns=['Parameter', 'Value'])
    score_df = pd.DataFrame({"Metric": ["Best Score"], "Value": [random_search.best_score_]})

    dfi.export(params_df, os.path.join(save_plot_to, 'lr_best_params.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(score_df, os.path.join(save_plot_to, 'lr_best_score.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)


    # Logistic Regression: On the test set

    # Use the selected hyperparameters
    best_C = random_search.best_params_['logisticregression__C']
    plot_confusion_matrix = True

    pipe_lr = make_pipeline(
        preprocessor,
        LogisticRegression(C=best_C,
                            random_state=RANDOM_STATE) 
        )
    # Train the model
    pipe_lr.fit(X_tr,  y_tr)

    with open(os.path.join(save_pipelines_to, "logistic_pipeline.pickle"), 'wb') as f:
        pickle.dump(pipe_lr, f)

    fig_lr, ax_lr, fpr_lr, tpr_lr, auc_lr = compute_and_plot_roc_curve(pipe_lr, X_test,  y_test, "Logistic Regression")

    fig_lr.savefig(os.path.join(save_plot_to, 'lr_roc_auc.png'), bbox_inches='tight', dpi=200)

    model_lr, classification_rep = model_report(pipe_lr, X_test, y_test, "Logistic Regression", os.path.join(save_plot_to, 'lr_conf_matr.png'))

    dfi.export(classification_rep, os.path.join(save_plot_to, 'lr_class_rep.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(model_lr, os.path.join(save_plot_to, 'lr_model.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)



    # KNN: Tuning
    param_dist = {
        "kneighborsclassifier__n_neighbors": range(10,50),
        "kneighborsclassifier__weights":['uniform', 'distance']
    }

    pipe = make_pipeline(
        preprocessor,
        models['KNN']  
        )

    grid_search = GridSearchCV(pipe, 
                                param_dist, 
                                n_jobs=-1, 
                                cv=5,
                                scoring=classification_metrics,
                                refit='roc_auc',
                                return_train_score=True
                                )



    grid_search.fit(X_tr, y_tr)

    with open(os.path.join(save_pipelines_to, "KNN_grid_search.pickle"), 'wb') as f:
        pickle.dump(grid_search, f)

    grid_search.best_params_

    grid_search.best_score_


    params_df = pd.DataFrame(list(grid_search.best_params_.items()), columns=['Parameter', 'Value'])
    score_df = pd.DataFrame({"Metric": ["Best Score"], "Value": [grid_search.best_score_]})

    dfi.export(params_df, os.path.join(save_plot_to, 'KNN_best_params.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(score_df, os.path.join(save_plot_to, 'KNN_best_score.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)


    # KNN: On the test set
        # Use the selected hyperparameters
    best_n_neighbors = grid_search.best_params_['kneighborsclassifier__n_neighbors']
    best_weights = grid_search.best_params_['kneighborsclassifier__weights']

    pipe = make_pipeline(
        preprocessor,
        KNeighborsClassifier(n_neighbors=best_n_neighbors,
                            weights=best_weights
                            ) 
        )
    # Train the model
    pipe.fit(X_tr,  y_tr)

    with open(os.path.join(save_pipelines_to, "KNN_pipeline.pickle"), 'wb') as f:
        pickle.dump(pipe, f)

    fig_knn, ax_knn, fpr_knn, tpr_knn, auc_knn = compute_and_plot_roc_curve(pipe, X_test,  y_test, "KNN")

    model_knn, classification_rep = model_report(pipe, X_test, y_test, "KNN", os.path.join(save_plot_to, 'KNN_conf_matr.png'))
    
    fig_knn.savefig(os.path.join(save_plot_to, 'KNN_roc_auc.png'), bbox_inches='tight', dpi=200)

    dfi.export(classification_rep, os.path.join(save_plot_to, 'KNN_class_rep.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(model_knn, os.path.join(save_plot_to, 'KNN_model.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)

    # Decision Tree: Tuning

    param_dist = {
        "decisiontreeclassifier__max_depth": range(2, 200),
        "decisiontreeclassifier__criterion": ['gini', 'entropy', 'log_loss']
    }

    pipe = make_pipeline(
        preprocessor,
        models['Decision Tree']  
        )

    random_search = RandomizedSearchCV(pipe, 
                                    param_dist, 
                                    n_iter=100, 
                                    n_jobs=-1, 
                                    cv=5,
                                    scoring=classification_metrics,
                                    refit='roc_auc',
                                    return_train_score=True,
                                    random_state=RANDOM_STATE
                                    )
    
    # Train the model
    random_search.fit(X_tr, y_tr)


    with open(os.path.join(save_pipelines_to, "tree_random_search.pickle"), 'wb') as f:
        pickle.dump(random_search, f)

    random_search.best_params_

    random_search.best_score_


    params_df = pd.DataFrame(list(random_search.best_params_.items()), columns=['Parameter', 'Value'])
    score_df = pd.DataFrame({"Metric": ["Best Score"], "Value": [random_search.best_score_]})

    dfi.export(params_df, os.path.join(save_plot_to, 'dt_best_params.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(score_df, os.path.join(save_plot_to, 'dt_best_score.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)


    # Decision Tree: On the test set
    # Use the selected hyperparameters
    best_max_depth = random_search.best_params_['decisiontreeclassifier__max_depth']
    best_criterion= random_search.best_params_['decisiontreeclassifier__criterion']

    pipe = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(max_depth=best_max_depth,
                            criterion=best_criterion
                            ) 
        )
    # Train the model
    pipe.fit(X_tr,  y_tr)

    with open(os.path.join(save_pipelines_to, "tree_pipeline.pickle"), 'wb') as f:
        pickle.dump(pipe, f)

    fig_dt, ax_dt, fpr_dt, tpr_dt, auc_dt = compute_and_plot_roc_curve(pipe, X_test,  y_test, "Decision Tree")

    model_dt, classification_rep = model_report(pipe, X_test, y_test, "Decision Tree", os.path.join(save_plot_to, 'dt_conf_matr.png'))
    
    fig_dt.savefig(os.path.join(save_plot_to, 'dt_roc_auc.png'), bbox_inches='tight', dpi=200)

    dfi.export(classification_rep, os.path.join(save_plot_to, 'dt_class_rep.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(model_dt, os.path.join(save_plot_to, 'dt_model.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)

    # Naive Bayes: Tuning
    param_dist = {
        "gaussiannb__var_smoothing": uniform(0, 1),
    }

    pipe = make_pipeline(
        preprocessor,
        models['Naive Bayes']  
        )

    random_search = RandomizedSearchCV(pipe, 
                                    param_dist, 
                                    n_iter=100, 
                                    n_jobs=-1, 
                                    cv=5,
                                    scoring=classification_metrics,
                                    refit='roc_auc',
                                    return_train_score=True,
                                    random_state=RANDOM_STATE
                                    )
    
        # Train the model
    random_search.fit(X_tr, y_tr)


    with open(os.path.join(save_pipelines_to, "nb_random_search.pickle"), 'wb') as f:
        pickle.dump(random_search, f)

    random_search.best_params_

    random_search.best_score_


    params_df = pd.DataFrame(list(random_search.best_params_.items()), columns=['Parameter', 'Value'])
    score_df = pd.DataFrame({"Metric": ["Best Score"], "Value": [random_search.best_score_]})

    dfi.export(params_df, os.path.join(save_plot_to, 'nb_best_params.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(score_df, os.path.join(save_plot_to, 'nb_best_score.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)


    # Naive Bayes: On the test set
    # Use the selected hyperparameters
    best_var_smoothing = random_search.best_params_['gaussiannb__var_smoothing']

    pipe = make_pipeline(
    preprocessor,
    GaussianNB(var_smoothing=best_var_smoothing) 
    )
    # Train the model
    pipe.fit(X_tr, y_tr)

    with open(os.path.join(save_pipelines_to, "nb_pipeline.pickle"), 'wb') as f:
        pickle.dump(pipe, f)

    fig_nb, ax_nb, fpr_nb, tpr_nb, auc_nb = compute_and_plot_roc_curve(pipe, X_test,  y_test, "Naive Bayes")

    model_nb, classification_rep = model_report(pipe, X_test, y_test, "Naive Bayes", os.path.join(save_plot_to, 'nb_conf_matr.png'))

    fig_nb.savefig(os.path.join(save_plot_to, 'nb_roc_auc.png'), bbox_inches='tight', dpi=200)

    dfi.export(classification_rep, os.path.join(save_plot_to, 'nb_class_rep.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)
    dfi.export(model_nb, os.path.join(save_plot_to, 'nb_model.png'), table_conversion='chrome', fontsize=40, max_rows=None, max_cols=None)


if __name__ == '__main__':
    main()
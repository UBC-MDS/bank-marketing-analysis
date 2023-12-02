import click
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


@click.command()
@click.option('--resampled_training_data', type=str, default='data/processed/X_train_trans.csv')
@click.option('--pipeline_model', type=str, default='results/models/logistic_pipeline.pickle')
@click.option('--save_plot_to', type=str, default='results/figures')
@click.option('--seed', type=int, default=522)

def main(resampled_training_data, pipeline_model, save_plot_to, seed):
    '''Evaluates the breast cancer classifier on the test data, extracts feature importance, and saves a bar plot.

    Parameters:
        resampled_training_data (str): Path to the preprocessed training data file.
        pipeline_model (str): Path to the saved logistic regression pipeline model.
        save_plot_to (str): Directory to save the feature importance plot.
        seed (int): Seed for reproducibility
        
    Example:
    ```bash
    python scripts/feat_imp.py \
        --resampled_training_data='data/processed/X_train_trans.csv' \
        --pipeline_model='results/models/logistic_pipeline.pickle' \
        --save_plot_to='results/figures' \
        --seed=522
    ```
        
        '''


    np.random.seed(seed)
    
    with open(pipeline_model, 'rb') as f:
        bank_lr_fit = pickle.load(f)

    X_tr = pd.read_csv(resampled_training_data, sep=",")    
    logistic_regression_model = bank_lr_fit.named_steps['logisticregression']
    coefficients = list(logistic_regression_model.coef_[0])
    feature_names = X_tr.columns.to_list()


    df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort the DataFrame by the 'Coefficient' column in descending order
    df_sorted = df.sort_values('Coefficient', ascending=False)

    # Plot the sorted coefficients using a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted['Feature'], df_sorted['Coefficient'], color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.title('Feature Importance from Logistic Regression')
    plt.xticks(rotation=90)  # Rotate feature names for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    plt.savefig(os.path.join(save_plot_to, 'feat_imp.png'), bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    main()
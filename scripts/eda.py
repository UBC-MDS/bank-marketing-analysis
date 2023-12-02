import click
import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_viz import plot_variables


@click.command()
@click.option('--training_data', type=str, default='data/processed/bank_train.csv')
@click.option('--save_plot_to', type=str, default='results/figures')

def main(training_data, save_plot_to):

    """
    Main function to perform exploratory data analysis (EDA) on the bank marketing dataset and generate visualizations.

    Parameters:
    - training_data (str): Path to the CSV file containing the bank marketing dataset. Default is 'data/processed/bank_train.csv'.
    - save_plot_to (str): Directory path to save the generated plots. Default is 'results/figures'.

    Returns:
    Plots of the different variables.

    Example:
    To run the script, use the following command in the terminal:
    ```
    python scripts/eda.py --training_data='data/processed/bank_train.csv' --save_plot_to='results/figures'
    ```
    """

    bank_train = pd.read_csv(training_data, sep=",", index_col=0)
    bank_train.drop(columns=["y"])

    bank_str = list(bank_train.select_dtypes(include = ['object']).columns)

    bank_categorical = bank_str+['day']
    bank_continuous = ['balance', 'duration', 'campaign', 'pdays', 'previous', 'age', 'day_of_week']

    var_types = ['categorical', 'continuous', 'log']
    all_variables = {
        'categorical': bank_categorical,
        'continuous': bank_continuous,
        'log': bank_continuous  
    }

    # plot data
    for var_type in var_types:
            plot = plot_variables(bank_train, all_variables[var_type], var_type=var_type)
            plot.save(os.path.join(save_plot_to, f"eda_{var_type}_variables.png"), scale_factor=2.0)


if __name__ == '__main__':
    main()
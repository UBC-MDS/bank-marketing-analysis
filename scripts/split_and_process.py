import click
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.resample import re_sample

@click.command()
@click.option('--raw_data', type=str, default='data/raw/bank-full.csv')
@click.option('--save_to', type=str, default='data/processed')
@click.option('--preprocessor_to', type=str, default='results/models')
@click.option('--seed', type=int, default=522)




def main(raw_data, save_to, preprocessor_to, seed):
    """
    This script performs data splitting, resampling, and preprocessing on a banking dataset.
    
    Parameters:
    - raw_data (str): Path to the raw data CSV file.
    - save_to (str): Directory to save the processed data.
    - preprocessor_to (str): Directory to save the preprocessor.
    - seed (int): Seed for random number generation.

    Example:
    To run the script, use the following command in the terminal:
    ```
    python script_name.py --raw_data='data/raw/bank-full.csv' --save_to='data/processed' --preprocessor_to='results/models' --seed=522
    ```
    """

    RANDOM_STATE = seed
    bank = pd.read_csv(raw_data, sep=',')


    # create the split
    bank_train, bank_test = train_test_split(bank
                                        , test_size=0.2
                                        , random_state=RANDOM_STATE
                                        , stratify=bank.y
                                        )
    
    X_train, y_train = bank_train.drop(columns=["y"]), bank_train["y"]
    X_test, y_test = bank_test.drop(columns=["y"]), bank_test["y"]
    y_train = y_train.map({'yes':1, 'no':0})
    y_test = y_test.map({'yes':1, 'no':0})


       # resample data
    X_tr, y_tr= re_sample(X_train, y_train, func='random_under_sample')

    # export train and test sets
    X_train.to_csv(os.path.join(save_to, "X_train.csv"))
    y_train.to_csv(os.path.join(save_to, "y_train.csv"))
    X_tr.to_csv(os.path.join(save_to, "X_train_resmp.csv"))
    y_tr.to_csv(os.path.join(save_to, "y_train_resmp.csv"))
    X_test.to_csv(os.path.join(save_to, "X_test.csv"))
    y_test.to_csv(os.path.join(save_to, "y_test.csv"))



    # preprocessing
    numeric_features = bank.select_dtypes('number').columns.tolist()
    categorical_features = ['job', 'marital', 'contact', 'month', 'poutcome']
    ordinal_features = ['education']
    binary_features = ['default', 'housing', 'loan']
    drop_features = []
    target = "y"

    education_levels = ['tertiary', 'secondary', 'primary']
    ordinal_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                        OrdinalEncoder(categories=[education_levels], dtype=int))

    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    binary_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                        OneHotEncoder(dtype=int, drop='if_binary'))

    categorical_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                            OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('ordinal', ordinal_transformer, ordinal_features),
        ('binary', binary_transformer, binary_features),
        ('categorical', categorical_transformer, categorical_features),
        ('drop', 'passthrough', drop_features)
    ])

    pickle.dump(preprocessor, open(os.path.join(preprocessor_to, "bank_preprocessor.pickle"), "wb"))


if __name__ == '__main__':
    main()
# Changes

- fit_bank_classifier.py: Bug addressed. No warning message on execution.
    The size of the training data would not be reduced as that was not an error.


- feat_imp.py: resampled_training_data changed to transformed_training_data.

- test-data_viz.py: Added tests to verify that input is not null and type of input.

- data_viz.py: plot_variables function fixed to check inputs. FutureWarnings hidden.


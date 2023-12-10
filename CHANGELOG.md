# Changes

## scripts
- fit_bank_classifier.py: Bug addressed. No warning message on execution.
    The size of the training data would not be reduced as that was not an error.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/1102ea5b76e3852b51daa09bbed9f53b25e9af65

- feat_imp.py: resampled_training_data changed to transformed_training_data.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/ba1e092ee606d25fafbd2fa629d2608251e0440b


## src & tests
- data_viz.py: plot_variables function fixed to check inputs. FutureWarnings hidden.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/a17e2667dc143959dad2cc693c86778bfb9b697c

- test-data_viz.py: Added tests to verify that input is not null and type of input.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/783927025da9c154baaa8c4b8600f1d2fb931ef7

- compute_and_plot_roc_curve.py: fixed to split key functionalities and check for proper input and provide useful error messages.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/38bf744fb9dccd06540d5d8ec188a5f1686a487d

- test-compute_and_plot_roc_curve.py: added expected and edge-case input test examples.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/38bf744fb9dccd06540d5d8ec188a5f1686a487d

- model_report.py: fixed to split key functionalities and check for proper input and provide useful error messages.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/38bf744fb9dccd06540d5d8ec188a5f1686a487d

- test-model_report.py: added expected and edge-case input test examples.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/38bf744fb9dccd06540d5d8ec188a5f1686a487d
    
- data_read.py: added a seperate function for reading the data only.
https://github.com/UBC-MDS/bank-marketing-analysis/blob/main/src/data_read.py

- data_write.py: added a seperate function for writing the data into a dataframe.
https://github.com/UBC-MDS/bank-marketing-analysis/blob/main/src/data_write.py

- test-data_read.py: added test for new functions.
https://github.com/UBC-MDS/bank-marketing-analysis/blob/main/tests/test-data_read.py


## Report content
- Changed bullet points to paragraph style.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/38bf744fb9dccd06540d5d8ec188a5f1686a487d

- Revised EDA, Feature Importance and Conclusion parts, added assumptions and limitations of methods and findings.
    https://github.com/UBC-MDS/bank-marketing-analysis/commit/09e95cf626da04b0ac7a526f440a0f5657a8b52f
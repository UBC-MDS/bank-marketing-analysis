# Predicting Bank Marketing Succuss on Term Deposit Subsciption

- Team: Marco Bravo, Nasim Ghazanfari Nasrabadi, Shizhe Zhang, Celeste Zhao
- Milestone 1 for DSCI 522 course


## About

In this analysis, we attempt to build a predictive model aimed at determining whether a client will subscribe to a term deposit, utilizing the data associated with direct marketing campaigns, specifically phone calls, in a Portuguese banking institution. 

After exploring on several models (logistic regression, KNN, decision tree, naive Bayers), we have selected the logistic regression model as our primary predictive tool. The final model performs fairly well when tested on an unseen dataset, achieving the highest AUC (Area Under the Curve) of 0.899. This exceptional AUC score underscores the model's capacity to effectively differentiate between positive and negative outcomes. Notably, certain factors such as last contact duration, last contact month of the year and the clients' types of jobs play a significant role in influencing the classification decision.

The dataset used in this project originates from the Bank Marketing dataset created by S. Moro, P. Rita and P. Cortez at Iscte - University Institute of Lisbon. This dataset is accessible through the UCI Machine Learning Repository and can be accessed [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). Among the four available datasets, we have utilized [bank-full.csv](https://archive.ics.uci.edu/static/public/222/data.csv) which contains all examples and 17 inputs. Each row in the dataset represents an individual client data including the personal details (e.g., age, occupation, loan status, etc.), information regarding their response to the marketing campaign (e.g., outcomes of the previous marketing campaign, number of contacts made during the current campaign, etc.), and the eventual subscription outcome for the term deposit.


## Report

The final report can be found [here](src/analysis.html).


## Usage

First time running the project, run the following from the root of this repository:

``` bash
conda env create --file environment.yml
```

To run the analysis, run the following from the root of this repository:

``` bash
conda activate bank_marketing_env
jupyter lab 
```

Open `src/Analysis.ipynb` in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells...".


## Dependencies

- conda (version 23.6.0 or higher)
- Python (version 3.10.0)
- Others listed in [environment.yml](environment.yml)


## License

The Bank Marketing dataset and materials are licensed under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) (CC BY 4.0) license. If re-using/re-mixing please provide attribution and link to this webpage.

Software licensed under the MIT License. See the [license file](LICENSE) for more information.


## References

Moro,S., Rita,P., and Cortez,P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

Davis, J., & Goadrich, M. The Relationship Between Precision-Recall and ROC Curves. https://www.biostat.wisc.edu/~page/rocpr.pdf

Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Flach, P. A., & Kull, M. Precision-Recall-Gain Curves: PR Analysis Done Right. https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf

Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & Roth, A. (2015, September 28). Generalization in Adaptive Data Analysis and Holdout Reuse. https://arxiv.org/pdf/1506.02629.pdf

Turkes (Vînt), M. C. (Year, if available). Concept and Evolution of Bank Marketing. Transylvania University of Brasov Faculty of Economic Sciences. Retrieved from link to the PDF or ResearchGate. https://www.researchgate.net/publication/49615486_CONCEPT_AND_EVOLUTION_OF_BANK_MARKETING/fulltext/0ffc5db50cf255165fc80b80/CONCEPT-AND-EVOLUTION-OF-BANK-MARKETING.pdf

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decis. Support Syst., 62, 22-31. https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf


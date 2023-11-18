# Bank Marketing Data Analysis

- Team: Marco Bravo, Nasim Ghazanfari Nasrabadi, Shizhe Zhang, Celeste Zhao
- Milestone 1 for DSCI 522 course


## About

Here we attempt to build a model to predict if the client will subscribe a term deposit, utilizing the information which is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 

We finally choose logistic regression model to do the prediction. Our final predictor performs fairly well on an unseen test data set, achieving a score of xxx

The dataset used in this project is the Bank Marketing dataset created by S. Moro, P. Rita and P. Cortez at Iscte - University Institute of Lisbon. It is sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/222/bank+marketing). There are four datasets and we use [bank-full.csv](https://archive.ics.uci.edu/static/public/222/data.csv) which contains all examples and 17 inputs. Each row in the dataset represents a client data including the client's personal information (e.g. age, job, loan, etc.), marketing campaign reaction (e.g. outcome of the previous marketing campaign, number of contacts performed during this campaign, etc.) and term deposit subscription result. 


## Report

The final report can be found [here] (TBU).


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

Open `src/EDA.ipynb` in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells...".


## Dependencies

- conda (version 23.6.0 or higher)
- Python (version 3.10.0)
- Others listed in [environment.yml](environment.yaml)


## License

The Bank Marketing dataset and materials are licensed under a [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) (CC BY 4.0) license. If re-using/re-mixing please provide attribution and link to this webpage.

Software licensed under the MIT License. See the [license file](LICENSE) for more information.


## References

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decis. Support Syst., 62, 22-31.


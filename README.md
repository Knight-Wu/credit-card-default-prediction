# Credit Card Default Prediction

Objective of this project is to predict which customers may default in upcoming months. Credit card default occurs when you have become severely delinquent on your credit card payments. Missing credit card payments once or twice does not count as a default. A payment default occurs when you fail to pay the Minimum Amount Due on the credit card for a few consecutive months.

## Table of Contents

1. Dataset Information
2. Deliverables
3. Assumptions

* Key Metric
* Business Impact
* General Assumptions
* Constraints

4. MVP Roadmap and Time Allocation
5. Future Work
6. Set up
7. Repo Structure
8. Tools
9. Dataset Content
10. Authors and acknowledgement

## Dataset Information

This dataset contains information spanning six months on default payments, demographic factors, credit data, payment history, and credit card bill statements of clients, covering the period from April 2005 to September 2005.

## Deliverables

1. Perform exploratory data analysis on provided dataset
2. Train & Build model to predict on default payment (Probability of default payment by different feature, strongest/weakest indicators of default payment, etc.)
3. Using current dataset, generate new synthetic dataset that is statistically similar to the original data
4. Organise and present work in a PPT

## Assumptions

### Key Metric

The key evaluation metric selected for this classification task is F1 score to balance precision and recall as the dataset is imbalanced, striking a balance between minimising false alarms and ensuring no risk is overlooked. In this context, a 'false alarm' (false positive) is wrongly predicting a customer will default when they would not, and 'overlooking a risk' (false negative) is failing to identify a customer who will default.

Other metrics that will be tracked including ROC-AUC score, precision, and recall. ROC-AUC score is a measure of the model's overall ability to distinguish between customers who will default and those who won't, across a range of thresholds. A higher ROC-AUC score means the model is better at identifying true defaulters as well as true non-defaulters, across various decision thresholds.

### Business Impact

If we predict too many false defaults (low precision), we might unnecessarily restrict credit opportunities to potentially good customers, affecting customer relationships and potential revenue. Conversely, if we fail to identify actual defaulters (low recall), the company risks financial losses. The F1 score helps us find a middle ground, optimizing both aspects.

### General Assumptions

The following assumptions were made for this project:

* The selected model will be consider both explainability and performance, for instance it is likely ensemble models will be most performant but these are more difficult to explain
* Model will output a probability and a threshold will be set to determine the label rather than a binary 1/0
* Dimensionality reduction is not required as the dataset is relatively small
* Time period of synthetic generation is to be xxx
* Assume latency/inference times are not a consideration for model selection and to focus on inference
* Apply `Black` for automated styling

### Constraints

Due to the time constraints (1.5 working days), the following features will not be implemented:

* Software Engineering best practices
  * Unit testing with Pytest
  * Thorough documentation of code
  * Logging
* Data Science best practices
  * Storing parameters in a `yaml` config file
* MLOps best practices
  * Exporting of the model into a runnable file format such as .bin, .exe, .onnx
  * Data version control as the dataset is static
  * Continuous model deployment CI/CD with GitHub actions or Kubeflow
  * Monitoring of model in production is not required, i.e. data drift

## MVP Roadmap and Time Allocation

With a budget of 12 hours for this project, time will be distributed below to meet the deliverables:

1. Plan and scope project (10%)
2. Read existing approaches to similar problems (5%)
3. Perform exploratory data analysis on provided dataset (10%)
    * Set up project
    * Define the problem statement and evaluation metrics
    * Explore the dataset, understand distributions, and correlations with Auto Data Visualisation tools
    * Identify outliers
4. Train & Build model to predict on default payment (40%)
    * Apply data preprocessing and data cleaning, check for duplicates, etc.
    * Create basic end-to-end pipeline with minimal feature engineering and robust validation technique (K-folds cross-validation) and test a baseline model
        * Apply standardisation and scaling
        * Convert categorical variables with one-hot encoding (N-1)
    * Use AutoML framework to test many models and for automatic hyperparameter optmisation
    * Apply further feature engineering and class rebalancing / resampling
        * NOTE: Feature transformations and class resampling will only be applied to training splits to prevent data leakage
    * Reapply AutoML framework with new data
    * Create feature importance plots
5. Generate new synthetic dataset (10%)
6. Export requirements.txt
7. Create presentation (20%)
8. Update README (5%)
    * Add section on approach and considerations
    * Add set up instructions
    * Add repo structure
    * Update TOC

## Future Work

For tabular machine learning tasks, feature engineering can have the largest impact on performance - more time would be spent on:

* scaling techniques and dealing with outliers
* data resampling methods
* creating new features
  * feature transformations, such as log transformations
  * combining features

With more time, the following features could be explored or implemented:

* the constraints mentioned in above Constraints section
* manually test and evaluate models with hyperparameter optimisation with grid search, random search, or `Optuna` framework
* convert jupyter notebooks into python scripts
* set up `MLflow` and `Dagshub` for collaborative experiment and model tracking
* apply GitHub pre-commit hooks and GitHub Actions for automated testing and styling
* use Explainable AI techniques to understand the model, such as `SHAP` and `LIME` methods
* compare inference speeds of different models
* make the model deployable through a `REST API` using `FastAPI` or via a front end interface, such as `Flask` or `Streamlit`.
* containerise and deploy the model with Docker

## Set Up

1. Clone this repo
2. Create a new virtual environment with `venv`
3. Install requirements.txt

## Repo Structure

```bash
.
├── LICENSE
├── README.md
├── data
│   ├── processed
│   │   └── processed-data.csv
│   └── raw
│       └── credit_card_default.csv
├── notebooks
│   ├── 1_exploration-analysis.ipynb
│   ├── 1_exploration-auto-datavis.ipynb
│   ├── 2_model-training.ipynb
│   ├── 3_automl.ipynb
│   ├── 4_synthetic-data-generation.ipynb
├── outputs
│   ├── pandas-profiling-report.html
│   ├── synth_data_ctgan.csv
│   └── synth_data_gaussian.csv
├── requirements.txt
└── src
    ├── preprocessing.py
    └── training.py
```

## Tools

Tools used in this project:

* `ydata-profiling` (previously `pandas-profiling`) and `dabl` for quick exploratory data analysis and correlations on raw data
* `mljar-supervised` for AutoML, has inbuilt `Optuna` functionality for auto hyperparameter tuning
* `ydata-synthetic` for synthetic data generation with state-of-the-art neural networks for tabular and time-series data.

## Dataset Content

There are 25 variables:

* ID: ID of each client
* LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
* SEX: Gender (1=male, 2=female)
* EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* MARRIAGE: Marital status (1=married, 2=single, 3=others)
* AGE: Age in years
* PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
* PAY_2: Repayment status in August, 2005 (scale same as above)
* PAY_3: Repayment status in July, 2005 (scale same as above)
* PAY_4: Repayment status in June, 2005 (scale same as above)
* PAY_5: Repayment status in May, 2005 (scale same as above)
* PAY_6: Repayment status in April, 2005 (scale same as above)
* BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
* BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
* BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
* BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
* BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
* BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
* PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
* PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
* PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
* PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
* PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
* PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
* default.payment.next.month: Default payment (1=yes, 0=no)

## Authors and acknowledgment

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

The original dataset can be found here at the UCI Machine Learning Repository.

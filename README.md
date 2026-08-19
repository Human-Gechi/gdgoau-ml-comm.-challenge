# GDG OAU ML Community Challenge: Loan Default Prediction

This project is a loan default prediction task built around a credit-risk dataset used in the GDG OAU / Zindi-style machine learning challenge. The goal is to predict whether a customer will default on a loan (`target = 1`) or repay it (`target = 0`) using historical loan and customer information.

## Project objective

The main objective was to build a model that can estimate the probability of loan default from available transactional and customer-level data, while handling class imbalance, feature engineering, and out-of-distribution country effects.

## What I did

- Loaded the train, test, sample submission, and economic indicator datasets
- Inspected the target distribution and checked class imbalance
- Reviewed missing values, data types, and date fields
- Performed exploratory data analysis on:
  - `loan_type`
  - `lender_id`
  - `New_versus_Repeat`
  - `Total_Amount`
  - `duration`
  - `disbursement_date`
  - customer-level history
- Converted date columns and extracted temporal features
- Merged country-level economic indicators with the loan data by country and year
- Engineered new features, including:
  - repayment ratios
  - implied interest metrics
  - lender funding ratio
  - customer prior loan count and default history
  - recency, frequency, and loan velocity features
- Identified a useful `is_relevant` flag based on due-date vs. disbursement-date logic to isolate the portion of the data with the bulk of default risk
- Investigated train/test differences and recognized that Ghana appears in the test set but not in training, which creates an out-of-distribution problem
- Built categorical frequency and label-encoded features
- Trained models on the relevant subset of the data and evaluated them using cross-validation
- Used an ensemble of LightGBM and XGBoost models
- Blended model probabilities and forced non-relevant rows to zero default probability
- Generated a submission file for the challenge

## Key insights from the analysis

- `loan_type` was one of the strongest signals:
  - some loan types had very low default rates
  - rare loan types had much higher default risk
- `lender_id` strongly affected default behavior
- `New_versus_Repeat` showed that first-time borrowers were far riskier than repeat borrowers
- `Total_Amount` was highly skewed and large loan amounts had much higher default rates
- A date-based relevance filter captured most of the defaults in the training data
- The model had to account for country-level shifts, especially Ghanaian loans in the test set with no Ghana examples in training

## Modeling approach

I used a robust tabular machine learning workflow:

- Data cleaning and feature engineering
- Encoding of categorical variables
- Train/test feature alignment
- Stratified cross-validation
- LightGBM classifier
- XGBoost classifier
- Probability blending
- Threshold tuning for final binary classification

## Evaluation

The model was evaluated primarily using:

- ROC AUC
- F1-score
- confusion matrix
- threshold optimization for final decision-making

The notebook also reports the best threshold for converting predicted probabilities into binary labels.

## Tools used

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- LightGBM
- XGBoost

## How to run

1. Open the notebook in Jupyter
2. Ensure the required libraries are installed
3. Run all cells in order
4. The final output includes a submission file with ID and prediction columns

Example:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
jupyter notebook
```

## Project outcome

The final notebook builds a credit-risk model for the challenge by combining domain understanding, engineered features, and an ensemble of boosting models. The workflow is designed to capture both obvious loan-risk patterns and the more subtle temporal and behavioral signals that affect repayment probability.

## Notes

This challenge is a practical credit-risk and fraud-like classification task, where the main goal is not just prediction accuracy, but understanding which borrower and loan characteristics are most informative for default risk.

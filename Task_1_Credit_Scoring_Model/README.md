# CodeAlpha ML Internship — Task 1: Credit Scoring Model

## Objective
Predict an individual's creditworthiness from historical financial/application data using classification algorithms.

## Dataset
UCI Statlog (German Credit Data). The program downloads the dataset automatically from the UCI repository.

## Algorithms
- Logistic Regression
- Random Forest

## Main preprocessing
1. Separate numerical and categorical features.
2. Fill missing values using median/most-frequent strategies.
3. Standardize numerical features.
4. One-hot encode categorical features.
5. Split data into 80% training and 20% testing with stratification.

## Evaluation metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```

The script creates `model_results.csv` and confusion-matrix images.

> Note: This is an educational ML project, not a production credit-decision system.

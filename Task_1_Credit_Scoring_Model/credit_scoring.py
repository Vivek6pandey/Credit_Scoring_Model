"""
CodeAlpha ML Internship - Task 1: Credit Scoring Model
Dataset: UCI Statlog (German Credit Data)
Objective: Predict whether a credit applicant is good or bad credit risk.

The dataset is downloaded automatically from the UCI repository.
"""

import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_FILE = "german.data"

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "credit_risk"
]

def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading German Credit dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)

def main():
    df = download_data()

    # UCI target: 1 = good credit, 2 = bad credit.
    # Convert to 0 = bad, 1 = good for binary classification.
    df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["credit_risk"].value_counts())

    X = df.drop(columns="credit_risk")
    y = df["credit_risk"]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }
        results.append(metrics)

        print(f"\n===== {name} =====")
        print(classification_report(
            y_test, pred,
            target_names=["Bad Credit", "Good Credit"],
            zero_division=0
        ))

        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test, pred),
            display_labels=["Bad Credit", "Good Credit"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(name.lower().replace(" ", "_") + "_confusion_matrix.png", dpi=200)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nFiles created: model_results.csv and confusion-matrix PNG files.")

if __name__ == "__main__":
    main()

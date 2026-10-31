"""Beginner-friendly version of the main Telco churn analysis.

Run from the project folder:
    python src/simple_analysis.py

The full notebook still creates the project's existing figures and reports.
This script is intended to make the main Python workflow easy to explain.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import churn_lib as helpers


PROJECT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT / "data" / "Telco-Customer-Churn.csv"
RESULTS_FOLDER = PROJECT / "results"


def load_data():
    """Load the CSV file and apply the project's standard cleaning rules."""
    raw_data = pd.read_csv(DATA_FILE)
    return helpers.clean(raw_data)


def split_data(data):
    """Create the same stratified train/test split used by the project."""
    return train_test_split(
        data,
        test_size=0.20,
        stratify=data["ChurnFlag"],
        random_state=helpers.RANDOM_STATE,
    )


def regression_models():
    """Return the four regression models used in the analysis."""
    polynomial_model = Pipeline(
        [
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )

    return {
        "Simple Linear": helpers.make_pipeline(
            LinearRegression(), ["tenure"], categorical=[], scale=False
        ),
        "Multiple Linear": helpers.make_pipeline(
            LinearRegression(), helpers.REG_NUMERIC, scale=False
        ),
        "Polynomial": Pipeline(
            [
                (
                    "select_numeric",
                    helpers.preprocessor(
                        ["tenure", "MonthlyCharges"], categorical=[], scale=False
                    ),
                ),
                ("polynomial", polynomial_model),
            ]
        ),
        "Random Forest": helpers.make_pipeline(
            RandomForestRegressor(
                n_estimators=250,
                random_state=helpers.RANDOM_STATE,
                n_jobs=-1,
            ),
            helpers.REG_NUMERIC,
            scale=False,
        ),
    }


def compare_regression_models(train_data, test_data):
    """Train regression models and return a simple comparison table."""
    rows = []

    for name, model in regression_models().items():
        features = ["tenure"] if name == "Simple Linear" else helpers.REG_NUMERIC
        model.fit(train_data, train_data["TotalCharges"])
        predictions = model.predict(test_data)

        mse = mean_squared_error(test_data["TotalCharges"], predictions)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(test_data["TotalCharges"], predictions),
                "MSE": mse,
                "RMSE": mse**0.5,
                "R2": r2_score(test_data["TotalCharges"], predictions),
                "Features": ", ".join(features),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def classification_models():
    """Return the six classification models used in the analysis."""
    balanced = {"class_weight": "balanced", "random_state": helpers.RANDOM_STATE}

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, **balanced),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, **balanced),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            **balanced,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            class_weight="balanced",
            random_state=helpers.RANDOM_STATE,
        ),
    }


def compare_classification_models(train_data, test_data):
    """Train classifiers and return a simple comparison table."""
    rows = []
    feature_columns = [
        column
        for column in train_data.columns
        if column not in {"customerID", "Churn", "ChurnFlag", "TotalCharges"}
    ]
    numeric_columns = [column for column in helpers.NUMERIC_ALL if column != "TotalCharges"]

    for name, classifier in classification_models().items():
        model = helpers.make_pipeline(
            classifier,
            numeric_columns,
            categorical=helpers.CATEGORICAL,
            scale=name != "Decision Tree" and name != "Random Forest",
        )
        model.fit(train_data, train_data["ChurnFlag"])
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(test_data["ChurnFlag"], predictions),
                "AUC": roc_auc_score(test_data["ChurnFlag"], probabilities),
                "F1": f1_score(test_data["ChurnFlag"], predictions),
                "Features used": len(feature_columns),
            }
        )

    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def main():
    """Run the readable analysis and save two comparison tables."""
    data = load_data()
    train_data, test_data = split_data(data)

    print(f"Customers: {len(data):,}")
    print(f"Churn rate: {data['ChurnFlag'].mean():.1%}")
    print(f"Training rows: {len(train_data):,}")
    print(f"Test rows: {len(test_data):,}")

    regression_results = compare_regression_models(train_data, test_data)
    classification_results = compare_classification_models(train_data, test_data)

    RESULTS_FOLDER.mkdir(exist_ok=True)
    regression_results.to_csv(
        RESULTS_FOLDER / "simple_regression_comparison.csv", index=False
    )
    classification_results.to_csv(
        RESULTS_FOLDER / "simple_classification_comparison.csv", index=False
    )

    print("\nRegression comparison:")
    print(regression_results.round(3).to_string(index=False))
    print("\nClassification comparison:")
    print(classification_results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

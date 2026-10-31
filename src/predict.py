"""Score customers with BOTH saved models.

Usage (from the project root):
    python src/predict.py data/Telco-Customer-Churn.csv results/scored_customers.csv

Input: CSV with the original 20 feature columns (customerID optional; Churn optional).
Output: predicted TotalCharges (regression), churn probability and churn flag (classification),
and a priority group that combines the two.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import churn_lib as L


def main(src, dst):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    regression_model = joblib.load(
        os.path.join(root, "models", "final_regression_model.joblib")
    )
    classifier = joblib.load(
        os.path.join(root, "models", "final_churn_classifier.joblib")
    )
    with open(os.path.join(root, "models", "model_metadata.json")) as file:
        metadata = json.load(file)

    raw = pd.read_csv(src)
    if "Churn" not in raw.columns:
        raw["Churn"] = "No"

    df = L.clean(raw)
    predicted_revenue = regression_model.predict(df)
    churn_probability = classifier.predict_proba(df)[:, 1]

    output = pd.DataFrame(
        {
            "predicted_total_charges": predicted_revenue.round(2),
            "p_churn": churn_probability.round(4),
            "churn_flag": (
                churn_probability >= metadata["churn_threshold"]
            ).astype(int),
        }
    )

    if "customerID" in df.columns:
        output.insert(0, "customerID", df["customerID"].values)

    high_risk = output.p_churn >= metadata["churn_threshold"]
    high_value = output.predicted_total_charges >= metadata["revenue_median"]
    output["priority_group"] = np.select(
        [high_risk & high_value, high_risk, high_value],
        [
            "High risk, high revenue",
            "High risk, lower revenue",
            "Low risk, high revenue",
        ],
        "Low risk, lower revenue",
    )

    output.to_csv(dst, index=False)
    print(f"Scored {len(output)} customers -> {dst}")
    print(output.priority_group.value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

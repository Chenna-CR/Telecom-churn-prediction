"""Score customers with BOTH saved models.

Usage (from the project root):
    python src/predict.py data/Telco-Customer-Churn.csv results/scored_customers.csv

Input: CSV with the original 20 feature columns (customerID optional; Churn optional).
Output: predicted TotalCharges (regression), churn probability and churn flag (classification),
and a priority group that combines the two.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
import churn_lib as L


def main(src, dst):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg = joblib.load(os.path.join(root, 'models', 'final_regression_model.joblib'))
    clf = joblib.load(os.path.join(root, 'models', 'final_churn_classifier.joblib'))
    meta = json.load(open(os.path.join(root, 'models', 'model_metadata.json')))
    raw = pd.read_csv(src)
    if 'Churn' not in raw.columns:
        raw['Churn'] = 'No'                      # placeholder so that the shared cleaning code runs
    df = L.clean(raw)
    pred_rev = reg.predict(df)
    p = clf.predict_proba(df)[:, 1]
    out = pd.DataFrame({'predicted_total_charges': pred_rev.round(2), 'p_churn': p.round(4), 'churn_flag': (p >= meta['churn_threshold']).astype(int)})
    if 'customerID' in df.columns:
        out.insert(0, 'customerID', df['customerID'].values)
    hi_r = out.p_churn >= meta['churn_threshold']; hi_v = out.predicted_total_charges >= meta['revenue_median']
    out['priority_group'] = np.select([hi_r & hi_v, hi_r, hi_v], ['High risk, high revenue', 'High risk, lower revenue', 'Low risk, high revenue'], 'Low risk, lower revenue')
    out.to_csv(dst, index=False)
    print(f'Scored {len(out)} customers -> {dst}')
    print(out.priority_group.value_counts().to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

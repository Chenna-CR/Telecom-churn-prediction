"""churn_lib.py - reusable helpers for the Telco project (syllabus-aligned).

Part A: regression (predict TotalCharges)   Part B: classification (predict Churn)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC_ALL = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
REG_TARGET = "TotalCharges"
REG_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges"]          # TotalCharges is the target in Part A


def load_raw(path):
    return pd.read_csv(path)


def clean(df):
    """Fix types. The 11 blank TotalCharges belong to tenure 0 (not billed yet) -> 0.0."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].astype(str).str.strip().replace("", np.nan))
    out.loc[out["TotalCharges"].isna() & (out["tenure"] == 0), "TotalCharges"] = 0.0
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out["ChurnFlag"] = (out["Churn"] == "Yes").astype(int)
    return out


def preprocessor(numeric, categorical=CATEGORICAL, scale=True):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
                             verbose_feature_names_out=False)


def make_pipeline(model, numeric, categorical=CATEGORICAL, scale=True):
    return Pipeline([("prep", preprocessor(numeric, categorical, scale)), ("model", model)])


def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return dict(MAE=mean_absolute_error(y, pred), MSE=mse, RMSE=float(np.sqrt(mse)), R2=r2_score(y, pred))


def clf_metrics(y, pred, proba):
    return dict(Accuracy=accuracy_score(y, pred), LogLoss=log_loss(y, proba), AUC=roc_auc_score(y, proba),
                Precision=precision_score(y, pred, zero_division=0), Recall=recall_score(y, pred), F1=f1_score(y, pred))

# Predicting Telecom Customer Revenue and Churn (CA2 project)

Predictive-analysis project on the IBM *Telco Customer Churn* data, using only methods from the course syllabus (Units I to III).

> **Data note.** The data are IBM's sample data for a *fictional* telecom company (7,043 customers). Results demonstrate a method; they are not facts about a real operator.

## What the project does
**Part A - Regression (predict `TotalCharges`, a customer's lifetime revenue):** Simple Linear, Multiple Linear, Polynomial and Random Forest regression, compared with MAE, MSE, RMSE and R-squared. The best model (selected on validation RMSE) is used for prediction.
**Part B - Classification (predict `Churn`):** Logistic Regression, k-Nearest Neighbours, Naive Bayes, Decision Tree, Random Forest and SVM, compared with accuracy, log loss, AUC, precision, recall, F1 and confusion matrices. The best classifier (selected on validation F1) is used for prediction.
**Combined use:** predicted revenue x churn probability gives four customer priority groups.

| ID | Objective |
|----|-----------|
| O1 | Data quality, preprocessing, correlations, churn by segment |
| O2 | Simple Linear Regression (OLS by formula and by scikit-learn) |
| O3 | Multiple and Polynomial Regression (degree chosen on validation) |
| O4 | Compare four regression models; select the best |
| O5 | Compare six classifiers |
| O6 | Select best classifier; decision-tree rules, Random Forest importance, error profile |
| O7 | Use both selected models together |

## Project structure
```
data/        Telco-Customer-Churn.csv, DATASET_README.md
notebooks/   churn_prediction.ipynb (executed), build_notebook.py
src/         churn_lib.py (helpers), predict.py (score new customers with both models)
figures/     17 PNG figures
results/     all tables (CSV), summary.json, test predictions
models/      final_regression_model.joblib, final_churn_classifier.joblib, model_metadata.json
report/      Predictive_Analysis_CA2_Report.docx (+ PDF preview) and generator scripts
visual/      churn_visual_experience.html (interactive page, open in any browser)
docs/        Viva_Preparation_Guide, Screenshot_and_Submission_Checklist
```

## Setup and run
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb       # Run All (about 2 minutes)
python src/predict.py data/Telco-Customer-Churn.csv results/scored_customers.csv
```
Use the versions in `results/environment_versions.json` (scikit-learn 1.8.0) so the saved model files load.

## Method in brief
- One stratified 80/20 train/test split (seed 42); the training set is split again 75/25 into fit and validation sets. The test set is used once.
- Scaling and encoding are inside scikit-learn Pipelines (learned from training data only).
- Settings (polynomial degree, forest depth, k, tree depth, C) are chosen on the validation set.
- Selection rules were fixed in advance: lowest validation RMSE (regression) and highest validation F1 (classification).

## Headline results
- Churn rate 26.5%; month-to-month 42.7% vs two-year 2.8%.
- Regression (test): Simple Linear R2 0.657 (RMSE 1,297); Multiple Linear R2 0.900 (RMSE 702); **Polynomial degree 2 R2 0.999 (RMSE 66.2, MAE 44.9)**; Random Forest RMSE 79.5. TotalCharges is almost tenure x monthly charge, so the near-perfect fit is expected.
- Classification (test): best classifier **Random Forest**: accuracy 0.769, AUC 0.841, precision 0.548, recall 0.754, F1 0.634. The six models are close (F1 0.58 to 0.63).
- Combined: 165 test customers are both high risk and high revenue.

## Dataset source
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv (also on Kaggle, blastchar/telco-customer-churn, and OpenML 42178). Check licence terms before redistributing.

## Upload to GitHub
```
git init
git add .
git commit -m "Predicting telecom customer revenue and churn (CA2)"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Needs your own GitHub account; see `docs/Screenshot_and_Submission_Checklist.md`.)

## Report
`report/Predictive_Analysis_CA2_Report.docx` follows the LPU CA2 format. Fill the yellow placeholders, then in Word press Ctrl+A and F9 to refresh the contents, figure and table lists.

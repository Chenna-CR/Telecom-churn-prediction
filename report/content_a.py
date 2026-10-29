# -*- coding: utf-8 -*-
"""Shared helpers + chapters 1-3. Every number is read from ../results so the report matches the notebook."""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = f'{ROOT}/results'; FIGD = f'{ROOT}/figures'
S = json.load(open(f'{RES}/summary.json'))
ENV = json.load(open(f'{RES}/environment_versions.json'))
rd = lambda f: pd.read_csv(f'{RES}/{f}.csv')
B = []

def h1(t): B.append(dict(t='h1', text=t))
def h2(t): B.append(dict(t='h2', text=t))
def h3(t): B.append(dict(t='h3', text=t))
def p(t): B.append(dict(t='p', text=t))
def bl(items): B.append(dict(t='bullets', items=items))
def eq(t): B.append(dict(t='eq', text=t))
def table(caption, head, rows, widths=None, align=None, note=None):
    B.append(dict(t='table', caption=caption, head=head, rows=[[str(c) for c in r] for r in rows], widths=widths, align=align, note=note))
def fig(name, caption, w=6.0):
    B.append(dict(t='fig', file=f'{FIGD}/{name}.png', caption=caption, w=w))

def pf(x): return '< 0.0001' if float(x) < 0.0001 else f'{float(x):.4f}'
def f2(x): return f'{float(x):.2f}'
def f3(x): return f'{float(x):.3f}'
def f4(x): return f'{float(x):.4f}'
def pct(x, d=1): return f'{100*float(x):.{d}f}%'
def usd(x): return f'USD {float(x):,.0f}'

N, NC, NS, CR = S['n_rows'], S['n_churn'], S['n_stay'], S['churn_rate']
sp = S['split']; q = S['quality']
BEST_REG, BEST_CLF = S['o4_best_regression'], S['o6_best_classifier']
BEST_REG_NAME = BEST_REG.split('. ', 1)[1] + ' Regression'

# ============================== CHAPTER 1 =====================================================
h1('1. INTRODUCTION')
h2('1.1 Background and motivation')
p('Telecommunications companies sell services on recurring plans, so the value of a customer builds up month after month, and a departure removes all the revenue that would have followed. '
  'Two practical questions follow. First, *how much revenue does a customer generate over their life with the company?* Second, *which customers are likely to leave?* '
  'Both are prediction problems. The first asks for a number and is solved with regression; the second asks for a yes/no answer and is solved with classification. '
  'Together they let a company focus its retention effort on customers who are both at risk and valuable.')
p('This project uses the IBM Telco Customer Churn data set (7,043 customers, 20 attributes) to study both problems with the methods of a first course in predictive analytics: data preprocessing, correlation analysis, '
  'simple, multiple and polynomial regression, logistic regression, k-nearest neighbours, Naive Bayes, decision trees, Random Forest and support vector machines, evaluated with MAE, MSE, RMSE, R-squared, accuracy, log loss, AUC, precision, recall, F1 score and the confusion matrix.')
h2('1.2 Problem statement')
p(f'**Part A (regression).** Predict the lifetime revenue of a customer, measured by *TotalCharges* (USD billed to date), from the customer\'s account and service attributes, using four regression models, and select the best model for prediction. '
  f'**Part B (classification).** Predict whether a customer will churn (*Churn* = Yes; {NC:,} of {N:,} customers, {pct(CR)}), using six classifiers, and select the best classifier. '
  'Finally, combine the two selected models to rank customers by risk and revenue.')
h2('1.3 Objectives')
p('Seven specific, measurable and non-overlapping objectives were fixed before modelling. Each is answered with actual results in Chapter 4.')
objs = [
 ('O1', 'Understand the data: quality, correlations and churn by segment.', 'Preprocessing, Pearson correlation, group comparison', 'Clean data set, ranked relationships'),
 ('O2', 'Model TotalCharges from tenure with Simple Linear Regression.', 'Ordinary Least Squares, residual analysis', 'Slope, intercept, MAE, RMSE, R-squared'),
 ('O3', 'Improve the model with Multiple and Polynomial Regression.', 'OLS with several inputs, polynomial features', 'Lower validation RMSE'),
 ('O4', 'Compare four regression models and select the best for prediction.', 'MAE, MSE, RMSE, R-squared; train versus test', 'Lowest validation RMSE without overfitting'),
 ('O5', 'Compare six classifiers for churn.', 'Logistic Regression, kNN, Naive Bayes, Decision Tree, Random Forest, SVM', 'Accuracy, log loss, AUC, precision, recall, F1'),
 ('O6', 'Select the best classifier and study its rules and errors.', 'Tree rules, Random Forest importance, confusion matrix', 'Best validation F1; error profile'),
 ('O7', 'Use both selected models together to predict for customers.', 'Predicted revenue and churn probability', 'Four customer priority groups'),
]
table('Research objectives and how success is measured', ['ID', 'Objective', 'Methods', 'Success measure'], objs, widths=[.06, .34, .32, .28], align=['c', 'l', 'l', 'l'])
h2('1.4 Objective-to-analysis mapping')
table('Mapping of objectives to variables, outputs and report locations', ['ID', 'Key variables', 'Visual output', 'Report section'],
      [('O1', 'All 19 attributes, correlations of tenure, MonthlyCharges, TotalCharges and churn', 'Figures 2 to 4', '4.1'),
       ('O2', 'tenure (input), TotalCharges (target)', 'Figure 5', '4.2'),
       ('O3', 'SeniorCitizen, tenure, MonthlyCharges and 15 categorical inputs; polynomial degree 1 to 5', 'Figure 6', '4.3'),
       ('O4', 'Four regression models; MAE, MSE, RMSE, R-squared', 'Figures 7 to 9', '4.4'),
       ('O5', 'Churn label; six classifiers; accuracy, log loss, AUC, precision, recall, F1', 'Figures 10 to 13', '4.5'),
       ('O6', 'Selected classifier; decision-tree rules; Random Forest importance', 'Figures 14 to 16', '4.6'),
       ('O7', 'Predicted TotalCharges and churn probability', 'Figure 17', '4.7')], widths=[.07, .50, .23, .20], align=['c', 'l', 'l', 'c'])
h2('1.5 Link to the course syllabus')
table('Where each syllabus topic is used', ['Unit', 'Syllabus topic', 'Where used in this project'],
      [('I', 'Introduction to predictive analytics; supervised learning; data preprocessing', 'Chapters 1 to 3 (cleaning, encoding, scaling, splitting)'),
       ('II', 'Simple, multiple and polynomial regression; OLS; correlation', 'O1 (correlation), O2, O3, O4'),
       ('II', 'MAE, MSE, RMSE, R-squared', 'O2, O3, O4'),
       ('II', 'Logistic regression', 'O5, O6 (churn classifier)'),
       ('III', 'k-nearest neighbours (lazy learning), Naive Bayes, decision trees and rules, SVM', 'O5, O6'),
       ('III', 'Random Forest (ensemble of decision trees)', 'O4 (regression) and O5, O6 (classification)'),
       ('III', 'Accuracy, log loss, AUC, precision, recall, F1, confusion matrix', 'O5, O6')], widths=[.08, .50, .42], align=['c', 'l', 'l'])
h2('1.6 Tools and environment')
p(f'The analysis used Python {ENV["python"]} with pandas {ENV["pandas"]}, NumPy {ENV["numpy"]}, SciPy {ENV["scipy"]}, scikit-learn {ENV["scikit_learn"]}, Matplotlib {ENV["matplotlib"]} and seaborn {ENV["seaborn"]}. '
  'The work is a Jupyter Notebook (notebooks/churn_prediction.ipynb) that imports reusable code from src/churn_lib.py. A fixed random seed (42) makes every result reproducible.')
h2('1.7 Organisation of the report')
p('Chapter 2 documents the source of the data. Chapter 3 describes the exploratory process: data audit, cleaning, preprocessing and the train/validation/test design. '
  'Chapter 4 contains the seven analyses, each with Introduction, General Description, Specific Requirements with Functions and Formulas, Analysis Results, and Visualization. '
  'Chapters 5 and 6 give the conclusion and future scope, and Chapter 7 lists the references.')

# ============================== CHAPTER 2 =====================================================
h1('2. SOURCE OF DATASET')
h2('2.1 Provenance')
p('The data set is the *Telco Customer Churn* sample data published by IBM as part of its Cognos Analytics sample data sets. It describes a **fictional** telecommunications company that provides home phone and internet services. '
  'IBM generated it as a teaching and benchmarking resource, so it is realistic in structure but is not a record of the customers of a real operator. Findings describe patterns in a benchmark data set and should not be presented as facts about any real company.')
table('Sources of the dataset', ['Role', 'Source', 'URL'],
      [('Downloaded for this project', 'IBM telco-customer-churn-on-icp4d repository (GitHub)', 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'),
       ('Popular listing', 'Kaggle: "Telco Customer Churn" (blastchar)', 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn'),
       ('Description by IBM', 'IBM Community blog: Telco customer churn', 'https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113'),
       ('Mirror', 'OpenML data set "telco-customer-churn"', 'https://www.openml.org/d/42178')], widths=[.2, .35, .45], align=['l', 'l', 'l'])
p(f'The downloaded file is 970,457 bytes, has MD5 checksum {S["md5"]}, and contains {N:,} rows and {S["n_cols"]} columns. The licence terms should be confirmed on the Kaggle and IBM pages before the file is redistributed; a copy is included in the project folder only so that the academic work is reproducible.')
h2('2.2 Description of the data')
p('Each row is one customer. The columns fall into four groups: the churn label; the services subscribed to (phone, multiple lines, internet type, online security, online backup, device protection, technical support, streaming TV and streaming movies); account information (tenure, contract, payment method, paperless billing, monthly and total charges); and demographics (gender, senior-citizen flag, partner and dependents).')
table('Data dictionary', ['Column', 'Type', 'Description'], rd('data_dictionary').values.tolist(), widths=[.2, .17, .63], align=['l', 'l', 'l'])
h2('2.3 Why this dataset was chosen')
bl(['**Two natural prediction targets.** TotalCharges is a continuous number (regression) and Churn is a binary label (classification), so one data set serves Units II and III of the syllabus.',
    '**Right size.** 7,043 rows are enough for stable train, validation and test sets and small enough to run every analysis within minutes.',
    '**Mixed variable types.** Numeric, binary and multi-level categorical columns call for encoding and scaling (Unit I).',
    '**Interpretable domain.** Contract, tenure and charges have direct business meaning, so results are easy to explain.'])
h2('2.4 Limitations of the dataset')
bl(['The company is fictional and the data are a single snapshot; there are no dates, so true temporal validation is impossible.',
    '*Churn* means leaving within the last month, so the rate is a point-in-time proportion, not a rate per period.',
    'There is no usage, network-quality, complaint or competitor information, which real churn models rely on.',
    'TotalCharges is billing history up to the snapshot. It is almost arithmetic in tenure and monthly charge (Section 4.3), so predicting it is a good exercise in regression but is not a forecast of *future* revenue.'])

# ============================== CHAPTER 3 =====================================================
h1('3. EDA PROCESS')
h2('3.1 Data inspection')
p(f'The file has {N:,} rows and {S["n_cols"]} columns: one identifier, one target (Churn), 16 categorical or binary attributes, and three numeric attributes (tenure, MonthlyCharges, TotalCharges) plus SeniorCitizen stored as 0/1. '
  'Read with default settings, pandas finds no missing values, but TotalCharges is stored as text, which signals that a closer audit is needed.')
d = S['desc']
table('Descriptive statistics of the numeric attributes', ['Variable', 'Mean', 'Std. dev.', 'Min', '25%', 'Median', '75%', 'Max'],
      [(k, f'{v["mean"]:,.2f}', f'{v["std"]:,.2f}', f'{v["min"]:,.2f}', f'{v["25%"]:,.2f}', f'{v["50%"]:,.2f}', f'{v["75%"]:,.2f}', f'{v["max"]:,.2f}') for k, v in d.items()],
      widths=[.19, .11, .12, .1, .11, .12, .12, .13], align=['l'] + ['r'] * 7)
h2('3.2 Data quality assessment')
table('Data-quality checks and decisions', ['Check', 'Finding', 'Decision'],
      [('Missing cells as read', str(q['missing_cells_as_read']), 'None to impute at read time'),
       ('Blank TotalCharges', f'{q["blank_totalcharges"]} rows, all with tenure = {q["blank_totalcharges_tenure_values"][0]}', 'Set to 0.0: a customer with no completed month has not been billed yet (logical rule)'),
       ('Duplicate customer IDs', str(q['duplicate_ids']), 'No action'),
       ('Duplicate feature rows (ignoring ID)', str(q['duplicate_rows']), 'Kept: identical attribute combinations can belong to different customers'),
       ('Inconsistent labels', f'"No internet service" in {q["no_internet_service_rows"]:,} rows and "No phone service" in {q["no_phone_service_rows"]:,}; consistent with InternetService and PhoneService', 'Kept as an explicit category'),
       ('Outliers (1.5 x IQR rule)', 'tenure: 0, MonthlyCharges: 0, TotalCharges: 0', 'No outliers to treat')], widths=[.24, .40, .36], align=['l', 'l', 'l'])
p(f'Only the TotalCharges blanks needed action. The {q["duplicate_rows"]} duplicated feature rows were kept because deleting them would remove genuine customers. No numeric column has values outside the 1.5 x IQR fences, so no outlier treatment was applied.')
h2('3.3 Preprocessing: encoding and scaling')
p(f'Machine-learning models need numbers. The 15 categorical attributes are converted to 0/1 indicator columns (one-hot encoding), and numeric inputs are standardised (z = (x - mean) / standard deviation) for the models that depend on distances or gradients (linear models, kNN, Naive Bayes, SVM). Tree-based models do not need scaling. '
  f'After encoding, the inputs of the classification models have {S["n_features_after_encoding"]} columns. Both steps are placed inside scikit-learn Pipelines, so the means, standard deviations and categories are learned from training data only.')
h2('3.4 Target variables')
p(f'In Part B the target is imbalanced: {NS:,} customers stayed ({pct(1-CR)}) and {NC:,} churned ({pct(CR)}). A model that always predicts "stay" would reach {pct(1-CR)} accuracy and find no churner, so accuracy alone is not enough; precision, recall, F1 and AUC are reported as well, and a majority-class baseline shows what "no skill" looks like. '
  'In Part A the target is TotalCharges, a right-skewed number from 0 to about 8,700 USD.')
fig('fig01_target_distribution', 'Distribution of the churn target', 4.4)
h2('3.5 Train, validation and test design')
p(f'The data were split once, with stratification on the churn label, into a training set of {sp["n_train"]:,} customers and a test set of {sp["n_test"]:,} customers (20%). For choosing between models and settings, the training set was split again into a fit set of {sp["n_fit"]:,} customers (75%) and a validation set of {sp["n_val"]:,} customers (25%). '
  'The churn rate is about 26.5% in every part. The rules that prevent leakage are:')
bl(['The test set is touched once, after all choices are made, to report final scores.',
    'Settings such as the polynomial degree, the number of neighbours k or the tree depth are chosen on the validation set only.',
    'After a choice is made, the chosen model is refitted on the full training set.',
    'Scaling and encoding are learned inside Pipelines from the training data only.',
    'The identifier column is dropped; TotalCharges is never used as an input in Part A, and Churn is never used as an input in either part.'])

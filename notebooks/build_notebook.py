"""Builds notebooks/churn_prediction.ipynb (syllabus-aligned: Units I-III)."""
import nbformat as nbf

nb = nbf.v4.new_notebook(); cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md(r"""
# Predictive Analysis of Telecom Customers: Regression and Classification
**CA2 Project - Predictive Analytics (Units I, II and III)**

Dataset: IBM *Telco Customer Churn* sample data (7,043 customers x 21 columns; a fictional company).

* **Part A - Regression (Unit II):** predict `TotalCharges` (lifetime revenue of a customer) with four models: Simple Linear, Multiple Linear, Polynomial and Random Forest regression. Metrics: MAE, MSE, RMSE, R-squared.
* **Part B - Classification (Unit III):** predict `Churn` with Logistic Regression, k-Nearest Neighbours, Naive Bayes, Decision Tree, Random Forest and SVM. Metrics: accuracy, log loss, AUC, precision, recall, F1, confusion matrix.

| ID | Objective | Methods | Success measure |
|----|-----------|---------|-----------------|
| O1 | Understand the data: quality, correlations and churn by segment | Preprocessing, Pearson correlation, group comparison | Clean data set, ranked relationships |
| O2 | Model TotalCharges with Simple Linear Regression | OLS (closed form), residuals | Slope, intercept, MAE, RMSE, R2 |
| O3 | Improve with Multiple and Polynomial Regression | OLS, polynomial features | Lower validation RMSE |
| O4 | Compare four regression models and select the best for prediction | MAE, MSE, RMSE, R2, train vs test | Lowest validation RMSE, no overfitting |
| O5 | Compare six classifiers for churn | LR, kNN, NB, DT, RF, SVM | Accuracy, log loss, AUC, P, R, F1 |
| O6 | Select the best classifier and study its rules and errors | Tree rules, RF importance, confusion matrix | Best F1 on validation, error profile |
| O7 | Use both selected models together to predict for customers | Predicted revenue x churn probability | Customer priority groups |
""")

code(r"""
import sys, os, json, warnings, hashlib
sys.path.insert(0, os.path.abspath('../src')); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display
import churn_lib as L
RS = L.RANDOM_STATE; np.random.seed(RS)
FIG, RES, MOD = '../figures', '../results', '../models'
for d in (FIG, RES, MOD): os.makedirs(d, exist_ok=True)
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'axes.titleweight': 'bold', 'font.size': 11.5, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.frameon': False})
C_STAY, C_CHURN, C_ACC, C_GRN = '#2E6F95', '#D1495B', '#EDAE49', '#4C9F70'
SUMMARY = {}
def save_fig(name):
    plt.tight_layout(); plt.savefig(f'{FIG}/{name}.png', bbox_inches='tight'); plt.show()
""")

md("## 1. Data acquisition and inspection (Unit I)")
code(r"""
DATA = '../data/Telco-Customer-Churn.csv'
raw = L.load_raw(DATA)
SUMMARY['md5'] = hashlib.md5(open(DATA, 'rb').read()).hexdigest(); SUMMARY['n_rows'], SUMMARY['n_cols'] = raw.shape
print(raw.shape, SUMMARY['md5']); display(raw.head()); raw.info()
""")
code(r"""
dictionary = pd.DataFrame([
 ('customerID','Identifier','Unique customer ID (dropped before modelling)'),('gender','Categorical','Female / Male'),
 ('SeniorCitizen','Binary','1 if senior citizen'),('Partner','Binary','Has a partner'),('Dependents','Binary','Has dependents'),
 ('tenure','Numeric (months)','Months with the company'),('PhoneService','Binary','Has phone service'),
 ('MultipleLines','Categorical','Yes / No / No phone service'),('InternetService','Categorical','DSL / Fiber optic / No'),
 ('OnlineSecurity','Categorical','Yes / No / No internet service'),('OnlineBackup','Categorical','Yes / No / No internet service'),
 ('DeviceProtection','Categorical','Yes / No / No internet service'),('TechSupport','Categorical','Yes / No / No internet service'),
 ('StreamingTV','Categorical','Yes / No / No internet service'),('StreamingMovies','Categorical','Yes / No / No internet service'),
 ('Contract','Categorical','Month-to-month / One year / Two year'),('PaperlessBilling','Binary','Uses paperless billing'),
 ('PaymentMethod','Categorical','Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)'),
 ('MonthlyCharges','Numeric (USD)','Amount charged per month'),('TotalCharges','Numeric (USD)','Total charged to date (text in the file); regression target in Part A'),
 ('Churn','Target','Yes if the customer left within the last month; classification target in Part B')], columns=['Column', 'Type', 'Description'])
dictionary.to_csv(f'{RES}/data_dictionary.csv', index=False); display(dictionary)
""")

md("## 2. Data quality assessment and preprocessing (Unit I)")
code(r"""
q = {}
q['missing_cells_as_read'] = int(raw.isna().sum().sum())
q['duplicate_rows'] = int(raw.drop(columns='customerID').duplicated().sum()); q['duplicate_ids'] = int(raw.customerID.duplicated().sum())
blank = raw['TotalCharges'].astype(str).str.strip() == ''
q['blank_totalcharges'] = int(blank.sum()); q['blank_totalcharges_tenure_values'] = sorted(raw.loc[blank, 'tenure'].unique().tolist())
q['no_internet_service_rows'] = int((raw.InternetService == 'No').sum()); q['no_phone_service_rows'] = int((raw.PhoneService == 'No').sum())
print(json.dumps(q, indent=1)); SUMMARY['quality'] = q
df = L.clean(raw)
rows = []
for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1; lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    rows.append([c, lo, hi, int(((df[c] < lo) | (df[c] > hi)).sum()), float(df[c].skew())])
outl = pd.DataFrame(rows, columns=['Variable', 'Lower fence', 'Upper fence', 'Outliers (IQR)', 'Skewness']).round(2)
outl.to_csv(f'{RES}/outliers.csv', index=False); display(outl); SUMMARY['outliers'] = outl.set_index('Variable')['Outliers (IQR)'].to_dict()
desc = df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().T.round(2); desc.to_csv(f'{RES}/descriptive_statistics.csv'); display(desc)
SUMMARY['desc'] = desc.to_dict('index')
print('Remaining missing:', int(df.isna().sum().sum())); df.to_csv(f'{RES}/cleaned_dataset.csv', index=False)
""")
code(r"""
# Encoding and scaling demonstration (what the Pipelines do inside every model)
prep_demo = L.preprocessor(['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']).fit(df)
SUMMARY['n_features_after_encoding'] = int(len(prep_demo.get_feature_names_out()))
print('Columns after one-hot encoding + scaling:', SUMMARY['n_features_after_encoding'])
vc = df.Churn.value_counts(); SUMMARY['churn_rate'] = float(df.ChurnFlag.mean()); SUMMARY['n_churn'], SUMMARY['n_stay'] = int(vc['Yes']), int(vc['No'])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(['Stayed', 'Churned'], [vc['No'], vc['Yes']], color=[C_STAY, C_CHURN], width=.55)
for b, v in zip(bars, [vc['No'], vc['Yes']]): ax.text(b.get_x()+b.get_width()/2, v+60, f'{v:,}\n({v/len(df):.1%})', ha='center', va='bottom')
ax.set_ylim(0, vc.max()*1.2); ax.set_ylabel('Customers'); ax.set_title('Target distribution: customer churn')
save_fig('fig01_target_distribution')
""")

md("""## 3. Train / validation / test design
One stratified 80/20 split gives the **training set** and the **test set**. The test set is used only once, for final scores. For choosing between models and settings (for example the polynomial degree or *k* in kNN), the training set is split again into 75 % *fit* and 25 % *validation*. All scaling and encoding sit inside Pipelines so they are learned from training data only.""")
code(r"""
from sklearn.model_selection import train_test_split
FEATS = [c for c in df.columns if c not in ('customerID', 'Churn', 'ChurnFlag')]
tr, te = train_test_split(df, test_size=0.20, stratify=df.ChurnFlag, random_state=RS)
fit, val = train_test_split(tr, test_size=0.25, stratify=tr.ChurnFlag, random_state=RS)
print(len(fit), len(val), len(te)); SUMMARY['split'] = dict(n_fit=len(fit), n_val=len(val), n_test=len(te), n_train=len(tr),
    churn_fit=float(fit.ChurnFlag.mean()), churn_val=float(val.ChurnFlag.mean()), churn_test=float(te.ChurnFlag.mean()))
""")

# ---------------------------------------------------------------- O1
md("""## O1. Understanding the data: correlations and churn by segment
**Question.** Which variables move together, and which customer groups churn most? **Methods.** Pearson correlation, group means, churn rate by category (descriptive).""")
code(r"""
cats = L.CATEGORICAL + ['SeniorCitizen']
rows = []
for c in cats:
    r = df.groupby(c).ChurnFlag.agg(['mean', 'size']); hi, lo = r['mean'].idxmax(), r['mean'].idxmin()
    rows.append([c, str(hi), r['mean'].max(), str(lo), r['mean'].min(), r['mean'].max() - r['mean'].min()])
spread = pd.DataFrame(rows, columns=['Variable', 'Highest-churn group', 'Rate', 'Lowest-churn group', 'Rate (low)', 'Spread']).sort_values('Spread', ascending=False).reset_index(drop=True)
spread.round(4).to_csv(f'{RES}/o1_churn_rate_spread.csv', index=False); display(spread.round(3))
SUMMARY['o1_spread_top'] = spread.head(6).round(4).to_dict('records'); SUMMARY['o1_spread_bottom'] = spread.tail(3).round(4).to_dict('records')
means = df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean().round(2); means.to_csv(f'{RES}/o1_group_means.csv'); display(means)
SUMMARY['o1_means'] = means.to_dict('index')
corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'ChurnFlag']].corr(method='pearson'); corr.round(4).to_csv(f'{RES}/o1_pearson_correlation.csv'); display(corr.round(3))
SUMMARY['corr'] = corr.round(4).to_dict()
""")
code(r"""
key = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling', 'SeniorCitizen', 'Dependents']
fig, axes = plt.subplots(4, 2, figsize=(8.6, 11.5), sharey=True); overall = df.ChurnFlag.mean()
for ax, c in zip(axes.ravel(), key):
    r = df.groupby(c).ChurnFlag.mean().sort_values(ascending=False)
    labels = [str(i).replace(' (automatic)', '\n(auto)').replace('Month-to-month', 'Month-to-\nmonth').replace('Electronic check', 'Electronic\ncheck').replace('Mailed check', 'Mailed\ncheck').replace('No internet service', 'No\ninternet').replace('Fiber optic', 'Fiber\noptic') for i in r.index]
    ax.bar(labels, r.values, color=[C_CHURN if v > overall else C_STAY for v in r.values], width=.65); ax.axhline(overall, color='k', ls='--', lw=1)
    for i, v in enumerate(r.values): ax.text(i, v+.01, f'{v:.0%}', ha='center', fontsize=10.5)
    ax.set_title(c); ax.tick_params(axis='x', labelsize=10)
[axes[i, 0].set_ylabel('Churn rate') for i in range(4)]
fig.suptitle(f'Churn rate by segment (dashed = overall {overall:.1%})', fontweight='bold', y=1.01); save_fig('fig02_churn_rate_by_segment')
fig, ax = plt.subplots(figsize=(6.2, 5)); sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, cbar_kws={'shrink': .8}, ax=ax)
ax.set_title('Pearson correlation matrix'); save_fig('fig03_correlation_heatmap')
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, c in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    sns.histplot(data=df, x=c, hue='Churn', stat='density', common_norm=False, element='step', bins=30, palette={'No': C_STAY, 'Yes': C_CHURN}, ax=ax); ax.set_title(c)
save_fig('fig04_numeric_distributions')
""")

# ---------------------------------------------------------------- O2
md("""## O2. Simple Linear Regression: TotalCharges from tenure
**Question.** How much of a customer's lifetime revenue is explained by tenure alone? **Method.** Ordinary Least Squares: slope and intercept from the closed-form formulas, checked against scikit-learn.""")
code(r"""
from sklearn.linear_model import LinearRegression
x, y = tr['tenure'].values.astype(float), tr['TotalCharges'].values
b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum(); b0 = y.mean() - b1 * x.mean()
slr = L.make_pipeline(LinearRegression(), ['tenure'], categorical=[], scale=False).fit(tr, tr.TotalCharges)
lr_ = slr.named_steps['model']
print('Closed form: intercept %.4f slope %.4f | sklearn: intercept %.4f slope %.4f' % (b0, b1, lr_.intercept_, lr_.coef_[0]))
SUMMARY['o2_ols'] = dict(slope=float(b1), intercept=float(b0), slope_sklearn=float(lr_.coef_[0]), intercept_sklearn=float(lr_.intercept_), r_pearson=float(np.corrcoef(x, y)[0, 1]))
m_tr = L.reg_metrics(tr.TotalCharges, slr.predict(tr)); m_te = L.reg_metrics(te.TotalCharges, slr.predict(te))
o2 = pd.DataFrame({'Train': m_tr, 'Test': m_te}).T.round(3); o2.to_csv(f'{RES}/o2_slr_metrics.csv'); display(o2); SUMMARY['o2_metrics'] = o2.to_dict('index')
# normal-equation check (matrix form used for multiple regression)
Xd = np.c_[np.ones(len(x)), x]; beta = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y; SUMMARY['o2_normal_eq'] = beta.tolist(); print('Normal equation:', beta)
res = te.TotalCharges.values - slr.predict(te)
SUMMARY['o2_resid'] = dict(mean=float(res.mean()), sd=float(res.std()), corr_abs_resid_tenure=float(np.corrcoef(np.abs(res), te.tenure)[0, 1]))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
axes[0].scatter(tr.tenure, tr.TotalCharges, s=7, alpha=.35, color=C_STAY); xs = np.linspace(0, 72, 100); axes[0].plot(xs, b0 + b1*xs, color=C_CHURN, lw=2.5, label=f'TotalCharges = {b0:.1f} + {b1:.1f} x tenure')
axes[0].set_xlabel('Tenure (months)'); axes[0].set_ylabel('TotalCharges (USD)'); axes[0].set_title('Simple linear regression (training set)'); axes[0].legend()
axes[1].scatter(slr.predict(te), res, s=8, alpha=.4, color=C_STAY); axes[1].axhline(0, color=C_CHURN, lw=2); axes[1].set_xlabel('Predicted TotalCharges (USD)'); axes[1].set_ylabel('Residual (actual - predicted)'); axes[1].set_title('Residuals (test set)')
save_fig('fig05_simple_regression')
""")

# ---------------------------------------------------------------- O3
md("""## O3. Multiple and Polynomial Regression
**Question.** Does adding more variables (multiple regression) or curvature and interaction between tenure and monthly charge (polynomial regression) reduce the error? **Method.** The polynomial degree (1 to 5) is chosen on the validation set.""")
code(r"""
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
mlr = L.make_pipeline(LinearRegression(), L.REG_NUMERIC, scale=True)
mlr_fit = mlr.fit(fit, fit.TotalCharges); SUMMARY['o3_mlr_val_rmse'] = L.reg_metrics(val.TotalCharges, mlr_fit.predict(val))['RMSE']
def poly_pipe(d): return Pipeline([('sel', ColumnTransformer([('n', StandardScaler(), ['tenure', 'MonthlyCharges'])])), ('poly', PolynomialFeatures(d, include_bias=False)), ('model', LinearRegression())])
rows = []
for d in range(1, 6):
    p = poly_pipe(d).fit(fit, fit.TotalCharges)
    rows.append([d, L.reg_metrics(fit.TotalCharges, p.predict(fit))['RMSE'], L.reg_metrics(val.TotalCharges, p.predict(val))['RMSE'], L.reg_metrics(val.TotalCharges, p.predict(val))['R2']])
deg = pd.DataFrame(rows, columns=['Degree', 'Fit RMSE', 'Validation RMSE', 'Validation R2']).round(3); deg.to_csv(f'{RES}/o3_polynomial_degree.csv', index=False); display(deg)
BEST_DEG = int(deg.loc[deg['Validation RMSE'].idxmin(), 'Degree']); SUMMARY['o3_degree_table'] = deg.to_dict('records'); SUMMARY['o3_best_degree'] = BEST_DEG
print('Selected degree:', BEST_DEG)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(deg.Degree, deg['Fit RMSE'], 'o-', color=C_STAY, lw=2, label='Fit set'); ax.plot(deg.Degree, deg['Validation RMSE'], 'o-', color=C_CHURN, lw=2, label='Validation set')
ax.axvline(BEST_DEG, color='k', ls='--', lw=1); ax.set_xlabel('Polynomial degree'); ax.set_ylabel('RMSE (USD)'); ax.set_xticks(range(1, 6)); ax.set_title('Choosing the polynomial degree'); ax.legend()
save_fig('fig06_polynomial_degree')
mlr_final = L.make_pipeline(LinearRegression(), L.REG_NUMERIC, scale=False).fit(tr, tr.TotalCharges)
names = mlr_final.named_steps['prep'].get_feature_names_out(); coef = pd.DataFrame({'Feature': names, 'Coefficient': mlr_final.named_steps['model'].coef_}).sort_values('Coefficient', key=abs, ascending=False)
coef.round(3).to_csv(f'{RES}/o3_mlr_coefficients.csv', index=False); display(coef.head(10).round(3)); SUMMARY['o3_mlr_coef_top'] = coef.head(8).round(3).to_dict('records'); SUMMARY['o3_mlr_intercept'] = float(mlr_final.named_steps['model'].intercept_)
""")

# ---------------------------------------------------------------- O4
md("""## O4. Comparing four regression models and selecting the best
**Models.** (1) Simple Linear Regression, (2) Multiple Linear Regression, (3) Polynomial Regression (degree from O3), (4) Random Forest Regression. The Random Forest settings are chosen on the validation set from a small grid. **Selection rule (fixed in advance):** lowest validation RMSE; if two models are within 1 % of each other, the simpler one is preferred. The chosen model is then refitted on the full training set and scored once on the test set.""")
code(r"""
from sklearn.ensemble import RandomForestRegressor
import itertools
grid = list(itertools.product([5, 8, 12, None], [1, 5, 15]))
rf_rows = []
for md_, ml_ in grid:
    m = L.make_pipeline(RandomForestRegressor(n_estimators=200, max_depth=md_, min_samples_leaf=ml_, random_state=RS, n_jobs=1), L.REG_NUMERIC, scale=False).fit(fit, fit.TotalCharges)
    rf_rows.append([md_ if md_ else 'None', ml_, L.reg_metrics(val.TotalCharges, m.predict(val))['RMSE']])
rfg = pd.DataFrame(rf_rows, columns=['max_depth', 'min_samples_leaf', 'Validation RMSE']).round(3); rfg.to_csv(f'{RES}/o4_rf_regressor_grid.csv', index=False); display(rfg)
bi = rfg['Validation RMSE'].idxmin(); RF_D = None if rfg.loc[bi, 'max_depth'] == 'None' else int(rfg.loc[bi, 'max_depth']); RF_L = int(rfg.loc[bi, 'min_samples_leaf'])
SUMMARY['o4_rf_params'] = dict(max_depth=RF_D, min_samples_leaf=RF_L, n_estimators=200)
builders = {
 '1. Simple Linear': lambda: L.make_pipeline(LinearRegression(), ['tenure'], categorical=[], scale=False),
 '2. Multiple Linear': lambda: L.make_pipeline(LinearRegression(), L.REG_NUMERIC, scale=False),
 '3. Polynomial': lambda: poly_pipe(BEST_DEG),
 '4. Random Forest': lambda: L.make_pipeline(RandomForestRegressor(n_estimators=200, max_depth=RF_D, min_samples_leaf=RF_L, random_state=RS, n_jobs=1), L.REG_NUMERIC, scale=False)}
val_rmse, models_reg, rows = {}, {}, []
for name, mk in builders.items():
    val_rmse[name] = L.reg_metrics(val.TotalCharges, mk().fit(fit, fit.TotalCharges).predict(val))['RMSE']
    m = mk().fit(tr, tr.TotalCharges); models_reg[name] = m
    a, b = L.reg_metrics(tr.TotalCharges, m.predict(tr)), L.reg_metrics(te.TotalCharges, m.predict(te))
    rows.append(dict(Model=name, Val_RMSE=val_rmse[name], Train_MAE=a['MAE'], Train_RMSE=a['RMSE'], Train_R2=a['R2'], Test_MAE=b['MAE'], Test_MSE=b['MSE'], Test_RMSE=b['RMSE'], Test_R2=b['R2']))
regc = pd.DataFrame(rows).set_index('Model'); regc.round(4).to_csv(f'{RES}/o4_regression_comparison.csv'); display(regc.round(3))
order = regc.Val_RMSE.sort_values(); BEST_REG = order.index[0]
for nm in order.index[1:]:
    if val_rmse[nm] <= val_rmse[BEST_REG] * 1.01 and nm < BEST_REG: BEST_REG = nm       # prefer the simpler (lower-numbered) model within 1 %
SUMMARY['o4_best_regression'] = BEST_REG; SUMMARY['o4_regression_table'] = regc.round(4).reset_index().to_dict('records'); print('Selected regression model:', BEST_REG)
""")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(9, 8.4), sharex=True, sharey=True)
for ax, (name, m) in zip(axes.ravel(), models_reg.items()):
    p = m.predict(te); ax.scatter(te.TotalCharges, p, s=7, alpha=.4, color=C_CHURN if name == BEST_REG else C_STAY)
    ax.plot([0, 9000], [0, 9000], 'k--', lw=1); r2 = L.reg_metrics(te.TotalCharges, p)['R2']; ax.set_title(f'{name} (R2 = {r2:.3f})', fontsize=12)
for ax in axes[1]: ax.set_xlabel('Actual TotalCharges (USD)')
for ax in axes[:, 0]: ax.set_ylabel('Predicted (USD)')
fig.suptitle('Actual vs predicted on the test set (dashed = perfect)', fontweight='bold'); save_fig('fig07_actual_vs_predicted')
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3)); lab = [n.split('. ')[1] for n in regc.index]; xx = np.arange(4)
axes[0].bar(xx-.2, regc.Train_RMSE, .4, color=C_STAY, label='Train'); axes[0].bar(xx+.2, regc.Test_RMSE, .4, color=C_CHURN, label='Test'); axes[0].set_xticks(xx); axes[0].set_xticklabels(lab, fontsize=9.5); axes[0].set_ylabel('RMSE (USD)'); axes[0].set_title('RMSE: train vs test'); axes[0].legend()
axes[1].bar(xx-.2, regc.Train_R2, .4, color=C_STAY, label='Train'); axes[1].bar(xx+.2, regc.Test_R2, .4, color=C_CHURN, label='Test'); axes[1].set_xticks(xx); axes[1].set_xticklabels(lab, fontsize=9.5); axes[1].set_ylim(.7, 1.01); axes[1].set_ylabel('R-squared'); axes[1].set_title('R-squared: train vs test'); axes[1].legend(loc='lower right')
save_fig('fig08_regression_metrics')
best_m = models_reg[BEST_REG]; r_ = te.TotalCharges.values - best_m.predict(te)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4)); axes[0].hist(r_, bins=40, color=C_STAY); axes[0].axvline(0, color=C_CHURN, lw=2); axes[0].set_xlabel('Residual (USD)'); axes[0].set_ylabel('Customers'); axes[0].set_title(f'Residual distribution: {BEST_REG}')
axes[1].scatter(best_m.predict(te), r_, s=8, alpha=.4, color=C_STAY); axes[1].axhline(0, color=C_CHURN, lw=2); axes[1].set_xlabel('Predicted TotalCharges (USD)'); axes[1].set_ylabel('Residual (USD)'); axes[1].set_title('Residuals vs predicted')
save_fig('fig09_best_regression_residuals')
SUMMARY['o4_best_resid'] = dict(mean=float(r_.mean()), sd=float(r_.std()), within_100=float((np.abs(r_) <= 100).mean()), within_250=float((np.abs(r_) <= 250).mean()))
""")

# ---------------------------------------------------------------- O5
md("""## O5. Comparing six classifiers for churn (Unit III)
**Models.** Logistic Regression, k-Nearest Neighbours (lazy learning), Gaussian Naive Bayes, Decision Tree, Random Forest and Support Vector Machine, plus a majority-class baseline. Settings are chosen on the validation set by **F1 score** (churn is the minority class, and F1 balances precision and recall). Class weights are *balanced* for LR, DT, RF and SVM. Final scores are on the untouched test set at the default 0.5 threshold.""")
code(r"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
NUM = L.NUMERIC_ALL
specs = {
 'Logistic Regression': (lambda C: LogisticRegression(C=C, max_iter=3000, class_weight='balanced'), 'C', [0.01, 0.1, 1, 10], True),
 'k-Nearest Neighbours': (lambda k: KNeighborsClassifier(n_neighbors=k), 'k', [5, 11, 21, 31, 51, 81], True),
 'Naive Bayes': (lambda v: GaussianNB(var_smoothing=v), 'var_smoothing', [1e-9, 1e-6, 1e-3, 1e-1], True),
 'Decision Tree': (lambda d: DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, class_weight='balanced', random_state=RS), 'max_depth', [3, 4, 5, 6, 8, 10], False),
 'Random Forest': (lambda d: RandomForestClassifier(n_estimators=200, max_depth=d, min_samples_leaf=5, class_weight='balanced_subsample', random_state=RS, n_jobs=1), 'max_depth', [4, 6, 8, 12], False),
 'Support Vector Machine': (lambda C: SVC(C=C, kernel='rbf', class_weight='balanced', probability=True, random_state=RS), 'C', [0.5, 1, 2], True)}
tune_rows, chosen, clf = [], {}, {}
for name, (mk, pname, vals, scale) in specs.items():
    best = (-1, None)
    for v in vals:
        m = L.make_pipeline(mk(v), NUM, scale=scale).fit(fit, fit.ChurnFlag); f1v = L.clf_metrics(val.ChurnFlag, m.predict(val), m.predict_proba(val)[:, 1])['F1']
        tune_rows.append([name, pname, v, f1v])
        if f1v > best[0]: best = (f1v, v)
    chosen[name] = dict(param=pname, value=best[1], val_F1=best[0])
    clf[name] = L.make_pipeline(mk(best[1]), NUM, scale=scale).fit(tr, tr.ChurnFlag)
    print(f'{name:24s} best {pname} = {best[1]}  (validation F1 {best[0]:.4f})')
tune = pd.DataFrame(tune_rows, columns=['Model', 'Setting', 'Value', 'Validation F1']).round(4); tune.to_csv(f'{RES}/o5_tuning_log.csv', index=False)
SUMMARY['o5_chosen'] = {k: dict(param=v['param'], value=(float(v['value']) if not isinstance(v['value'], int) else v['value']), val_F1=round(v['val_F1'], 4)) for k, v in chosen.items()}
base = L.make_pipeline(DummyClassifier(strategy='prior'), NUM).fit(tr, tr.ChurnFlag); clf = {'Baseline (majority class)': base, **clf}
rows, cms = [], {}
from sklearn.metrics import confusion_matrix
for name, m in clf.items():
    p = m.predict_proba(te)[:, 1]; pred = m.predict(te); r = L.clf_metrics(te.ChurnFlag, pred, np.clip(p, 1e-6, 1-1e-6)); r['Model'] = name; rows.append(r); cms[name] = confusion_matrix(te.ChurnFlag, pred).tolist()
clfc = pd.DataFrame(rows).set_index('Model')
clfc.round(4).to_csv(f'{RES}/o5_classification_comparison.csv'); display(clfc.round(3)); SUMMARY['o5_table'] = clfc.round(4).reset_index().to_dict('records'); SUMMARY['o5_cms'] = cms
""")
code(r"""
from sklearn.metrics import roc_curve
pal = {'Baseline (majority class)': '#9A9A9A', 'Logistic Regression': C_STAY, 'k-Nearest Neighbours': '#8E7DBE', 'Naive Bayes': C_ACC, 'Decision Tree': '#7A7A7A', 'Random Forest': C_GRN, 'Support Vector Machine': C_CHURN}
fig, ax = plt.subplots(figsize=(6.6, 5.2))
for name, m in clf.items():
    p = m.predict_proba(te)[:, 1]; fpr, tpr, _ = roc_curve(te.ChurnFlag, p); ax.plot(fpr, tpr, color=pal[name], lw=1.8, label=f'{name} ({clfc.loc[name, "AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k:', lw=1); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title('ROC curves (test set, AUC in brackets)'); ax.legend(fontsize=9, loc='lower right')
save_fig('fig10_roc_curves')
mets = ['Accuracy', 'Precision', 'Recall', 'F1']; d = clfc.drop(index='Baseline (majority class)')
fig, ax = plt.subplots(figsize=(10.5, 4.4)); w = .2; xx = np.arange(len(d))
for i, mt in enumerate(mets): ax.bar(xx + (i-1.5)*w, d[mt], w, label=mt, color=[C_STAY, C_ACC, C_CHURN, C_GRN][i])
ax.set_xticks(xx); ax.set_xticklabels([n.replace(' ', '\n', 1) for n in d.index], fontsize=9.5); ax.set_ylim(0, 1); ax.set_title('Classification metrics on the test set'); ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.0))
save_fig('fig11_classification_metrics')
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
for ax, name in zip(axes.ravel(), d.index):
    sns.heatmap(np.array(cms[name]), annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'], ax=ax, annot_kws={'size': 13}); ax.set_title(name, fontsize=11.5); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
save_fig('fig12_confusion_matrices')
kn = tune[tune.Model == 'k-Nearest Neighbours']; fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(kn.Value, kn['Validation F1'], 'o-', color=C_CHURN, lw=2); ax.set_xlabel('Number of neighbours k'); ax.set_ylabel('Validation F1'); ax.set_title('Choosing k for kNN'); save_fig('fig13_knn_k')
""")

# ---------------------------------------------------------------- O6
md("""## O6. Selecting the best classifier; rules and errors
**Selection rule (fixed in advance):** highest validation F1. We then read the rules of a shallow decision tree, the importance of inputs in the Random Forest, and profile the errors of the selected model.""")
code(r"""
cand = {k: v['val_F1'] for k, v in chosen.items()}; BEST_CLF = max(cand, key=cand.get); SUMMARY['o6_best_classifier'] = BEST_CLF
SUMMARY['o6_val_f1'] = {k: round(v, 4) for k, v in cand.items()}; print('Selected classifier:', BEST_CLF, cand)
from sklearn.tree import export_text, plot_tree
shallow = L.make_pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight='balanced', random_state=RS), NUM, scale=False).fit(tr, tr.ChurnFlag)
fn = shallow.named_steps['prep'].get_feature_names_out(); rules = export_text(shallow.named_steps['model'], feature_names=list(fn)); open(f'{RES}/o6_tree_rules.txt', 'w').write(rules); print(rules)
SUMMARY['o6_tree_rules'] = rules
SUMMARY['o6_tree_test'] = L.clf_metrics(te.ChurnFlag, shallow.predict(te), shallow.predict_proba(te)[:, 1])
fig, ax = plt.subplots(figsize=(11, 5.6)); plot_tree(shallow.named_steps['model'], feature_names=list(fn), class_names=['Stay', 'Churn'], filled=True, rounded=True, fontsize=8, ax=ax, impurity=False, proportion=True)
ax.set_title('Decision tree (depth 3) - readable churn rules'); save_fig('fig14_decision_tree')
rf = clf['Random Forest']; imp = pd.DataFrame({'Feature': rf.named_steps['prep'].get_feature_names_out(), 'Importance': rf.named_steps['model'].feature_importances_}).sort_values('Importance', ascending=False)
imp.round(4).to_csv(f'{RES}/o6_rf_importance.csv', index=False); display(imp.head(10).round(4)); SUMMARY['o6_rf_top'] = imp.head(10).round(4).to_dict('records')
top = imp.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(7.4, 5)); ax.barh(top.Feature, top.Importance, color=C_GRN); ax.set_xlabel('Importance (mean decrease in impurity)'); ax.set_title('Random Forest: most important inputs')
save_fig('fig15_rf_importance')
""")
code(r"""
bm = clf[BEST_CLF]; pb = bm.predict(te); ea = te.copy(); ea['pred'] = pb; ea['tenure_band'] = pd.cut(ea.tenure, [-1, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
ea['outcome'] = np.select([(ea.ChurnFlag == 1) & (ea.pred == 1), (ea.ChurnFlag == 0) & (ea.pred == 1), (ea.ChurnFlag == 1) & (ea.pred == 0)], ['TP', 'FP', 'FN'], 'TN')
prof = ea.groupby('outcome').agg(n=('ChurnFlag', 'size'), mean_tenure=('tenure', 'mean'), mean_monthly=('MonthlyCharges', 'mean'), share_m2m=('Contract', lambda s: (s == 'Month-to-month').mean()), share_fiber=('InternetService', lambda s: (s == 'Fiber optic').mean())).round(3).reindex(['TP', 'FP', 'FN', 'TN'])
prof.to_csv(f'{RES}/o6_error_profile.csv'); display(prof); SUMMARY['o6_profile'] = prof.reset_index().to_dict('records')
miss = {}
for col in ['Contract', 'tenure_band', 'InternetService']:
    g = ea[ea.ChurnFlag == 1].groupby(col, observed=True).apply(lambda d: pd.Series({'churners': len(d), 'missed_rate': (d.pred == 0).mean()}), include_groups=False).round(3); miss[col] = g
SUMMARY['o6_miss'] = {k: v.reset_index().astype({v.index.name: str}).to_dict('records') for k, v in miss.items()}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
for ax, (col, g) in zip(axes, miss.items()):
    ax.bar(g.index.astype(str), g.missed_rate, color=C_CHURN, width=.6)
    for i, (v, n) in enumerate(zip(g.missed_rate, g.churners)): ax.text(i, v+.01, f'{v:.0%}\n(n={int(n)})', ha='center', fontsize=10)
    ax.set_title(f'Missed churners by {col}'); ax.set_ylim(0, 1)
axes[0].set_ylabel('False-negative rate among churners'); save_fig('fig16_error_analysis')
""")

# ---------------------------------------------------------------- O7
md("""## O7. Using both selected models together
The selected **regression model** predicts the lifetime revenue (TotalCharges) of each customer; the selected **classifier** gives the probability of churn. Together they split customers into four priority groups.""")
code(r"""
reg_best = models_reg[BEST_REG]; clf_best = clf[BEST_CLF]
pd.DataFrame({'customerID': te.customerID.values, 'actual': te.TotalCharges.values, **{nm: m.predict(te) for nm, m in models_reg.items()}}).round(2).to_csv(f'{RES}/test_regression_predictions.csv', index=False)
out = te[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'ChurnFlag']].copy()
out['pred_total_charges'] = reg_best.predict(te).round(2); out['p_churn'] = clf_best.predict_proba(te)[:, 1].round(4)
rev_cut = float(out.pred_total_charges.median()); risk_cut = 0.5
out['group'] = np.select([(out.p_churn >= risk_cut) & (out.pred_total_charges >= rev_cut), (out.p_churn >= risk_cut), (out.pred_total_charges >= rev_cut)],
                         ['High risk, high revenue', 'High risk, lower revenue', 'Low risk, high revenue'], 'Low risk, lower revenue')
grp = out.groupby('group').agg(customers=('ChurnFlag', 'size'), actual_churn_rate=('ChurnFlag', 'mean'), mean_pred_revenue=('pred_total_charges', 'mean'), mean_monthly=('MonthlyCharges', 'mean')).round(3)
grp = grp.reindex(['High risk, high revenue', 'High risk, lower revenue', 'Low risk, high revenue', 'Low risk, lower revenue']); grp.to_csv(f'{RES}/o7_priority_groups.csv'); display(grp)
SUMMARY['o7_groups'] = grp.reset_index().to_dict('records'); SUMMARY['o7_cuts'] = dict(revenue_median=rev_cut, risk=risk_cut)
out.round(4).to_csv(f'{RES}/test_predictions.csv', index=False)
colors = {'High risk, high revenue': C_CHURN, 'High risk, lower revenue': C_ACC, 'Low risk, high revenue': C_STAY, 'Low risk, lower revenue': '#B0B0B0'}
fig, ax = plt.subplots(figsize=(8, 5.4))
for g_, c_ in colors.items():
    s = out[out.group == g_]; ax.scatter(s.pred_total_charges, s.p_churn, s=12, alpha=.55, color=c_, label=f'{g_} ({len(s)})')
ax.axvline(rev_cut, color='k', ls='--', lw=1); ax.axhline(risk_cut, color='k', ls='--', lw=1); ax.set_xlabel('Predicted TotalCharges (USD)'); ax.set_ylabel('Predicted churn probability'); ax.set_title('Customer priority map (test set)'); ax.legend(fontsize=9, loc='upper right')
save_fig('fig17_priority_map')
""")

md("## 4. Save the final models and summary")
code(r"""
import joblib, sklearn, datetime
joblib.dump(reg_best, f'{MOD}/final_regression_model.joblib'); joblib.dump(clf_best, f'{MOD}/final_churn_classifier.joblib')
json.dump(dict(regression_model=BEST_REG, classifier=BEST_CLF, features=FEATS, regression_target='TotalCharges', churn_threshold=0.5, revenue_median=rev_cut,
               sklearn_version=sklearn.__version__, random_state=RS, created=str(datetime.date.today())), open(f'{MOD}/model_metadata.json', 'w'), indent=2)
def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
json.dump(SUMMARY, open(f'{RES}/summary.json', 'w'), indent=1, default=_j)
print('Saved models:', sorted(os.listdir(MOD)), '| figures:', len([f for f in os.listdir(FIG) if f.endswith('.png')]))
""")

nb['cells'] = cells; nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbf.write(nb, 'churn_prediction.ipynb'); print('notebook written with', len(cells), 'cells')

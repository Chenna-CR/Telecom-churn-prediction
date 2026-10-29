# -*- coding: utf-8 -*-
from content_a import *

df = pd.read_csv(f'{RES}/cleaned_dataset.csv'); dd_ = df[df.tenure > 0]
ratio = (dd_.TotalCharges / (dd_.tenure * dd_.MonthlyCharges))
cr = lambda col, val: df.loc[df[col] == val, 'ChurnFlag'].mean()
C = S['corr']

h1('4. ANALYSIS ON DATASET')
p('This chapter reports seven analyses. Each follows the prescribed structure: (i) Introduction, (ii) General Description, (iii) Specific Requirements, Functions and Formulas, (iv) Analysis Results, and (v) Visualization with interpretation. Every number in the tables and text is produced by the notebook and stored in the results folder.')

# ================================ O1 ==========================================================
h2('4.1 Objective O1: Understanding the data')
h3('i. Introduction')
p('Before any model is built, an analyst needs to know how the variables relate to each other and which customer groups churn most. Correlation shows which numeric variables move together; comparing churn rates across groups shows where the problem is concentrated and which attributes are unlikely to help a model.')
h3('ii. General Description')
p('Three things are examined: the Pearson correlation between the numeric variables and the two targets; the average of each numeric variable for churners and stayers; and the churn rate in every group of every categorical attribute. For each attribute the *spread* (highest group rate minus lowest group rate) summarises how much the attribute separates churners from stayers.')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O1', ['Purpose', 'Function'], [('Correlation matrix', 'pandas.DataFrame.corr(method="pearson")'), ('Group statistics', 'pandas.DataFrame.groupby().agg()'), ('Plots', 'matplotlib, seaborn (heatmap, histplot, bar)')], widths=[.4, .6], align=['l', 'l'])
p('The Pearson correlation coefficient between two variables x and y with n observations is:')
eq('r = Σ (xᵢ − x̄)(yᵢ − ȳ) / √( Σ (xᵢ − x̄)² · Σ (yᵢ − ȳ)² )        (1)')
p('r lies between −1 (perfect negative linear relationship) and +1 (perfect positive); 0 means no linear relationship. With the binary churn indicator (0/1), r measures how strongly a numeric variable differs between churners and stayers. The spread of an attribute is:')
eq('spread = max over groups of churn rate − min over groups of churn rate        (2)')
h3('iv. Analysis Results')
means = S['o1_means']
table('Mean of the numeric variables by churn status', ['Variable', 'Stayed', 'Churned'], [(k, f'{means["No"][k]:,.2f}', f'{means["Yes"][k]:,.2f}') for k in ['tenure', 'MonthlyCharges', 'TotalCharges']], widths=[.4, .3, .3], align=['l', 'r', 'r'])
p(f'Churners have been customers for {means["Yes"]["tenure"]:.1f} months on average against {means["No"]["tenure"]:.1f} for stayers, yet they pay more each month (USD {means["Yes"]["MonthlyCharges"]:.2f} against {means["No"]["MonthlyCharges"]:.2f}). Because they leave early, their accumulated TotalCharges is lower (USD {means["Yes"]["TotalCharges"]:,.0f} against {means["No"]["TotalCharges"]:,.0f}).')
corr = pd.DataFrame(C).loc[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'ChurnFlag'], ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'ChurnFlag']]
table('Pearson correlation matrix', [''] + list(corr.columns), [[i] + [f'{v:.3f}' for v in r] for i, r in corr.iterrows()], widths=[.2] + [.16] * 5, align=['l'] + ['r'] * 5)
p(f'TotalCharges is strongly correlated with tenure (r = {C["TotalCharges"]["tenure"]:.2f}) and fairly strongly with MonthlyCharges (r = {C["TotalCharges"]["MonthlyCharges"]:.2f}), which makes both natural inputs for the regression in Part A. '
  f'Tenure has the strongest correlation with churn (r = {C["ChurnFlag"]["tenure"]:.2f}): longer-standing customers churn less. MonthlyCharges has a weak positive correlation with churn ({C["ChurnFlag"]["MonthlyCharges"]:.2f}).')
sp_ = rd('o1_churn_rate_spread')
table('Churn-rate spread of each categorical attribute (largest first)', ['Attribute', 'Highest-churn group', 'Rate', 'Lowest-churn group', 'Rate', 'Spread'],
      [(r.Variable, r['Highest-churn group'], pct(r.Rate), r['Lowest-churn group'], pct(r['Rate (low)']), pct(r.Spread)) for _, r in sp_.iterrows()], widths=[.19, .2, .1, .22, .1, .19], align=['l', 'l', 'r', 'l', 'r', 'r'])
top = S['o1_spread_top']; bot = S['o1_spread_bottom']
p(f'The attributes that separate churners from stayers most are **Contract** (spread {pct(top[0]["Spread"])}), **InternetService** ({pct(top[1]["Spread"])}), **OnlineSecurity** ({pct(top[2]["Spread"])}) and the other protection and support services and payment method. '
  f'At the bottom, **gender** ({pct(bot[2]["Spread"])}), **PhoneService** ({pct(bot[1]["Spread"])}) and **MultipleLines** ({pct(bot[0]["Spread"])}) barely differ between groups, so they carry almost no information about churn on their own.')
h3('v. Visualization and Interpretation')
fig('fig02_churn_rate_by_segment', 'Churn rate by key customer segments (dashed line marks the overall rate of 26.5%)', 5.9)
p(f'Figure 2 shows the largest gaps. Month-to-month customers churn at {pct(cr("Contract","Month-to-month"))}, compared with {pct(cr("Contract","One year"))} on one-year and {pct(cr("Contract","Two year"))} on two-year contracts. '
  f'Fibre-optic customers churn at {pct(cr("InternetService","Fiber optic"))} against {pct(cr("InternetService","DSL"))} for DSL. Electronic-check payers churn at {pct(cr("PaymentMethod","Electronic check"))}, about three times the rate of customers who pay by bank transfer or credit card. '
  f'Customers without online security churn at {pct(cr("OnlineSecurity","No"))} against {pct(cr("OnlineSecurity","Yes"))} with it. These are associations; the data cannot say whether the services keep customers or whether already committed customers are the ones who buy them.')
fig('fig03_correlation_heatmap', 'Pearson correlation matrix of the numeric variables and the churn indicator', 4.3)
p('The heat map shows the strong blocks of positive correlation among tenure, MonthlyCharges and TotalCharges and the negative correlation of tenure and TotalCharges with churn.')
fig('fig04_numeric_distributions', 'Distributions of tenure, monthly charges and total charges for churners and stayers', 6.3)
p('Churners are concentrated at short tenures and at higher monthly charges. Stayers include a large block of low-priced, phone-only customers who almost never leave, and a peak at 72 months, the end of the observation window.')
p('**Conclusion of O1.** Tenure, contract type, internet service and protection services are the main correlates of churn; gender and phone service are not. TotalCharges is strongly tied to tenure and MonthlyCharges.')

# ================================ O2 ==========================================================
h2('4.2 Objective O2: Simple Linear Regression')
h3('i. Introduction')
p('The simplest way to predict a number is a straight line through one input. For lifetime revenue, the natural input is tenure: the longer a customer stays, the more they have been billed. This objective measures how far a straight line in tenure alone can go.')
h3('ii. General Description')
p('TotalCharges is the target (y) and tenure in months is the single input (x). The line is fitted on the training set by Ordinary Least Squares (OLS), which chooses the intercept and slope that minimise the sum of squared residuals. The slope and intercept are calculated by hand from the closed-form formulas and checked against scikit-learn. The fit is judged on the training and test sets with MAE, MSE, RMSE and R-squared, and the residuals are inspected for patterns.')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O2', ['Purpose', 'Function'], [('OLS fit', 'sklearn.linear_model.LinearRegression'), ('Closed-form check', 'numpy (sums, matrix inverse)'), ('Metrics', 'churn_lib.reg_metrics: mean_absolute_error, mean_squared_error, r2_score')], widths=[.4, .6], align=['l', 'l'])
p('The model is ŷ = β₀ + β₁x. OLS minimises the sum of squared errors, giving:')
eq('β₁ = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²,     β₀ = ȳ − β₁x̄        (3)')
p('For several inputs the same idea is written in matrix form (the normal equation), which is used again in O3:')
eq('β = (XᵀX)⁻¹ Xᵀy        (4)')
p('The evaluation metrics, with n test observations, actual values yᵢ, predictions ŷᵢ and mean ȳ, are:')
eq('MAE = (1/n) Σ |yᵢ − ŷᵢ|,     MSE = (1/n) Σ (yᵢ − ŷᵢ)²,     RMSE = √MSE        (5)')
eq('R² = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²        (6)')
p('MAE is the average size of the error in dollars; MSE squares the errors and so punishes large ones; RMSE brings MSE back to dollars; R² is the share of the variation in y that the model explains (1 is perfect, 0 is no better than predicting the mean).')
h3('iv. Analysis Results')
o = S['o2_ols']; m = S['o2_metrics']
table('OLS estimates for TotalCharges on tenure', ['Quantity', 'By formula', 'By scikit-learn'], [('Slope (USD per month of tenure)', f2(o['slope']), f2(o['slope_sklearn'])), ('Intercept (USD)', f2(o['intercept']), f2(o['intercept_sklearn'])), ('Pearson r (tenure, TotalCharges), training set', f3(o['r_pearson']), '')], widths=[.5, .25, .25], align=['l', 'r', 'r'])
p(f'The hand calculation and scikit-learn agree to many decimals, and the matrix formula gives the same pair ({S["o2_normal_eq"][0]:.2f}, {S["o2_normal_eq"][1]:.2f}). The fitted line is TotalCharges = {o["intercept"]:.2f} + {o["slope"]:.2f} x tenure: each extra month of tenure is associated with about USD {o["slope"]:.0f} more billed.')
table('Simple Linear Regression: error measures', ['Data set', 'MAE (USD)', 'MSE', 'RMSE (USD)', 'R-squared'], [(k, f'{v["MAE"]:,.1f}', f'{v["MSE"]:,.0f}', f'{v["RMSE"]:,.1f}', f3(v['R2'])) for k, v in m.items()], widths=[.2, .2, .2, .2, .2], align=['l', 'r', 'r', 'r', 'r'])
r_ = S['o2_resid']
p(f'On the test set the model explains {pct(m["Test"]["R2"],0)} of the variation (R² = {m["Test"]["R2"]:.3f}), with a typical error (MAE) of about USD {m["Test"]["MAE"]:,.0f}. Training and test results are close, so the model does not overfit, but an RMSE of USD {m["Test"]["RMSE"]:,.0f} is large compared with a mean TotalCharges of about USD {S["desc"]["TotalCharges"]["mean"]:,.0f}. '
  f'The residuals have a mean of {r_["mean"]:.0f}, but their spread grows with tenure (correlation of absolute residual with tenure {r_["corr_abs_resid_tenure"]:.2f}).')
h3('v. Visualization and Interpretation')
fig('fig05_simple_regression', 'Simple linear regression: fitted line on the training set (left) and residuals on the test set (right)', 6.3)
p('The scatter plot shows why one line is not enough. At each tenure the points fan out into a band: customers with a low monthly bill sit near the bottom of the fan and customers with a high bill near the top, and the single line runs through the middle. '
  'The residual plot shows the same thing as a funnel, with errors of several thousand dollars for long-tenure customers. A linear model assumes errors of similar size everywhere; here that assumption fails. The missing ingredient is the monthly charge, which is added in O3.')
p('**Conclusion of O2.** Tenure alone explains about two-thirds of the variation in lifetime revenue. The remaining error follows a clear pattern, which suggests adding MonthlyCharges.')

# ================================ O3 ==========================================================
h2('4.3 Objective O3: Multiple and Polynomial Regression')
h3('i. Introduction')
p('O2 showed that tenure alone leaves a fan-shaped error. Two natural improvements are to use more inputs (Multiple Linear Regression) and to let the model bend and combine inputs (Polynomial Regression). This objective tests how much each helps.')
h3('ii. General Description')
p('Multiple Linear Regression uses SeniorCitizen, tenure, MonthlyCharges and the 15 one-hot encoded categorical attributes, 18 inputs in the sense of columns before encoding. Polynomial Regression takes tenure and MonthlyCharges, standardises them, creates powers and products up to a chosen degree, and fits a linear model on those new columns. '
  'The degree (1 to 5) is chosen on the validation set as the one with the lowest RMSE; a higher degree always fits the fit set at least as well, so the validation set is what reveals overfitting.')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O3', ['Purpose', 'Function'], [('Multiple regression', 'LinearRegression inside a Pipeline with one-hot encoding'), ('Polynomial terms', 'sklearn.preprocessing.PolynomialFeatures'), ('Scaling', 'sklearn.preprocessing.StandardScaler'), ('Validation scores', 'churn_lib.reg_metrics')], widths=[.4, .6], align=['l', 'l'])
eq('ŷ = β₀ + β₁x₁ + β₂x₂ + … + β_p x_p        (7)')
eq('Degree 2 with inputs t (tenure) and m (monthly charge):  ŷ = β₀ + β₁t + β₂m + β₃t² + β₄tm + β₅m²        (8)')
p('In Equation 8, the product term tm lets the effect of tenure depend on the monthly charge. This matches how billing works: total charges are roughly the monthly charge multiplied by the number of months.')
h3('iv. Analysis Results')
dg = pd.DataFrame(S['o3_degree_table'])
table('Polynomial degree selection (fit set and validation set)', ['Degree', 'Fit RMSE (USD)', 'Validation RMSE (USD)', 'Validation R-squared'], [(int(r.Degree), f'{r["Fit RMSE"]:,.2f}', f'{r["Validation RMSE"]:,.2f}', f3(r['Validation R2'])) for _, r in dg.iterrows()], widths=[.15, .27, .31, .27], align=['c', 'r', 'r', 'r'])
bd = S['o3_best_degree']
p(f'Moving from degree 1 (a plain plane in tenure and monthly charge) to degree 2 cuts the validation RMSE from USD {dg["Validation RMSE"].iloc[0]:,.0f} to USD {dg["Validation RMSE"].iloc[1]:,.0f} and lifts R² from {dg["Validation R2"].iloc[0]:.3f} to {dg["Validation R2"].iloc[1]:.3f}. Degrees 3 to 5 bring no further gain (the validation RMSE rises very slightly), so **degree {bd}** is selected: it is the simplest model that captures the pattern.')
cf = S['o3_mlr_coef_top']
table('Multiple Linear Regression: largest coefficients (USD per unit, other inputs held fixed)', ['Input', 'Coefficient'], [(c['Feature'], f'{c["Coefficient"]:,.1f}') for c in cf[:6]] + [('Intercept', f'{S["o3_mlr_intercept"]:,.1f}')], widths=[.65, .35], align=['l', 'r'])
mlr_te = [r for r in S['o4_regression_table'] if r['Model'] == '2. Multiple Linear'][0]
p(f'Multiple Linear Regression, fitted with all inputs, has a test R² of {mlr_te["Test_R2"]:.3f} and a test RMSE of USD {mlr_te["Test_RMSE"]:,.0f}, compared with {S["o2_metrics"]["Test"]["R2"]:.3f} and USD {S["o2_metrics"]["Test"]["RMSE"]:,.0f} for tenure alone. Adding inputs helps a great deal, but the model is still linear and cannot represent the tenure-by-charge product. '
  'The coefficients of the categorical inputs in the table are mostly small adjustments that compensate for the missing interaction, not business effects, and should not be read as causes.')
p(f'**Why the polynomial model is so accurate.** For customers with tenure above zero, TotalCharges divided by (tenure x MonthlyCharges) has a median of {ratio.median():.2f}, and {pct((abs(ratio-1)<.05).mean(),0)} of customers lie within 5% of exactly 1. TotalCharges is therefore close to the product of tenure and monthly charge by the way bills accumulate. A degree-2 polynomial contains that product, so it fits almost perfectly. '
  'This is a legitimate regression result, but the very high R² reflects a near-arithmetical relationship in the data and should not be presented as a deep discovery.')
h3('v. Visualization and Interpretation')
fig('fig06_polynomial_degree', 'RMSE of polynomial regression for degrees 1 to 5 on the fit set and the validation set', 5.0)
p('The error collapses between degree 1 and degree 2 and is flat afterwards; the fit and validation curves lie almost on top of each other, so there is no sign of overfitting up to degree 5. The dashed line marks the selected degree.')
p('**Conclusion of O3.** More inputs reduce the error a lot, and allowing the tenure-by-charge product reduces it by another order of magnitude. Degree 2 is enough.')

# ================================ O4 ==========================================================
h2('4.4 Objective O4: Comparing four regression models')
h3('i. Introduction')
p('With three regression ideas in hand, the practical question is which model should be used for prediction. A fair comparison needs the same data for every model, a selection rule fixed in advance, and a look at both training and test error to detect overfitting.')
h3('ii. General Description')
p('Four models are compared: (1) Simple Linear Regression on tenure, (2) Multiple Linear Regression on all inputs, (3) Polynomial Regression of the degree chosen in O3, and (4) Random Forest Regression on all inputs. The Random Forest settings (maximum depth and minimum leaf size) are chosen from a 12-combination grid on the validation set. '
  '**Selection rule, fixed before the test set was used:** the model with the lowest validation RMSE wins, and if models are within 1% of each other the simpler one is preferred. The winner is refitted on the full training set and its test scores are reported with MAE, MSE, RMSE and R².')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O4', ['Purpose', 'Function'], [('Random Forest regression', 'sklearn.ensemble.RandomForestRegressor'), ('Settings grid', 'itertools.product over depth and leaf size'), ('Metrics', 'churn_lib.reg_metrics'), ('Plots', 'matplotlib scatter, bar and histogram')], widths=[.4, .6], align=['l', 'l'])
p('A Random Forest builds many decision trees, each on a bootstrap sample of the rows and a random subset of the inputs at every split, and averages their predictions (Breiman, 2001). For regression, each tree splits the data to reduce the squared error inside the resulting groups and predicts the group mean:')
eq('ŷ(x) = (1/T) Σ_{t=1..T} f_t(x)        (9)')
p('where T is the number of trees (200 here) and f_t is the prediction of tree t. Averaging many different trees reduces the variance of a single tree, which is the main reason forests are accurate; they need no scaling and can capture curves and interactions without being told which ones to use.')
h3('iv. Analysis Results')
rg = rd('o4_rf_regressor_grid'); rp = S['o4_rf_params']
p(f'The grid search on the validation set selected a maximum depth of {rp["max_depth"]} and a minimum leaf size of {rp["min_samples_leaf"]} for the forest (200 trees).')
reg = pd.DataFrame(S['o4_regression_table'])
table('Regression models: validation RMSE and test-set error measures', ['Model', 'Validation RMSE', 'Test MAE', 'Test MSE', 'Test RMSE', 'Test R-squared'], [(r.Model, f'{r.Val_RMSE:,.1f}', f'{r.Test_MAE:,.1f}', f'{r.Test_MSE:,.0f}', f'{r.Test_RMSE:,.1f}', f'{r.Test_R2:.4f}') for _, r in reg.iterrows()], widths=[.24, .15, .13, .15, .15, .18], align=['l', 'r', 'r', 'r', 'r', 'r'])
table('Overfitting check: training versus test', ['Model', 'Train RMSE', 'Test RMSE', 'Train R-squared', 'Test R-squared'], [(r.Model, f'{r.Train_RMSE:,.1f}', f'{r.Test_RMSE:,.1f}', f'{r.Train_R2:.4f}', f'{r.Test_R2:.4f}') for _, r in reg.iterrows()], widths=[.28, .18, .18, .18, .18], align=['l', 'r', 'r', 'r', 'r'])
rr = reg.set_index('Model'); br = S['o4_best_resid']
p(f'**{BEST_REG_NAME}** has the lowest validation RMSE (USD {rr.loc[BEST_REG, "Val_RMSE"]:.1f}) and is selected. On the test set it reaches MAE USD {rr.loc[BEST_REG, "Test_MAE"]:.1f}, RMSE USD {rr.loc[BEST_REG, "Test_RMSE"]:.1f} and R² {rr.loc[BEST_REG, "Test_R2"]:.4f}. '
  f'The Random Forest is a close second (test RMSE USD {rr.loc["4. Random Forest", "Test_RMSE"]:.1f}) but is slightly worse, and Multiple and Simple Linear Regression are far behind (test RMSE USD {rr.loc["2. Multiple Linear", "Test_RMSE"]:,.0f} and USD {rr.loc["1. Simple Linear", "Test_RMSE"]:,.0f}).')
p(f'The overfitting check favours the polynomial model too: its training and test RMSE are almost identical (USD {rr.loc["3. Polynomial", "Train_RMSE"]:.1f} against {rr.loc["3. Polynomial", "Test_RMSE"]:.1f}), whereas the Random Forest fits the training data much better (RMSE USD {rr.loc["4. Random Forest", "Train_RMSE"]:.1f}) than new data (USD {rr.loc["4. Random Forest", "Test_RMSE"]:.1f}), a typical sign of mild overfitting. '
  f'The polynomial model is also simpler and its equation can be written down. For the selected model the test residuals have a mean of {br["mean"]:.2f} and a standard deviation of USD {br["sd"]:.1f}; {pct(br["within_100"],0)} of predictions are within USD 100 of the actual value and {pct(br["within_250"],1)} are within USD 250.')
h3('v. Visualization and Interpretation')
fig('fig07_actual_vs_predicted', 'Actual versus predicted TotalCharges on the test set for the four regression models', 5.6)
p('Points on the dashed diagonal are perfect predictions. Simple Linear Regression shows the funnel and a flat ceiling near USD 5,300 (the largest value a straight line in tenure can give for 72 months). Multiple Linear Regression is closer but curved away from the diagonal and even predicts negative values for new customers. The polynomial and Random Forest models lie on the diagonal.')
fig('fig08_regression_metrics', 'RMSE and R-squared on the training and test sets for the four regression models', 6.3)
p('The bars make the ranking and the overfitting check visible at a glance: RMSE falls from about USD 1,300 to about USD 66 across the four models, and only the Random Forest shows a visible gap between its training and test bars.')
fig('fig09_best_regression_residuals', f'Residuals of the selected model ({BEST_REG_NAME}) on the test set', 6.3)
p('The residuals are centred on zero and roughly symmetric with no funnel, so the main weakness of the simple model has gone.')
p(f'**Conclusion of O4.** {BEST_REG_NAME} is selected as the best model for predicting TotalCharges: lowest validation RMSE, no overfitting and the simplest equation. It is the regression model used for prediction in O7.')

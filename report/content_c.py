# -*- coding: utf-8 -*-
from content_b import *

clf = pd.DataFrame(S['o5_table']).set_index('Model'); ch = S['o5_chosen']; cms = S['o5_cms']
bc = clf.loc[BEST_CLF]; vf1 = S['o6_val_f1']

# ================================ O5 ==========================================================
h2('4.5 Objective O5: Comparing six classifiers for churn')
h3('i. Introduction')
p('Part B asks a yes/no question: will this customer churn? Many algorithms can answer it, and none is best everywhere. The syllabus covers six that differ in how they learn, and this objective compares them on equal terms.')
h3('ii. General Description')
p('The six classifiers are Logistic Regression, k-Nearest Neighbours, Gaussian Naive Bayes, Decision Tree, Random Forest and Support Vector Machine, together with a majority-class baseline that always predicts "stay". All use the same encoded inputs and the same train, validation and test sets. '
  'One setting of each model is tuned on the validation set by the **F1 score**, because churners are a minority and F1 balances precision and recall. Logistic Regression, Decision Tree, Random Forest and SVM use balanced class weights, which make errors on churners count more. The final scores use the default threshold of 0.5.')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O5', ['Model', 'scikit-learn class', 'Setting tuned'], [('Logistic Regression', 'LogisticRegression', 'C (inverse regularisation strength)'), ('k-Nearest Neighbours', 'KNeighborsClassifier', 'k (number of neighbours)'), ('Naive Bayes', 'GaussianNB', 'var_smoothing'), ('Decision Tree', 'DecisionTreeClassifier', 'max_depth'), ('Random Forest', 'RandomForestClassifier', 'max_depth (200 trees)'), ('Support Vector Machine', 'SVC (RBF kernel)', 'C')], widths=[.28, .34, .38], align=['l', 'l', 'l'])
h3('Algorithms in brief')
p('**Logistic Regression** turns a weighted sum of the inputs into a probability with the logistic (sigmoid) function (Hosmer, Lemeshow and Sturdivant, 2013). **k-Nearest Neighbours** is a *lazy* learner: it stores the training data and predicts the majority class among the k closest customers (Cover and Hart, 1967). '
  '**Naive Bayes** applies Bayes\' theorem and assumes the inputs are independent given the class (Domingos and Pazzani, 1997). **A Decision Tree** asks a sequence of yes/no questions on the inputs, choosing each split to make the groups purer, which gives readable rules (Breiman et al., 1984). '
  '**A Random Forest** averages many trees (Breiman, 2001). **A Support Vector Machine** finds the boundary with the widest margin between the classes, and the RBF kernel lets that boundary curve (Cortes and Vapnik, 1995).')
eq('P(churn | x) = 1 / (1 + e^−(β₀ + β₁x₁ + … + β_p x_p))        (10)')
eq('P(C | x) ∝ P(C) · Π_j P(x_j | C),     d(a, b) = √( Σ_j (a_j − b_j)² ),     Gini = 1 − Σ_k p_k²        (11)')
p('Equation 11 shows the Naive Bayes rule, the Euclidean distance used by kNN and the Gini impurity used by decision trees. The evaluation measures are, with TP, FP, TN and FN the counts of the confusion matrix:')
eq('Accuracy = (TP + TN) / n,     Precision = TP / (TP + FP),     Recall = TP / (TP + FN)        (12)')
eq('F1 = 2 · Precision · Recall / (Precision + Recall),     LogLoss = −(1/n) Σ [ yᵢ ln pᵢ + (1 − yᵢ) ln(1 − pᵢ) ]        (13)')
p('AUC is the area under the ROC curve, which plots the true positive rate against the false positive rate as the threshold moves; it equals the probability that a randomly chosen churner gets a higher score than a randomly chosen stayer (Fawcett, 2006). 0.5 is chance and 1 is perfect. '
  'Log loss measures how good the predicted *probabilities* are, and punishes confident wrong answers heavily; lower is better.')
h3('iv. Analysis Results')
table('Settings chosen on the validation set', ['Model', 'Setting', 'Chosen value', 'Validation F1'], [(k, v['param'], f'{v["value"]:g}', f'{v["val_F1"]:.4f}') for k, v in ch.items()], widths=[.32, .24, .22, .22], align=['l', 'l', 'c', 'r'])
table('Test-set performance of all classifiers', ['Model', 'Accuracy', 'Log loss', 'AUC', 'Precision', 'Recall', 'F1'], [(m, f3(r.Accuracy), f3(r.LogLoss), f3(r.AUC), f3(r.Precision), f3(r.Recall), f3(r.F1)) for m, r in clf.iterrows()], widths=[.27, .11, .11, .1, .13, .12, .1], align=['l'] + ['r'] * 6)
table('Confusion matrices on the test set (actual churners: 374, actual stayers: 1,035)', ['Model', 'True negatives', 'False positives', 'False negatives', 'True positives'], [(m, cms[m][0][0], cms[m][0][1], cms[m][1][0], cms[m][1][1]) for m in clf.index if not m.startswith('Baseline')], widths=[.3, .175, .175, .175, .175], align=['l', 'r', 'r', 'r', 'r'])
p(f'Every real model is far better than the baseline: the baseline reaches {pct(clf.loc["Baseline (majority class)","Accuracy"])} accuracy but its recall and F1 are zero because it never predicts churn. Among the six classifiers, the differences are modest. '
  f'Logistic Regression and Random Forest tie for the best AUC ({clf.loc["Logistic Regression","AUC"]:.3f}); k-Nearest Neighbours has the best accuracy ({clf.loc["k-Nearest Neighbours","Accuracy"]:.3f}) and best log loss but the lowest recall of the six ({clf.loc["k-Nearest Neighbours","Recall"]:.3f}); '
  f'Naive Bayes has the highest recall ({clf.loc["Naive Bayes","Recall"]:.3f}) but the lowest accuracy ({clf.loc["Naive Bayes","Accuracy"]:.3f}) and a very high log loss ({clf.loc["Naive Bayes","LogLoss"]:.2f}).')
p(f'The log-loss values need care. Naive Bayes produces overconfident probabilities because the independence assumption counts related inputs (for example the several services) several times, so when it is wrong it is wrong with near-certainty. Class weights also push probabilities of the balanced models upwards, which raises their log loss relative to kNN. '
  f'The F1 scores of the six models lie between {clf.drop(index="Baseline (majority class)").F1.min():.3f} and {clf.drop(index="Baseline (majority class)").F1.max():.3f}; with only 374 churners in the test set, differences of about 0.01 are within noise.')
h3('v. Visualization and Interpretation')
fig('fig10_roc_curves', 'ROC curves of all classifiers on the test set (AUC in brackets)', 5.0)
p('The curves of the six classifiers are bunched together well above the diagonal, which again shows that the models have similar discriminating power. The baseline lies on the diagonal.')
fig('fig11_classification_metrics', 'Accuracy, precision, recall and F1 of the six classifiers on the test set', 6.3)
p('The bars show the trade-off between accuracy and recall. Models that catch more churners (Naive Bayes, Logistic Regression, SVM) also raise more false alarms and so lose accuracy; kNN is the opposite.')
fig('fig12_confusion_matrices', 'Confusion matrices of the six classifiers on the test set', 6.3)
p('Each matrix shows how many actual churners (bottom row) are caught and how many stayers are wrongly flagged (top right). The Random Forest raises fewer false alarms than Logistic Regression, the Decision Tree and the SVM while still catching about three-quarters of the churners.')
fig('fig13_knn_k', 'Validation F1 of k-nearest neighbours for different numbers of neighbours', 4.2)
p(f'F1 rises as k grows, because larger neighbourhoods give smoother and less noisy predictions; the best value in the grid was k = {ch["k-Nearest Neighbours"]["value"]:g}, at the edge of the grid, so a still larger k might help slightly.')
p('**Conclusion of O5.** All six classifiers are far better than the baseline and close to each other; the choice depends on whether accuracy or recall matters more, which is why the selection rule below is stated explicitly.')

# ================================ O6 ==========================================================
h2('4.6 Objective O6: Selecting the best classifier; rules and errors')
h3('i. Introduction')
p('Choosing a model is only half of the job. A manager also wants to know *why* it predicts churn, and an analyst wants to know where it fails. A shallow decision tree gives readable rules, the Random Forest gives an importance score for every input, and an error profile shows which churners are missed.')
h3('ii. General Description')
p(f'**Selection rule, fixed before the test set was used:** the classifier with the highest validation F1 is chosen. This favours models that find churners without flooding the retention team with false alarms. '
  f'Then three things are produced: the rules of a depth-3 decision tree; the importance of the inputs in the Random Forest; and an error profile of the selected classifier on the test set (true and false positives and negatives, and the share of churners missed in several groups).')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O6', ['Purpose', 'Function'], [('Tree rules and plot', 'sklearn.tree.export_text, plot_tree'), ('Random Forest importance', 'RandomForestClassifier.feature_importances_'), ('Confusion matrix', 'sklearn.metrics.confusion_matrix'), ('Error profile', 'pandas groupby on TP / FP / FN / TN groups')], widths=[.4, .6], align=['l', 'l'])
eq('Importance(j) = Σ over splits on input j of (impurity decrease × share of samples reaching the split), averaged over all trees        (14)')
p('Equation 14 describes the importance used by the Random Forest: an input that is often used for splits near the top of many trees, and that makes the groups much purer, gets a high score. The scores sum to 1. They show which inputs the model uses, not whether the relationship is causal.')
h3('iv. Analysis Results')
table('Validation F1 used for the selection', ['Model', 'Validation F1'], sorted([(k, f'{v:.4f}') for k, v in vf1.items()], key=lambda x: -float(x[1])), widths=[.6, .4], align=['l', 'r'])
p(f'**{BEST_CLF}** has the highest validation F1 ({vf1[BEST_CLF]:.4f}) and is selected, although the margin over Logistic Regression ({vf1["Logistic Regression"]:.4f}) and SVM ({vf1["Support Vector Machine"]:.4f}) is small. On the test set it achieves accuracy {bc.Accuracy:.3f}, AUC {bc.AUC:.3f}, precision {bc.Precision:.3f}, recall {bc.Recall:.3f}, F1 {bc.F1:.3f} and log loss {bc.LogLoss:.3f}: it identifies {cms[BEST_CLF][1][1]} of 374 churners and raises {cms[BEST_CLF][0][1]} false alarms.')
tt = S['o6_tree_test']
p(f'**Decision-tree rules.** A tree limited to depth 3 can be read as a short list of rules (Figure 14). Customers on a month-to-month contract with fibre-optic internet are classified as churners; month-to-month customers without fibre are classified as churners only if their tenure is at most 5 months; customers on longer contracts are classified as stayers. '
  f'This tiny tree reaches test accuracy {tt["Accuracy"]:.3f}, recall {tt["Recall"]:.3f} and F1 {tt["F1"]:.3f}, close to the full forest, which shows that most of the useful signal is in a few simple rules.')
imp = S['o6_rf_top']
table('Random Forest: the ten most important inputs', ['Rank', 'Input', 'Importance'], [(i + 1, r['Feature'], f'{r["Importance"]:.4f}') for i, r in enumerate(imp)], widths=[.12, .58, .30], align=['c', 'l', 'r'])
p(f'The most important inputs are tenure ({imp[0]["Importance"]:.3f}), the month-to-month contract indicator ({imp[1]["Importance"]:.3f}), TotalCharges ({imp[2]["Importance"]:.3f}) and MonthlyCharges ({imp[3]["Importance"]:.3f}), followed by missing online security and a two-year contract. This agrees with O1: contract, tenure and protection services matter most, and gender is irrelevant.')
pr = pd.DataFrame(S['o6_profile']).set_index('outcome')
table(f'Profile of prediction outcomes ({BEST_CLF}, test set)', ['Outcome', 'Customers', 'Mean tenure (months)', 'Mean monthly charge (USD)', 'Month-to-month', 'Fibre optic'], [(i, int(r.n), f'{r.mean_tenure:.1f}', f'{r.mean_monthly:.1f}', pct(r.share_m2m, 0), pct(r.share_fiber, 0)) for i, r in pr.iterrows()], widths=[.13, .14, .19, .22, .17, .15], align=['c', 'r', 'r', 'r', 'r', 'r'])
ms = S['o6_miss']
p(f'The missed churners (false negatives) look different from the detected ones: their mean tenure is {pr.loc["FN","mean_tenure"]:.0f} months against {pr.loc["TP","mean_tenure"]:.0f}, and only {pct(pr.loc["FN","share_m2m"],0)} are on month-to-month contracts against {pct(pr.loc["TP","share_m2m"],0)} of detected churners. '
  f'The model misses only {pct(ms["Contract"][0]["missed_rate"],0)} of month-to-month churners but {pct(ms["Contract"][1]["missed_rate"],0)} of the {int(ms["Contract"][1]["churners"])} one-year and all {int(ms["Contract"][2]["churners"])} two-year churners, and {pct(ms["tenure_band"][3]["missed_rate"],0)} of churners with more than four years of tenure. '
  'These groups are small, so the percentages are imprecise, but the direction is clear: churn among committed, long-standing customers is not explained by the available variables.')
h3('v. Visualization and Interpretation')
fig('fig14_decision_tree', 'Decision tree of depth 3: readable churn rules (orange = stay, blue = churn)', 6.3)
p('Each box shows the question asked, the share of customers reaching it and the class predicted. The tree makes the story of O1 explicit: contract type is the first question, then internet service and tenure.')
fig('fig15_rf_importance', 'Random Forest importance of the twelve most important inputs', 5.0)
p('Tenure, contract and the charges dominate. The importance of TotalCharges is partly shared with tenure, because the two are strongly correlated.')
fig('fig16_error_analysis', f'Share of actual churners missed by the selected classifier, by contract, tenure band and internet service', 6.3)
p('The model is weakest exactly where churn is rare: annual contracts, long tenure and customers without fibre. A retention team should not rely on the score alone for these customers.')
p(f'**Conclusion of O6.** {BEST_CLF} is selected by validation F1. Its behaviour is explained by a few simple rules about contract, internet service and tenure, and its main weakness is missing churners among long-standing, contracted customers.')

# ================================ O7 ==========================================================
h2('4.7 Objective O7: Using both selected models together')
h3('i. Introduction')
p('The two parts of the project answer different questions. The regression model says how much revenue a customer has generated, and the classifier says how likely the customer is to leave. A company would like to know which customers are *both* valuable and at risk. This objective combines the two predictions.')
h3('ii. General Description')
oc = S['o7_cuts']
p(f'For every customer in the test set, the selected regression model ({BEST_REG_NAME}) gives a predicted TotalCharges and the selected classifier ({BEST_CLF}) gives a churn probability. Customers are placed in four groups using two cut-offs: a churn probability of {oc["risk"]:.1f} and the median predicted TotalCharges (USD {oc["revenue_median"]:,.0f}). '
  'The actual churn rate in each group is then checked against what really happened.')
h3('iii. Specific Requirements, Functions and Formulas')
table('Functions used in O7', ['Purpose', 'Function'], [('Predicted revenue', 'final_regression_model.predict'), ('Churn probability', 'final_churn_classifier.predict_proba'), ('Grouping', 'numpy.select on the two cut-offs'), ('Group summary', 'pandas groupby')], widths=[.4, .6], align=['l', 'l'])
eq('group = f( P(churn) ≥ 0.5 ,  predicted TotalCharges ≥ median )        (15)')
h3('iv. Analysis Results')
gp = pd.DataFrame(S['o7_groups']).set_index('group')
table('Customer priority groups on the test set', ['Group', 'Customers', 'Actual churn rate', 'Mean predicted TotalCharges (USD)', 'Mean monthly charge (USD)'], [(i, int(r.customers), pct(r.actual_churn_rate), f'{r.mean_pred_revenue:,.0f}', f'{r.mean_monthly:.1f}') for i, r in gp.iterrows()], widths=[.30, .13, .17, .22, .18], align=['l', 'r', 'r', 'r', 'r'])
g1 = gp.loc['High risk, high revenue']; g2 = gp.loc['High risk, lower revenue']; g3 = gp.loc['Low risk, high revenue']; g4 = gp.loc['Low risk, lower revenue']
p(f'The grouping is informative. Customers flagged as high risk actually churn at {pct(g2.actual_churn_rate,0)} (lower-revenue group) and {pct(g1.actual_churn_rate,0)} (high-revenue group), while those flagged as low risk churn at only about {pct(g3.actual_churn_rate,0)} and {pct(g4.actual_churn_rate,0)}. '
  f'The **{int(g1.customers)} customers who are both high risk and high revenue** (mean predicted TotalCharges USD {g1.mean_pred_revenue:,.0f}, mean monthly bill USD {g1.mean_monthly:.0f}) are the natural first target for a retention offer.')
p('One caution applies. TotalCharges is revenue *to date*, so a low value often just means a new customer. The "lower revenue" high-risk group (mostly recent month-to-month customers) may still be valuable in the future; the grouping ranks past revenue, not future value.')
h3('v. Visualization and Interpretation')
fig('fig17_priority_map', 'Customer priority map: predicted TotalCharges against predicted churn probability (test set)', 5.6)
p('Each dot is a customer. The upper right quadrant (red) holds the high-risk, high-revenue customers. The vertical stripe of upper-left dots shows the many new customers with low revenue so far and high churn risk.')
p('**Conclusion of O7.** Used together, the polynomial regression model and the Random Forest classifier split customers into groups whose actual churn rates differ by a factor of about six, and they identify a clear priority group.')

# ================================ 5 CONCLUSION ================================================
h1('5. CONCLUSION')
p('The project analysed the IBM Telco Customer Churn data with the methods of the syllabus and answered each of the seven objectives with evidence.')
rr = reg.set_index('Model')
bl([f'**O1 (data).** The data are clean apart from 11 blank TotalCharges (tenure zero). Tenure, contract, internet service and protection services separate churners from stayers; gender and phone service do not. TotalCharges is strongly correlated with tenure (r = {C["TotalCharges"]["tenure"]:.2f}) and monthly charge (r = {C["TotalCharges"]["MonthlyCharges"]:.2f}).',
    f'**O2 (Simple Linear Regression).** TotalCharges = {S["o2_ols"]["intercept"]:.1f} + {S["o2_ols"]["slope"]:.1f} x tenure. Test R² = {S["o2_metrics"]["Test"]["R2"]:.3f}, RMSE USD {S["o2_metrics"]["Test"]["RMSE"]:,.0f}; the residuals fan out with tenure.',
    f'**O3 (Multiple and Polynomial).** More inputs lift test R² to {rr.loc["2. Multiple Linear","Test_R2"]:.3f}; a degree-{S["o3_best_degree"]} polynomial in tenure and monthly charge reaches {rr.loc["3. Polynomial","Test_R2"]:.3f}, because total charges are close to tenure times monthly charge.',
    f'**O4 (regression comparison).** {BEST_REG_NAME} is selected: validation RMSE USD {rr.loc[BEST_REG,"Val_RMSE"]:.1f}, test RMSE USD {rr.loc[BEST_REG,"Test_RMSE"]:.1f}, MAE USD {rr.loc[BEST_REG,"Test_MAE"]:.1f}, no overfitting. Random Forest is a close second.',
    f'**O5 (classification comparison).** All six classifiers beat the baseline with similar AUC (0.80 to 0.84) and F1 (0.58 to 0.63); they differ mainly in how they trade accuracy against recall.',
    f'**O6 (best classifier).** {BEST_CLF} is selected by validation F1 (test accuracy {bc.Accuracy:.3f}, AUC {bc.AUC:.3f}, recall {bc.Recall:.3f}, F1 {bc.F1:.3f}). Its behaviour reduces to rules on contract, internet service and tenure; it misses churn among long-tenure, contracted customers.',
    f'**O7 (combined use).** The two selected models rank customers into four groups with actual churn rates from about {pct(g4.actual_churn_rate,0)} to {pct(g2.actual_churn_rate,0)} and identify {int(g1.customers)} high-risk, high-revenue customers as the first retention target.'])
h2('5.1 Recommendations')
bl(['Use the polynomial regression model to estimate customer lifetime revenue to date, and the Random Forest to flag customers at risk of churning.',
    'Focus retention effort first on customers who are high risk and high revenue, then on new month-to-month customers, where churn is highest.',
    'Encourage month-to-month customers to move to annual contracts, and examine why fibre-optic and electronic-check customers churn more.',
    'Offer online security and technical support to customers who lack them, and test the effect before scaling it.',
    'Treat long-tenure and annual-contract customers separately, because the classifier misses most of their churn.'])
h2('5.2 Limitations')
bl(['The company is fictional and the data are a single snapshot; results show a method and should not be generalised to a real operator.',
    'Relationships are associations, not causes.',
    'TotalCharges is almost arithmetic in tenure and monthly charge, so the near-perfect regression fit is expected; it is not a forecast of future revenue.',
    'A single train/validation/test split was used. With 374 churners in the test set, small differences between classifiers are not meaningful.',
    'Classifier settings were tuned over small grids, and the best k for kNN lay at the edge of its grid.',
    'No temporal validation was possible because the data carry no dates.'])

h1('6. FUTURE SCOPE')
bl(['**Forward-looking revenue.** Predict TotalCharges from information known at sign-up (excluding tenure) or predict revenue over the next months, to make the regression a genuine customer-lifetime-value model.',
    '**More data.** Usage, complaints, network quality and competitor offers would help explain churn among long-tenure customers.',
    '**More methods.** Larger settings grids, cross-validation, regularised regression and boosting methods can be tried once the basic comparison is understood.',
    '**Business evaluation.** Attach costs and benefits to retention offers and choose the threshold that maximises profit, ideally using a randomised experiment.',
    '**Time-based validation.** With dated data, train on the past and test on the future.',
    '**Deployment.** Wrap the saved models in a simple application (src/predict.py is a starting point) and monitor their accuracy over time.',
    '**Local data.** Repeat the study on an authentic telecom data set from an open-data portal.'])

h1('7. REFERENCES')
refs = [
 'Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.',
 'Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.',
 'Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.',
 'Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.',
 'Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2–3), 103–130.',
 'Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.',
 'Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.',
 'Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.',
 'Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.',
 'IBM. Telco customer churn sample data (GitHub repository IBM/telco-customer-churn-on-icp4d). https://github.com/IBM/telco-customer-churn-on-icp4d',
 'IBM Community. Telco customer churn (11.1.3+). https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113',
 'James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.',
 'Kaggle. Telco Customer Churn (blastchar). https://www.kaggle.com/datasets/blastchar/telco-customer-churn',
 'Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15.',
 'McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.',
 'OpenML. telco-customer-churn, data set 42178. https://www.openml.org/d/42178',
 'Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.',
 'Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.',
 'Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.',
]
B.append(dict(t='refs', items=refs))

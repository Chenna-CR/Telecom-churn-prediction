---
title: "Viva Preparation Guide - Predicting Telecom Customer Revenue and Churn"
---

# 1. Thirty-second summary

I used the IBM Telco Customer Churn sample data (7,043 customers; a *fictional* company). **Part A (regression):** four models predict TotalCharges, a customer's lifetime revenue: Simple Linear, Multiple Linear, Polynomial and Random Forest. The polynomial model (degree 2 in tenure and monthly charge) was best, with test RMSE about USD 66 and R-squared 0.999. **Part B (classification):** six classifiers predict churn; Random Forest was selected (test accuracy 0.769, AUC 0.841, recall 0.754, F1 0.634). Finally I combined both models to rank customers by risk and revenue.

# 2. Numbers to remember

| Item | Value |
|---|---|
| Customers / churners | 7,043 / 1,869 (26.5%) |
| Split | 5,634 train (4,225 fit + 1,409 validation), 1,409 test, seed 42 |
| Strongest churn attribute | Contract: month-to-month 42.7% vs two-year 2.8% |
| Correlation with churn | tenure -0.35, MonthlyCharges +0.19 |
| Simple regression | TotalCharges = -201.0 + 77.0 x tenure; test R2 0.657, RMSE 1,297, MAE 876 |
| Multiple regression | test R2 0.900, RMSE 702 |
| Polynomial (degree 2) | test R2 0.999, RMSE 66.2, MAE 44.9, MSE 4,382 |
| Random Forest regression | test RMSE 79.5, train RMSE 34.4 (mild overfitting) |
| Best classifier | Random Forest: acc 0.769, log loss 0.462, AUC 0.841, precision 0.548, recall 0.754, F1 0.634 |
| Priority group | 165 test customers are high risk AND high revenue (actual churn 42%) |

# 3. Algorithms in plain language, with formulas

**OLS / simple regression.** Fit y = b0 + b1 x by minimising the sum of squared residuals. b1 = sum((x - xbar)(y - ybar)) / sum((x - xbar)^2); b0 = ybar - b1 xbar. Matrix form: b = (X'X)^-1 X'y.

**Multiple regression.** y = b0 + b1 x1 + ... + bp xp, same OLS idea with several inputs.

**Polynomial regression.** Add powers and products of inputs (t^2, t m, m^2) and fit a linear model on them. It is still linear in the coefficients.

**Metrics.** MAE = mean |y - yhat|; MSE = mean (y - yhat)^2; RMSE = sqrt(MSE); R2 = 1 - SSE/SST.

**Random Forest.** Many decision trees on bootstrap samples with random input subsets; average (regression) or vote (classification).

**Logistic regression.** P(churn) = 1 / (1 + exp(-(b0 + b'x))).

**kNN (lazy learning).** Stores the data; classifies by majority vote among the k nearest customers (Euclidean distance).

**Naive Bayes.** Bayes' theorem with the assumption that inputs are independent given the class.

**Decision tree.** Splits to reduce Gini impurity G = 1 - sum(p_k^2); gives readable rules.

**SVM.** Finds the widest-margin boundary; the RBF kernel lets it curve.

**Classification metrics.** Accuracy = (TP+TN)/n; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean; log loss = -mean[y ln p + (1-y) ln(1-p)]; AUC = area under ROC curve.

# 4. Design decisions and why

1. **Two targets.** MAE/MSE/RMSE/R2 need a number, so Part A predicts TotalCharges; churn is yes/no, so Part B uses classification.
2. **Train / validation / test.** The test set is used once; settings (degree, k, depth) are chosen on validation; scaling and encoding sit in Pipelines so they learn from training data only (no leakage).
3. **Selection rules fixed in advance.** Regression: lowest validation RMSE (simpler model if within 1%). Classification: highest validation F1.
4. **Why F1 for churn?** Churners are only 26.5%; accuracy alone is misleading (always "stay" gets 73.5%), and F1 balances precision and recall.
5. **Why balanced class weights?** To make errors on churners count more; this raises recall but also raises log loss.

# 5. Likely viva questions with model answers

**Q1. What is predictive analytics?** Using historical data and models to predict unknown outcomes: numbers (regression) or classes (classification).

**Q2. Supervised vs unsupervised learning?** Supervised learning has a known target (TotalCharges, Churn) to learn from; unsupervised learning looks for structure without a target (e.g. clustering). This project is supervised.

**Q3. How did you preprocess the data?** Converted TotalCharges from text to a number (11 blanks were customers with tenure 0, set to 0), checked missing values, duplicates and outliers (none), one-hot encoded the 15 categorical columns (45 columns after encoding with the numeric ones), and standardised numbers for scale-sensitive models.

**Q4. Why split into train, validation and test?** Train to learn, validation to choose settings, test for one honest final score. Choosing on the test set would make its score optimistic.

**Q5. What does the correlation analysis show?** TotalCharges is strongly correlated with tenure (0.83) and MonthlyCharges (0.65); tenure has the strongest negative correlation with churn (-0.35).

**Q6. Explain OLS.** It picks the line that minimises the sum of squared vertical distances between the data and the line. I calculated slope 76.97 and intercept -201.04 by formula and got the same values from scikit-learn.

**Q7. What do MAE, MSE, RMSE and R2 tell you?** MAE: average dollar error. MSE: average squared error (punishes big mistakes). RMSE: MSE in dollars. R2: share of variation explained. For my best model: MAE 44.9, RMSE 66.2, R2 0.999.

**Q8. Why was simple linear regression not enough?** Tenure alone gives R2 0.657; the residuals fan out with tenure because customers with different monthly bills accumulate charges at different speeds.

**Q9. What does multiple regression add?** More inputs, mainly MonthlyCharges; R2 rises to 0.900. But it is still linear and cannot represent that total charges are roughly tenure times monthly charge.

**Q10. Why is polynomial regression so accurate here?** The degree-2 model includes the tenure x monthly-charge product, and TotalCharges is almost exactly that product (the ratio has median 1.00). So the result is real but close to arithmetic, and not a deep discovery. I say this openly.

**Q11. How did you choose the polynomial degree?** Fitted degrees 1 to 5 and compared validation RMSE: it dropped from about 740 to 65 at degree 2 and did not improve afterwards, so degree 2.

**Q12. Overfitting: how did you check?** Compared training and test errors. The polynomial model has almost equal RMSE (67 vs 66). The Random Forest has 34 on training but 80 on test, so it overfits a little.

**Q13. Why Random Forest in the regression comparison?** It handles curves and interactions without being told, and it is on the syllabus. It was a close second, but the polynomial model is simpler and slightly more accurate.

**Q14. How does a Random Forest work?** Many trees, each trained on a random bootstrap sample and random inputs; their predictions are averaged. Averaging reduces variance.

**Q15. Why is accuracy not enough for churn?** The baseline that always says "stay" gets 73.5% accuracy but finds no churner (recall 0).

**Q16. Which classifier did you choose and why?** Random Forest, because it had the highest validation F1 (0.632). On the test set: accuracy 0.769, AUC 0.841, precision 0.548, recall 0.754, F1 0.634. The other models are close, so I do not claim it is far better.

**Q17. Why does Naive Bayes have such a high log loss (2.75)?** Its independence assumption double-counts related inputs, making its probabilities overconfident; when it is wrong, it is confidently wrong.

**Q18. What is a lazy learner?** kNN does no training: it stores the data and does the work at prediction time by looking up the nearest neighbours.

**Q19. What is AUC?** The probability that a random churner gets a higher score than a random stayer; 0.5 is chance, 1 is perfect. Logistic Regression and Random Forest both have 0.841.

**Q20. Explain the confusion matrix of your best model.** Out of 374 actual churners it finds 282 (true positives) and misses 92; out of 1,035 stayers it correctly keeps 802 and wrongly flags 233.

**Q21. What do the decision-tree rules say?** First split on contract: customers not on month-to-month are predicted to stay; month-to-month customers with fibre-optic internet are predicted to churn; month-to-month without fibre churn if tenure is at most 5 months.

**Q22. Which inputs matter most for the forest?** Tenure, month-to-month contract, TotalCharges and MonthlyCharges, then missing online security. It agrees with the EDA.

**Q23. Where does the classifier fail?** It misses 94% of the 36 one-year and all 9 two-year churners and 77% of churners with over four years of tenure; the data do not explain why long-standing customers leave.

**Q24. How do you use both models together?** Predicted TotalCharges (polynomial) and churn probability (Random Forest) split customers into four groups; the 165 high-risk, high-revenue customers are the first retention target. Caution: TotalCharges is revenue to date, so new customers look low value.

**Q25. What are the limitations?** Fictional single-snapshot data; associations not causes; TotalCharges is nearly arithmetic in tenure and monthly charge; one split; small grids; no temporal validation.

# 6. Limitations to state before they ask

- Fictional company, no dates.
- High R-squared reflects billing arithmetic.
- Differences among classifiers are small (F1 0.58 to 0.63).
- kNN's best k was at the edge of the grid.

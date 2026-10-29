# Dataset documentation
- **Name:** Telco Customer Churn (IBM sample data); 7,043 rows x 21 columns; fictional company, not real customer records
- **File used:** `Telco-Customer-Churn.csv`, 970,457 bytes, from https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **Other listings:** Kaggle https://www.kaggle.com/datasets/blastchar/telco-customer-churn ; OpenML https://www.openml.org/d/42178
- **Known issue:** `TotalCharges` has 11 blanks, all for customers with tenure 0 (set to 0.0 in `src/churn_lib.clean`)
- **Data dictionary:** `results/data_dictionary.csv`; **Licence:** confirm on the source pages before redistribution.

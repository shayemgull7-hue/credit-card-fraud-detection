# Credit Card Fraud Detection with Imbalanced Data

This MSc Business Analytics dissertation project investigates how class-imbalance handling techniques affect machine-learning performance in credit card fraud detection.

## Business problem

Fraud represented only **0.17%** of the cleaned dataset. In this setting, accuracy alone can be misleading because a model could classify almost every transaction as legitimate while failing to detect fraud. The project therefore focused on fraud detection performance, false alarms and the estimated cost of missed fraud.

## Dataset

- **Source:** [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Cleaned transactions:** 283,726
- **Fraudulent transactions:** 473 (0.17%)
- **Legitimate transactions:** 283,253 (99.83%)
- **Missing values:** None

The dataset is not included in this repository. To run the notebook, download the Kaggle CSV, place it in the same folder as the notebook and rename it `credit_card_imbalance_dataset.csv`.

## Approach

- Removed duplicate records and explored the class distribution and transaction amounts.
- Used a stratified 80:20 train-test split.
- Compared Logistic Regression, Random Forest and XGBoost.
- Applied Random Oversampling, Random Undersampling and SMOTE.
- Extended the Random Forest comparison with ADASYN and SMOTE-Tomek.
- Evaluated precision, recall, F1-score, PR-AUC, ROC-AUC and accuracy.
- Used an illustrative cost matrix of **£10 per false positive** and **£500 per false negative**.
- Tested decision thresholds from 0.05 to 0.95 on a validation split.
- Used SHAP to explain global and individual predictions.

## Key results

- Random Forest with SMOTE provided the strongest practical balance in the initial comparison.
- Random Forest with SMOTE-Tomek achieved the highest precision (**94.74%**) and F1-score (**84.21%**).
- Random Forest with SMOTE achieved higher recall (**76.84%**) and the lowest estimated cost at the standard threshold.
- Reducing the decision threshold from **0.50 to 0.12** reduced the estimated test cost from **£11,070 to £9,040**, a reduction of approximately **18.3%** under the assumed cost matrix.
- SHAP identified V14, V12, V4, V3 and V10 as the most influential anonymised features.

## Repository files

- [`credit_card_fraud_detection_GitHub.ipynb`](credit_card_fraud_detection_GitHub.ipynb) - complete Python analysis with saved outputs
- [`Credit_Card_Fraud_Detection_Poster.pdf`](Credit_Card_Fraud_Detection_Poster.pdf) - MSc degree-show poster

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, XGBoost and SHAP.

## Limitations

- The dataset contains anonymised transactions from 2013 and may not represent current fraud patterns.
- The false-positive and false-negative costs are illustrative assumptions, not verified banking costs.
- Extensive hyperparameter optimisation, repeated cross-validation and statistical testing between model predictions were outside the project scope.
- For threshold optimisation, scaling had already been fitted on the initial training split before it was divided into subtraining and validation data. A future version should fit preprocessing within the subtraining fold.
- The prototype requires validation on recent institution-specific data before any practical deployment.

## Author

**Nadia Ashraf**  
MSc Business Analytics, Robert Gordon University

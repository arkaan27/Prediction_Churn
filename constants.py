"""
Module to store constants required in churn_library.py file
Author: Arkaan Quanunga
Date: 11/01/2022
"""

DATA_PTH = './data/bank_data.csv'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

EDA_IMG_PTHS = {
    'images/eda/churn_distribution.png',
    'images/eda/marital_status_distribution.png',
    'images/eda/customer_age_distribution.png',
    'images/eda/heatmap.png',
    'images/eda/total_transaction_distribution.png'
}

RESULTS_IMG_PTHS = {
    'images/results/rf_classification_report.png',
    'images/results/lr_classification_report.png',
    'images/results/feature_importances.png',
    'images/results/lr_roc_curve.png',
    'images/results/rf_roc_curve.png',
    'images/results/rf_shap_values_summary.png'
}
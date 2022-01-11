# library doc string
"""
Module for assisting the churn scripting

Author: Arkaan Quanunga
Date: 25/12/2021
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay
import constants


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    """

    data = pd.read_csv(pth)
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


def perform_eda(data):
    """
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    """
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.savefig('./images/eda/Churn_histogram.png')

    # Saves Customer Age Distribution
    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.savefig('./images/eda/Customer_Age_Histogram.png')

    # Saves Martial Status Normalised graph
    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/Martial_Status_Normalized_graph.png')

    # Saves Total Transaction
    plt.figure(figsize=(20, 10))
    sns.distplot(data['Total_Trans_Ct'])
    plt.savefig('./images/eda/Total_Trans_Ct.png')

    # Saves Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    for col in category_lst:
        means = data.groupby(col)['Churn'].mean()
        data[col + response] = data[col].map(means)

    return data
    # category_group = data.groupby(response).mean()['Churn']
    #
    # for val in data[response]:
    #     category_lst.append(category_group.loc[val])
    #     data[response + '_Churn'] = category_lst


def perform_feature_engineering(data):
    """
    input:
              data: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y = data.Churn
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = data[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # RF model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/rf_classification_report.png')

    # Logistic regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/lr_classification_report.png')


def roc_curve_plot(model, X, y, output_pth):
    """

    :param model: model to be trained
    :param X: X data values
    :param y: y data values
    :param output_pth: path to save the figure
    :return:
    """
    plt.figure(figsize=(15, 8))
    RocCurveDisplay.from_estimator(model, X, y)
    plt.savefig(output_pth)


def shap_explainer_plot(model, X, output_pth):
    """

    :param model: model to be plotted for
    :param X: X data values
    :param output_pth: path to save the figure (str)
    :return:
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.savefig(output_pth)


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    # Calculate feature importance
    importance = model.feature_importances_

    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importance[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # Deep grid search
    # param_grid = {
    #     'n_estimators': [200, 500],
    #     'max_features': ['auto', 'sqrt'],
    #     'max_depth': [4, 5, 100],
    #     'criterion': ['gini', 'entropy']
    # }

    # Shallow grid search
    param_grid = {
        'n_estimators': [20, 50],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4],
        'criterion': ['gini']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save results to images/
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        'images/results/feature_importances.png')

    roc_curve_plot(cv_rfc, X_test, y_test, 'images/results/rf_roc_curve.png')
    roc_curve_plot(lrc, X_test, y_test, 'images/results/lr_roc_curve.png')

    shap_explainer_plot(cv_rfc.best_estimator_, X_test, 'images/results/rf_shap_values_summary.png')

    # Save best model to models/
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def main():
    """
    Running the file
    :return:
    """
    data = import_data('./data/bank_data.csv')
    perform_eda(data)
    encoded_df = encoder_helper(data, constants.CAT_COLUMNS, '_Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(encoded_df)
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

"""
Main file for the model

Author: Arkaan Quanunga

Date: 11/01/2022
"""
import churn_library as cl
import constants


if __name__ == '__main__':

    # Importing data
    data_frame = cl.import_data('./data/bank_data.csv')

    # Performing Exploratory Data Analysis
    cl.perform_eda(data_frame)

    # Encoding the data
    encoded_df = cl.encoder_helper(data_frame, constants.CAT_COLUMNS, '_Churn')

    # Splitting the data
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(encoded_df)

    # Training the model
    cl.train_models(X_train, X_test, y_train, y_test)


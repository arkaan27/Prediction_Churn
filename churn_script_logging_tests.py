"""
Module for testing churn_library.py

Author: Arkaan Quanunga

Date: 11/01/2021

"""
# Importing variables
import os
import logging
import glob
from math import isclose
import pandas as pd
from churn_library import import_data, perform_eda, \
    encoder_helper, perform_feature_engineering, \
    train_models
import constants

# Basic Logging
logging.basicConfig(
    filename='./logs/churn_library.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Tests


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    # Logging
    logging.info("TESTING: import_data function")
    try:
        data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """
    test perform eda function
    """

    logging.info("TESTING: Eda function")

    # Making sure all files in images are removed
    necessary_img_pths = constants.EDA_IMG_PTHS
    existing_files = set(glob.glob('images/eda/*.png'))
    files_removal = necessary_img_pths.intersection(existing_files)
    for pth in files_removal:
        os.remove(pth)

    # Importing data
    data = import_data(constants.DATA_PTH)

    # Running the function
    perform_eda(data)

    # assert
    resultant_image_files = glob.glob('images/eda/*.png')

    # Checking all files exists in the pth
    try:
        for pth in constants.EDA_IMG_PTHS:
            assert pth in resultant_image_files
        logging.info(
            "SUCCESS: All necessary eda image files have been created")
    except AssertionError as err:
        logging.error(
            'ERROR: %s required output not generated by perform_eda() function',
            pth)
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """
    logging.info("TESTING: test_encoder_helper function")

    # Importing data

    data = import_data(constants.DATA_PTH)

    # consists dtypes
    try:
        encoded_data = encoder_helper(
            data_frame=data,
            category_lst=constants.CAT_COLUMNS,
            response='_Churn')
    except TypeError as err:
        logging.error(
            "Testing encoder_helper: You have not provided the correct dtypes to encoder_helper()")
        raise err

    # assert
    for col in constants.CAT_COLUMNS:
        # check that each required column is created
        try:
            assert f"{col}_Churn" in encoded_data.columns
            logging.info("SUCCESS: %s column created", col)
        except AssertionError as err:
            logging.error("%s column wasn't created", col)
            raise err

        # check that each of the columns is now a float value
        try:
            assert pd.api.types.is_float_dtype(data[f"{col}_Churn"])
        except AssertionError as err:
            logging.error(
                "%s column should be float dtype, something went wrong with encoding", col)
            raise err

        # check that the number of unique values in each encoded column matches the number
        # of unique values in the original column
        try:
            assert len(
                data[col].unique()) == len(
                data[f"{col}_Churn"].unique())
            logging.info("SUCCESS: Unique values check is working for %s", col)
        except AssertionError as err:
            logging.error(
                "The number of unique values in the encoded column %s does not match the original",
                col)
            raise err

        # check some values
        try:
            assert isclose(
                data.groupby('Gender')['Gender_Churn'].mean()['F'],
                0.17357222844)
            assert isclose(data.groupby('Education_Level')['Education_Level_Churn'].mean()[
                               'High School'], 0.15201192250372578)
            assert isclose(data.groupby('Marital_Status')['Marital_Status_Churn'].mean()[
                               'Married'], 0.15126946874333264)
            logging.info("SUCCESS: Columns look to be correctly encoded")
        except AssertionError as err:
            logging.error(
                "Target encoding function looks like it's encoding incorrectly")
            raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    logging.info("Testing perform_feature_engineering")
    # arrange
    data = import_data(constants.DATA_PTH)

    encoded_df = encoder_helper(
        data_frame=data,
        category_lst=constants.CAT_COLUMNS,
        response='_Churn')

    # Running the function
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            encoded_df)
    except TypeError as err:
        logging.error(
            "ERROR: You have not provided the correct dtypes to perform_feature_engineering()")
        raise err

    # assert
    try:
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except AssertionError as err:
        logging.error(
            "ERROR: Train test split isn't outputting the correct data types")
        raise err

    # Test for train test split
    try:
        assert len(X_train) + len(X_test) == len(encoded_df)
        logging.info(
            "SUCCESS: All feature engineering tests passed")
    except AssertionError as err:
        logging.error(
            "ERROR: Train test split doesn't seem to be "
            "working properly in perform_feature_engineering()")
        raise err


def test_train_models():
    """
    test train_models
    """
    logging.info("TESTING: train_models function")

    # arrange

    # cleaning directory
    existing_files = set(glob.glob('images/results/*.png'))
    files_to_remove = constants.RESULTS_IMG_PTHS.intersection(existing_files)
    for pth in files_to_remove:
        os.remove(pth)

    # populating variables
    data = import_data(constants.DATA_PTH)
    encoded_df = encoder_helper(data, constants.CAT_COLUMNS, '_Churn')
    X_train, X_test, y_train, y_test = perform_feature_engineering(encoded_df)

    # Running the function
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Trained the model: SUCCESS")
    except NameError as err:
        logging.error(
            "ERROR: You haven't defined one of the required args in train_models()")
        raise err

    # assert
    resultant_image_files = glob.glob('images/results/*.png')
    try:
        for pth in constants.RESULTS_IMG_PTHS:
            assert pth in resultant_image_files
        logging.info("SUCCESS: All train_models tests passed")
    except AssertionError as err:
        logging.error(
            'ERROR: %s output not generated by train_models() function',
            pth)
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()

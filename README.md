# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Churn prediction model for bank data. This project reads a csv file bank_data.csv and performs exploratory data analysis in modular manner.
The code is production level ready for deployment as it is fully tested and logged.
This code has followed PEP8 compliance as required.
            


## Running Files
1. Set up Python environment

Install dependencies in the environment.yml file, ideally in a new conda environment by running

`$ conda env create -f environment.yml`

If this does not work, run the following commands in the terminal:

`$ pip install joblib`

`$ pip install sklearn`

`$ pip install scikit-learn`

`$ pip install ipython`


This will create a new conda environment named predict_churn, unless you change the name of the environment by adjusting the first line of environment.yml
2. Run churn_library.py

To preprocess the raw data, conduct exploratory data analysis on that data, train the models, and export model results run the following from the command line:

`$ python main.py`

Testing - OPTIONAL

With pytest installed you can simply run pytest from the shell from within the project directory to run the tests defined in test_churn_script_logging_and_tests.py.

Alternatively you can run the test file directly, i.e.

`$ python test_churn_script_logging_and_tests.py`


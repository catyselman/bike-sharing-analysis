# bike-sharing-analysis
Washington DC Bike Sharing User Predictions

In this analysis we have developed a model to predict the number of bike sharing users on an hourly basis in the city of Washington DC.

Based on the historical data for 2 years on the number bike sharing users a regression model was built considering the date of each recording and other related descriptive features such as the temperature and humidity of this specific day.

For this regression model a number of algorithms were tried such as Linear Regression, Random Forest Regressor and Gradient Boosting trees.

To implement the analysis and model summarized above the following libraries were used:
-pandas
-numpy
-sklearn
-dask

Description of Python notebooks:
1) 1_EDA.ipynb: Notebook with preliminary data analysis.
2) 2_FeatureEngineering.ipynb: Notebook with different feature engineering attempts (feature creation, selection and reduction)
3) 3_ParameterTuning+FinalModel.ipynb: Final model GridSearch for tuning.


Description of Python scripts:
1) utils.py: Basic functions used to train models with different inputs (feature engineerig functions)
2) preprocessing.py: Helper functions for preprocessing of datasets.
3) feature_creation.py: Helper functions for feature engineering and creation.
4) feature_selection.py: Helper functions for feature selection options and techniques.
5) feature_reduction.py: Helper functions for feature reduction options.

All notebooks/Python scripts have been developed using Python 3.7.
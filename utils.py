from datetime import datetime
import dask.dataframe as dd
from dask_ml.preprocessing import *
from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from dask.distributed import Client
import joblib
from sklearn.metrics import r2_score
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline


PATH = "https://gist.githubusercontent.com/catyselman/9353e4e480ddf2db44b44a79e14718b5/raw/ded23e586ca5db1b4a566b1e289acd12ebf69357/bikeshare_hourly.csv"
NUM_QUARTERS = 8
SEED = 20192602


def update_df(data, ops):
    """
    Performs all of the operations in ops on data, and returns the result

    :param data: a pandas dataframe where each row is an hour
    :param ops: An iterable that contains functions with a signature [DataFrame -> DataFrame]
    :return: a pandas dataframe with all of the ops applied
    """
    data = data.copy()
    for op in ops:
        data = op(data)

    return data


def get_train(data):
    """
    Get the train data from the time series data

    Since this is two years' worth of data and we want just the last quarter of 2012, we take the first 7/8s of the data
     for the train set

    :param data: a pandas dataframe where each row is an hour, containing every hour from 2011-2012
    :return: a pandas dataframe containing just 2011 and the first 3 quarters of 2012
    """    
    train = data[data['instant'] < 15212]
    return train


def get_holdout(data):
    """
    Get the test data from the time series data

    Since this is two years' worth of data and we want just the last quarter of 2012, we take the last 1/8 of the data
     for the test set

    :param data: a pandas dataframe where each row is an hour, containing every hour from 2011-2012
    :return: a pandas dataframe containing just the last quarter of 2012
    """
    test = data[data['instant'] >= 15212]
    return test


def model_name(model):
    """
    Give the name of the input model

    :param model: An sklearn estimator
    :return: the name of the estimator
    """
    name = type(model).__name__

    if isinstance(model, Pipeline):
        name = model.steps[-1][1]

    return name

def get_numeric_features(data):
    """
    Write documentation
    """
    numeric_dtypes = ['int16', 'int32', 'int64', 'int8',
                      'float16', 'float32', 'float64', 'uint8']
    numeric_features = []
    for i in data.columns:
        if data[i].dtype in numeric_dtypes: 
            numeric_features.append(i)
    return numeric_features


def get_object_features(data):
    """
    Write documentation
    """
    object_dtypes = ['object']
    object_features = []
    for i in data.columns:
        if data[i].dtype in object_dtypes: 
            object_features.append(i)
    return object_features


def get_categorical_features(data):
    """
    Write documentation
    """
    numericalColumns = get_numeric_features(data)
    return list(set(data.columns) - set(numericalColumns) - set(get_object_features(data)))

def pipeline_casero(data, preprocessing=[], creation=[], reduction=[], selection=[], models=[]):
    """
    A homemade pipeline to automate all the steps of data preparation, feature creation, feature selection, feature
    reduction, and outputting a fitted model

    This is not strictly necessary as it does not add more functionality than an sklearn Pipeline, but we thought it
    would be easier to use for our purposes and it has the added benefit of allowing us to control the verbosity of
    the output.


    :param path: A path to the file containing the data for the pipeline
    :param preprocessing: An iterable containing all the preprocessing steps (functions with signature [DataFrame -> DataFrame]
    :param creation: An iterable containing all the feature creation steps (functions with signature [DataFrame -> DataFrame]
    :param reduction: An iterable containing all the dimensionality reduction steps (functions with signature [DataFrame -> DataFrame]
    :param selection: An iterable containing all the feature selection steps (functions with signature [DataFrame -> DataFrame]
    :param models: An array of dicts containing the name for the model ("name"), the sklearn estimator ("model"),
                    and the parameters for Grid Search Cross Validation ("params")
    :return: A fitted model that represents the best model out of all the ones in 'models'
    """
    
    print("Beginning pipeline at {}\n".format(datetime.now()))
    print("Performing preprocessing steps...")
    data = update_df(data, preprocessing)
    print("Preprocessing completed at {}, performed {} steps".format(datetime.now(), len(preprocessing)))
    print("New Shape of data: {0}\n".format(len(data.columns)))

    print("Performing feature creation...")
    data = update_df(data, creation)
    print("Feature Creation completed at {}, performed {} steps".format(datetime.now(), len(creation)))
    print("New Shape of data: {0}\n".format(len(data.columns)))

    print("Dummifying...")
    categoricalDF = data[get_categorical_features(data)].categorize()
    de = DummyEncoder()
    trn = de.fit_transform(categoricalDF)
    numericalDF = data[get_numeric_features(data)]
    categoricalDummy = trn.repartition(npartitions=5)
    numericalDF = numericalDF.repartition(npartitions=5)
    data = dd.concat([categoricalDummy, numericalDF], axis=1)
    print("New Shape of data: {0}\n".format(len(data.columns)))

    print("Performing dimensionality reduction...")
    data = update_df(data, reduction)
    print("Dimensionality reduction completed at {}, performed {} steps".format(datetime.now(), len(reduction)))
    print("New Shape of data: {0}\n".format(len(data.columns)))
    
    best_model = select_best_model(models, data, selection)

    return best_model

def grid_search_model(data, model_meta, folds=3):
    """
    Perform Grid Search Cross Validation on the input model

    :param data: a pandas dataframe where each row is an hour
    :param model_meta: An dict containing the name for the model ("name"), the sklearn estimator ("model"),
                    and the parameters for Grid Search Cross Validation ("params")
    :param folds: The number of splits for cross validation
    :return: a tuple containing the best R^2 score found, the parameters used to obtain that score,
                and the estimator retrained on the whole dataset
    """
    model = model_meta["model"]
    model_params = model_meta["params"]
    model_name = model_meta["name"]

    X = data.drop("cnt", axis=1)
    y = data["cnt"]

    tscv=TimeSeriesSplit(n_splits=folds)

    grid_search = GridSearchCV(estimator=model, param_grid=model_params, scoring="r2", cv=tscv, refit=True, n_jobs = -1)
        
    with joblib.parallel_backend('dask'):
        grid_result = grid_search.fit(X, y)

    print("\tAverage result for best {}: {} +/- {:.5f}"
          .format(model_name,
                  grid_result.best_score_,
                  grid_result.cv_results_["std_test_score"][np.argmax(grid_result.cv_results_["mean_test_score"])]))

    print("\tBest parameters for {0}: {1}".format(model_name, grid_result.best_params_))

    # Need metrics to choose model, best estimator will have already been retrained on whole data set
    return grid_result.best_score_, grid_result.best_params_, grid_result.best_estimator_


def select_best_model(models, data, selection):
    """
    Given several models and data to fit, return the best model based on Grid Search Cross Validation

    :param models: An array of dicts containing the name for the model ("name"), the sklearn estimator ("model"),
                    and the parameters for Grid Search Cross Validation ("params")
    :param data: a pandas dataframe where each row is an hour
    :return: an sklearn estimator that is the best model refit on the whole train data
    """
#     instant = data['instant']
    
#     scaler = MinMaxScaler()
#     scaledData = scaler.fit_transform(data.drop("instant", axis=1))

#     scaledData = scaledData.repartition(npartitions=5)
#     scaledData = scaledData.reset_index(drop=True)
    
#     instant = instant.repartition(npartitions=5)
#     instant = instant.reset_index(drop=True)

#     data = dd.concat([instant, scaledData], axis=1)

    train = get_train(data)
    holdout = get_holdout(data)
    print("Performing feature selection...")
    train = update_df(train, selection)
    print("Feature Selection completed at {}, performed {} steps".format(datetime.now(), len(selection)))
    print("New Shape of train: {0}\n".format(len(train.columns)))

    holdout = holdout[train.columns]

    print("Scoring models....")
    testX = holdout.drop("cnt", axis=1)
    testY = holdout['cnt']
    
    results = [grid_search_model(train, model) for model in models]
        
    models = []
    for i in range(len(results)):
        models.append(results[i][2])

    with joblib.parallel_backend('dask'):
        scores = []
        for model in models:
            y_pred = model.predict(testX)
            scores.append(r2_score(testY, y_pred))
    
    best_score_index = np.argmax(scores)
    best_score = models[best_score_index]
    best_model = results[best_score_index][2]    
    
    print("\nBest model: {0} with params {1}".format(model_name(best_model), results[best_score_index][1]))

    print("Evaluating model on the holdout...")
    final_r2 = r2_score(holdout.cnt, best_model.predict(holdout.drop("cnt", axis=1)))
    print("Final R2: {0}".format(final_r2))
    print("\nPipeline finished! Completed execution at {}. Returning model...".format(datetime.now()))
    
    return best_model


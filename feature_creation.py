import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicTransformer
from utils import *
from sklearn.model_selection import TimeSeriesSplit
from dask_ml.model_selection import GridSearchCV


def categorize_time(r):
    """
    Helper function for 'commute_hours'

    :param r: a row of the data frame
    :return: Boolean, True if hour is during commute hours else flase
    """
    response = (((r['hr'] >= 5 and r['hr'] < 10) or (r['hr'] >= 16 and r['hr'] < 20)) and r['workingday'])
    return response

def commute_hours(data):
    """
    Creates a column that declares whether the hour is during commute hours

    :param data: a Dask dataframe where each row is an hour
    :return: a Dask dataframe containing the new column
    """
    print("\tAdding variable for Commute Hours, 1 for yes and 0 for false")
    data["commute_hours"] = data.apply(categorize_time, axis=1, meta=pd.Series())
    data['commute_hours'] = data.holiday.astype('bool')
    return data


def weather_cluster(data):
    """
    Creates a column that gives a cluster id based on KMeans clustering of only weather-related features

    :param data: a Dask dataframe where each row is an hour
    :return: a Dask dataframe containing the new column
    """
    print("\tAdding clustering variable based on weather-related features...")
    df = data[["weathersit", "temp", "atemp", "hum", "windspeed", "instant"]]
    
    print("\nDummifying to create clusters...")
    categoricalDF = df[get_categorical_features(df)].categorize()
    de = DummyEncoder()
    trn = de.fit_transform(categoricalDF)
    numericalDF = data[get_numeric_features(df)]
    categoricalDummy = trn.repartition(npartitions=5)
    numericalDF = numericalDF.repartition(npartitions=5)
    to_cluster = dd.concat([categoricalDummy, numericalDF], axis=1)
    print("New Shape of data: {0}\n".format(len(to_cluster.columns)))

    train = get_train(to_cluster)
    holdout = get_holdout(to_cluster)

    train.drop("instant", axis=1)
    holdout.drop("instant", axis=1)

    with joblib.parallel_backend('dask'):
        kmeans = KMeans(n_clusters=5, random_state=SEED).fit(train.values)  # magic numbers, blech

    clusters = np.append(kmeans.labels_, kmeans.predict(holdout.values))
    clustersDD = dd.from_array(clusters)

    clustersDD = clustersDD.repartition(npartitions=5)
    clustersDD = clustersDD.reset_index(drop=True)
    data = data.repartition(npartitions=5)
    data = data.reset_index(drop=True)

    data["weather_cluster"] = clustersDD

    data["weather_cluster"] = data["weather_cluster"].astype("category")

    return data


def deep_features(data):
    """
    Performs deep feature synthesis on the dataframe to generate new columns based on different groups

    Currently, the only group is by season.

    :param data: a Dask dataframe where each row is an hour
    :return: a Dask dataframe containing the new column
    """
    print("\tPerforming Deep Feature Synthesis...")
    df = data.compute()
    df = df.copy()
    count = df["cnt"]
    df = df.drop("cnt", axis=1)

    instant = df['instant']

    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id="bikeshare_hourly",
                                  index="instant",
                                 dataframe=df)

    es = es.normalize_entity(base_entity_id="bikeshare_hourly",
                             new_entity_id="seasons",
                             index="season")

    f_mtx, f_defs = ft.dfs(entityset=es,
                           target_entity="bikeshare_hourly",
                               agg_primitives=["std", "max", "min", "mean"]) # ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "n_unique", "mode"]
    extra_features = f_mtx.iloc[:, len(df.columns):len(f_mtx)]
    for f in extra_features.columns:
        print("\t\tCreated feature {}".format(f))

    f_mtx = f_mtx.reset_index()
    count = count.reset_index()
    f_mtx["cnt"] = count['cnt']

    f_mtx= dd.from_pandas(f_mtx, npartitions = 5)

    return f_mtx


def subcount_forecast(data, feature):
    """
    Creates a new a column that is the predicted value of the input feature

    Essentially an abstraction for 'prediction_forecasts'

    :param data: a Dask dataframe where each row is an hour
    :param feature: a String containing the feature that should be forecasted (one of: casual, registered)
    :return: a Dask dataframe containing the new column
    """
    var_name = feature + "_forecast"
    print("\tAdding {} variable...".format(var_name))
        
    print("Dummifying...")
    categoricalDF = data[get_categorical_features(data)].categorize()
    categoricalDF.head(10)
    de = DummyEncoder()
    if(len(categoricalDF.columns) > 0):
        trn = de.fit_transform(categoricalDF)
        numericalDF = data[get_numeric_features(data)]
        categoricalDummy = trn.repartition(npartitions=5)
        numericalDF = numericalDF.repartition(npartitions=5)
        tempDF = dd.concat([categoricalDummy, numericalDF], axis=1)
    else:
        tempDF = data
    
    df = tempDF.drop("cnt", axis=1)
    print("New Shape of data: {0}\n".format(len(df.columns)))

    to_predict = dd.read_csv(PATH, blocksize=25e4)[feature]
    to_predict = to_predict.repartition(npartitions=5)
    to_predict = to_predict.reset_index(drop=True)
    
    df = df.repartition(npartitions=5)
    df = df.reset_index(drop=True)

    df = dd.concat([df, to_predict], axis=1)
    train = get_train(df)

    model = RandomForestRegressor(random_state=SEED)
#    model_params = {"n_estimators": list(range(10, 110, 10))}
    model_params = {}

    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator=model, param_grid=model_params, scoring="r2", cv=tscv, refit=True)

    with joblib.parallel_backend('dask'):
        grid_search.fit(train.drop(feature, axis=1), train[feature])
    
    print("\t\tPredictions for GridSearchCV on {}: {:.5f} +/- {:.5f}"
          .format(feature,
              grid_search.best_score_,
              grid_search.cv_results_["std_test_score"][np.argmax(grid_search.cv_results_["mean_test_score"])]))


    predicted = dd.from_pandas(pd.Series(grid_search.best_estimator_.predict(tempDF.drop("cnt", axis=1))), npartitions=5)
    predicted = predicted.repartition(npartitions=5)
    predicted = predicted.reset_index(drop=True)

    return predicted


def prediction_forecasts(data):
    """
    Creates two new columns, one that is predictions for the 'casual' column and the other for the 'registered' column

    We wouldn't actually have the registered or casual variables at the time of predicting the "cnt" variable, but we
    do have all the other information and we have past information for registered and count. So the theory here is that
    we can use predictions for these values that would be accurate enough to be extremely helpful as features for a
    model

    :param data: a Dask dataframe where each row is an hour
    :return: a Dask dataframe containing the new columns
    """
    casual = subcount_forecast(data, "casual")
    registered = subcount_forecast(data, "registered")

    data = data.repartition(npartitions=5)
    data = data.reset_index(drop=True)
    data["casual_forecast"] = casual
    data["registered_forecast"] = registered
    data = data.dropna()

    return data



def genetic_programming(data):
    """
    Creates new features based on mathemtical calculations of exiting feautres.
    
    :param data: a Dask dataframe where each row is an hour
    :return: a Dask dataframe containing the new columns
    """
    print('Creating features through genetic programming...')
    target = 'cnt'
    y = data[target]
    X = data.copy().drop(target,axis=1)
    
    with joblib.parallel_backend('dask'):
        functions = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min' ]
        gp = SymbolicTransformer(generations=20, population_size=2000,
                        hall_of_fame=100, n_components=15,
                        function_set=functions,
                        parsimony_coefficient=0.0005,
                        max_samples=0.9, verbose=1,
                        random_state=999, n_jobs=3)
        gp_features = gp.fit_transform(X,y)
    
    print('Number of features created out of genetic programing: {}'.format(gp_features.shape))
    gp_dask = dd.from_array(gp_features)
    
    newData = dd.merge(data, gp_dask)
    
    return newData

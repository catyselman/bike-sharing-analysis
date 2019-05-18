import pandas as pd
from sklearn.decomposition import PCA
from utils import get_train, get_holdout
import joblib
import dask.dataframe as dd


def pca(data):
    train = get_train(data).drop("cnt", axis=1)
    test = get_holdout(data).drop("cnt", axis=1)

    print("\tPerforming PCA dimensionality reduction...")

    with joblib.parallel_backend("dask"):
        pca = PCA(n_components=0.95, svd_solver="full").fit(train)
        pca_train = pd.DataFrame(data=pca.transform(train))
        pca_test = pd.DataFrame(data=pca.transform(test))

    new_df = pca_train.append(pca_test)
    new_df["cnt"] = data["cnt"]
    new_df["instant"] = data["instant"]

    new_df = dd.from_pandas(new_df, npartitions=5)

    return new_df

import pandas as pd
import numpy as np
from pathlib import Path

def get_rating_matrix(X):
    """Function to generate a ratings matrx and mappings for
    the user and item ids to the row and column indices

    Parameters
    ----------
    X : pandas.DataFrame, shape=(n_ratings,>=3)
        First 3 columns must be in order of user, item, rating.

    Returns
    -------
    rating_matrix : 2d numpy array, shape=(n_users, n_items)
    user_map : pandas Series, shape=(n_users,)
        Mapping from the original user id to an integer in the range [0,n_users)
    item_map : pandas Series, shape=(n_items,)
        Mapping from the original item id to an integer in the range [0,n_items)
    """

    user_col, item_col, rating_col = X.columns[:3]
    rating = X[rating_col]

    user_map = pd.Series(
        index=np.unique(X[user_col]),
        data=np.arange(X[user_col].nunique()),
        name='user_map',
    )
    item_map = pd.Series(
        index=np.unique(X[item_col]),
        data=np.arange(X[item_col].nunique()),
        name='columns_map',
    )
    user_inds = X[user_col].map(user_map)
    item_inds = X[item_col].map(item_map)
    rating_matrix = (
        pd.pivot_table(
            data=X,
            values=rating_col,
            index=user_inds,
            columns=item_inds,
        )
        .fillna(0)
        .values
    )

    # user_ratings_mean = np.mean(rating_matrix, axis=1)
    # rating_matrix = rating_matrix - user_ratings_mean.reshape(-1, 1)

    return rating_matrix, user_map, item_map


def load_dataset(dataset=Path('../data/ratings_train.dat')) -> pd.DataFrame:
    """
    Load dataset from disk by dataset path and return it as pandas DataFrame
    :param dataset: dataset Path
    :return: loaded_df
    """
    ratings_df = pd.read_csv(dataset, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return ratings_df


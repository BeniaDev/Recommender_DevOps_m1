import numpy as np
import pandas as pd


def get_R_demeaned(R):
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    return R_demeaned


def load_dataset(dataset_name: str):
    # data format in rating file: UserID::MovieID::Rating::Timestamp
    # loading the data to dataframes
    ratings_df = pd.read_csv("./data/" + dataset_name, sep='::', header=None,
                             names=['user_id', 'movie_id', 'rating', 'timestamp'])

    R_df = ratings_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

    return get_R_demeaned(R_df.values)


def create_train_test():
    """
    split into training and test sets,
    remove 10 ratings from each user
    and assign them to the test set
    """

    ratings = load_dataset("ratings_train.dat")
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_index = np.random.choice(
            np.flatnonzero(ratings[user]), size=100, replace=False)

        train[user, test_index] = 0.0
        test[user, test_index] = ratings[user, test_index]

    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return train, test
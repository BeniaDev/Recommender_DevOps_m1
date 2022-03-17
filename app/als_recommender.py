import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path

from utils import get_rating_matrix, load_dataset
import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class ALSRecommender():
    def __init__(self, k=5, lmbda=0.1, max_epochs=20, error_metric='rmse', verbose=True):
        # Force integer in case it comes in as float
        self.k = int(np.round(k))
        self.lmbda = lmbda
        self.max_epochs = max_epochs
        self.error_metric = error_metric
        self.verbose = verbose

        self.U = None
        self.I = None
        self.initialized = False

    def _calc_train_error(self, U, I, R, R_selector=None):
        if R_selector is None:
            R_selector = (R > 0)
        R_hat = np.dot(U.T, I)

        error = np.sqrt(
            np.sum(R_selector * (R_hat - R) * (R_hat - R) / np.sum(R_selector))
        )
        return error

    def _fit_init(self, X: pd.DataFrame = None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col, rating_col = X.columns[:3]
        self.train_mean = X[rating_col].mean()
        self.R, self.user_map, self.item_map = get_rating_matrix(X)
        n_users, n_items = self.R.shape
        self.U = 3 * np.random.rand(self.k, n_users)
        self.I = 3 * np.random.rand(self.k, n_items)
        self.I[0, :] = self.R[self.R != 0].mean(axis=0)  # Avg. rating for each movie
        self.E = np.eye(self.k)  # (k x k)-dimensional idendity matrix
        self.epoch = 0
        self.train_errors = []
        self.initialized = True

    def fit(self, X: pd.DataFrame, n_epochs: int = None):
        if n_epochs is None:
            self.initialized = False
        if not self.initialized:
            self._fit_init(X)

        epoch_0 = self.epoch
        if n_epochs is None:
            n_epochs = self.max_epochs - epoch_0

        n_users, n_items = self.R.shape

        # Run n_epochs iterations
        for i_epoch in range(n_epochs):
            if self.epoch >= self.max_epochs:
                print("max_epochs = {}".format(self.max_epochs))
                break
            # Fix I and estimate U
            for i, Ri in enumerate(self.R):
                nui = np.count_nonzero(Ri)  # Number of items user i has rated
                if (nui == 0): nui = 1  # Be aware of zero counts!
                # Get array of nonzero indices in row Ii
                Ri_nonzero_selector = np.nonzero(Ri)[0]
                # Select subset of I associated with movies reviewed by user i
                I_Ri = self.I[:, Ri_nonzero_selector]
                # Select subset of row R_i associated with movies reviewed by user i
                Ri_nonzero = self.R[i, Ri_nonzero_selector]
                Ai = np.dot(I_Ri, I_Ri.T) + self.lmbda * nui * self.E
                Vi = np.dot(I_Ri, Ri_nonzero.T)
                self.U[:, i] = np.linalg.solve(Ai, Vi)
            # Fix U and estimate I
            for j, Rj in enumerate(self.R.T):
                nmj = np.count_nonzero(Rj)  # Number of users that rated item j
                if (nmj == 0): nmj = 1  # Be aware of zero counts!
                # Get array of nonzero indices in row Ij
                Rj_nonzero_selector = np.nonzero(Rj)[0]
                # Select subset of P associated with users who reviewed movie j
                U_Rj = self.U[:, Rj_nonzero_selector]
                # Select subset of column R_j associated with users who reviewed movie j
                Rj_nonzero = self.R[Rj_nonzero_selector, j]
                Aj = np.dot(U_Rj, U_Rj.T) + self.lmbda * nmj * self.E
                Vj = np.dot(U_Rj, Rj_nonzero)
                self.I[:, j] = np.linalg.solve(Aj, Vj)
            error = self._calc_train_error(self.U, self.I, self.R)
            self.train_errors.append(error)
            if self.verbose:
                logging.info(f"[Epoch {self.epoch + 1}/{self.max_epochs}] train error: {error}")

            self.epoch += 1

        self.to_disk("model/")

    def train(self, dataset: Path):
        df = load_dataset(dataset=dataset)
        self.train_df_path = dataset
        self.fit(df)

    def evaluate(self, dataset: Path):
        val_df = load_dataset(dataset=dataset)
        self = self.warmup()

        preds = self.predict(val_df[['user_id', 'movie_id']])
        val_err = mean_squared_error(preds, val_df['rating'], squared=False)
        logging.info(f"Validation RMSE: {val_err}")

    def warmup(self, model_name: str='als_baseline_model.pickle'):
        with open("./model/" + model_name, 'rb') as f:
            self = pickle.load(f)

        logging.info(f"Model: {model_name} successfully loaded!")

        return self

    def to_disk(self, save_path: str):
        # Save the trained model as a pickle string.
        with open(save_path + 'als_baseline_model.pickle', 'wb') as f:
            pickle.dump(self, f)

        logging.info("Model saved ./model/als_baseline_model.pickle \n\n\n")

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        X['rating'] = 0
        known_user_and_item_mask = (
                X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index)
        )
        X_known, X_unknown = X[known_user_and_item_mask], X[~known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        rating_pred = np.array([
            np.sum(self.U[:, u_ind] * self.I[:, i_ind])
            for u_ind, i_ind in zip(user_inds, item_inds)
        ])
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        return X['rating'].values

    def recommend(self, user_id, m=10):
        self = self.warmup()
        train_df = load_dataset(self.train_df_path)
        train_df = train_df.iloc[:, :3].copy()
        train_df.columns = ['user', 'item', 'rating']

        seen_movies = train_df[train_df['user'] == user_id]['item'].unique()
        unseen_movies = list(set(train_df['item'].unique()) - set(seen_movies))
        user_movie_df = pd.DataFrame({'user': [user_id] * len(unseen_movies), 'item': unseen_movies})
        user_movie_df = user_movie_df[['user', 'item']]
        user_movie_df['pred'] = self.predict(user_movie_df)
        user_movie_df = user_movie_df.sort_values('pred', ascending=False)
        movies, preds = user_movie_df[['item', 'pred']].values[:m, :].T
        #TODO movies_id -> movie_names

        return movies, preds

    def find_similar(self):
        pass
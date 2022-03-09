import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import create_train_test
import pickle

class ALSRecommender():
    def __init__(self, n_iters=100, n_factors=40, reg=0.01):
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.reg = reg

    def train(self, dataset=None):
        train, test = create_train_test()

        self.n_user, self.n_item = train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))

        self.train_rmse_record = []
        self.test_rmse_record = []

        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors)
            preds = self.predict()
            test_mse = self.compute_rmse(test, preds)
            train_mse = self.compute_rmse(train, preds)
            self.test_rmse_record.append(test_mse)
            self.train_rmse_record.append(train_mse)

        self.to_disk("./models/")

    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)

        return solve_vecs

    def evaluate(self, dataset: str):
        pass

    def predict(self):
        return self.user_factors.dot(self.item_factors.T)

    def predict_top_m(self, dataset=None, top_M=10):
        pass

    def warmup(self, model_path: str):
        """Loads the model from $model_path or Refresh if it's already loaded"""
        pass

    def find_similar(self, movie_id: int, N=5):
        """Returns N most similar movies for input movie_id"""
        pass

    @staticmethod
    def compute_rmse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask], squared=False)
        return mse

    def to_disk(self, save_path: str):
        # Save the trained model as a pickle string.
        with open(save_path + 'als_baseline_model.pickle', 'wb') as f:
            pickle.dump(self, f)

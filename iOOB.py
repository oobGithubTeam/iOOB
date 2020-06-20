import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ImprovedOOB (BaseEnsemble, ClassifierMixin):
    def __init__(self, ensemble_of_classifiers=None, time_decay_factor=0.9):
        """Initialization."""
        self.ensemble_of_classifiers = ensemble_of_classifiers
        self.time_decay_factor = time_decay_factor
        self.q_plus = 1
        self.q_minus = 1
        #self.chunk_processing = 0

    def update_quantity(self, y):
        if y==1:
            self.q_plus = self.q_plus*self.time_decay_factor + (1 - self.time_decay_factor)
        else:
            self.q_minus = self.q_minus*self.time_decay_factor + (1 - self.time_decay_factor)


    def fit(self, X, y):
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        #self.chunk_processing = self.chunk_processing + 1
        #print(self.chunk_processing)

        X, y = check_X_y(X, y)
        self.X_, self.y_ = X, y
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        for data_point in zip(X, y):
            self.update_quantity(data_point[1])

            for M in self.ensemble_of_classifiers:
                Lambda = 0
                if (data_point[1] == 1) and (self.q_plus < self.q_minus):
                    Lambda =  self.q_minus / self.q_plus
                elif (data_point[1] == 0) and (self.q_plus > self.q_minus):
                    Lambda = self.q_plus / self.q_minus
                else:
                    Lambda = 1

                K = np.random.poisson(Lambda)

                if (K != 0):
                    M.partial_fit(self.X_, self.y_, self.classes_, sample_weight=K)
            
        return self

    def predict(self, X):
        support_matrix = np.array([M.predict_proba(X) for M in self.ensemble_of_classifiers])
        means = np.mean(support_matrix, axis=0)
        predict = np.argmax(means, axis=1)

        return predict


import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Just init the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self.__predict(x) for x in X]

        return predictions

    def __predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]

        # Sort distances
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # Majority vote
        most_common = self.__get_most_common_label(k_nearest_labels)

        return most_common

    def __get_most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]

        return most_common

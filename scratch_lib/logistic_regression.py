import numpy as np


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))

    return sig


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=100000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = X.astype(float)

        n_samples, n_features = X.shape

        # Init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Compute sigmoid of the linear regression
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Adjust weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = X.astype(float)

        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = np.array(y_pred).round().astype(int)  # Round to 0 or 1 based on 0.5 threshold

        return class_pred

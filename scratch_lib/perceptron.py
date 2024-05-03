import numpy as np


def unit_step_func(x):
    result = np.where(x >= 0, 1, 0)

    return result


# Perceptron class
# Machine learning model for binary classification
# Based on linear model and activation function symbolizing a single neuron
class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr  # The lower learning rate, the more precise improvements but longer training
        self.n_iters = n_iters
        self.activation = unit_step_func  # Function for classification
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = X.astype(float)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)  # You can also use np.random.rand(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # Perceptron training
        # Update rule number of iterations
        for _ in range(self.n_iters):
            # Iterate over all samples
            for idx, x_i in enumerate(X):
                # Calculate new output with dot product
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)

                # Apply perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        X = X.astype(float)

        # Use weights and bias to predict
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_output)

        return y_predicted

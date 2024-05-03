from scratch_lib.decision_tree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, num_trees=10, max_depth=3, min_samples_split=10):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        # Create #num_trees trees with random samples of the data
        for _ in range(self.num_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self.__bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def __bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]

        return most_common

    def predict(self, X):
        # Predict value in every tree
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Find most common value given by trees
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])

        return predictions

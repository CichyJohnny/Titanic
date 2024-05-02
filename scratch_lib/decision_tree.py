import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    # First callable function
    def fit(self, X, y):
        # Num of features can't exceed number of actual features
        self.num_features = X.shape[1] if not self.num_features else min(self.num_features, X.shape[1])

        self.root = self.__grow_tree(X, y)

    # Helper functions
    def __grow_tree(self, X, y, depth=0):
        num_samples, num_feats = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        # Stops if max depth is reached or if clean leaf-node is found or if not enough samples to split
        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self.__most_common_label(y)

            return Node(value=leaf_value)

        # Finding best split
        feature_idxs = np.random.choice(num_feats, self.num_features, replace=False)
        best_threshold, best_feature = self.__best_split(X, y, feature_idxs)

        # Create child nodes
        lefts_idxs, rights_idxs = self.__split(X[:, best_feature], best_threshold)

        if len(lefts_idxs) == 0 or len(rights_idxs) == 0:
            leaf_value = self.__most_common_label(y)

            return Node(value=leaf_value)

        left = self.__grow_tree(X[lefts_idxs, :], y[lefts_idxs], depth + 1)
        right = self.__grow_tree(X[rights_idxs, :], y[rights_idxs], depth + 1)

        return Node(best_feature, best_threshold, left=left, right=right)

    def __most_common_label(self, y):
        count = Counter(y)
        value = count.most_common(1)[0][0]

        return value

    def __best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self.__information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = thr

        return split_threshold, split_idx

    def __information_gain(self, y, X_column, thr):
        # Entropy of a parent
        parent_entropy = self.__entropy(y)

        # Create children
        left_idxs, right_idxs = self.__split(X_column, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # The weighed average of children's entropy
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self.__entropy(y[left_idxs]), self.__entropy(y[right_idxs])
        child_entropy = (n_left/n) * e_left + (n_right/n) * e_right

        # Calculate information gain
        information_gain = parent_entropy - child_entropy

        return information_gain

    def __split(self, X_column, thr):
        left_idxs = np.argwhere(X_column <= thr).flatten()
        right_idxs = np.argwhere(X_column > thr).flatten()

        return left_idxs, right_idxs

    def __entropy(self, y):
        # P(X) = #x / n
        histogram = np.bincount(y)
        px = histogram / len(y)

        # E = -sum(P(X) * log2(P(X))
        entropy = -np.sum([p * np.log2(p) for p in px if p > 0])

        return entropy

    # Second callable function
    def predict(self, X):
        predictions = np.array([self.__traverse_tree(x, self.root) for x in X])

        return predictions

    # Helper functions
    def __traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.__traverse_tree(x, node.left)
        else:
            return self.__traverse_tree(x, node.right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        self.value = value

    def is_leaf_node(self):
        return self.value is not None

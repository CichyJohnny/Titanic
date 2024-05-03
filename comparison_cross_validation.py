import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from colorama import Fore, Style

from scratch_lib.decision_tree import DecisionTree
from scratch_lib.random_forest import RandomForest
from scratch_lib.knn import KNN
from scratch_lib.logistic_regression import LogisticRegression
from scratch_lib.naive_bayes import NaiveBayes
from scratch_lib.perceptron import Perceptron

from sklearn.tree import DecisionTreeClassifier as DecisionTree_lib
from sklearn.ensemble import RandomForestClassifier as RandomForest_lib
from sklearn.neighbors import KNeighborsClassifier as KNN_lib
from sklearn.linear_model import LogisticRegression as LogisticRegression_lib
from sklearn.naive_bayes import GaussianNB as NaiveBayes_lib
from sklearn.linear_model import Perceptron as Perceptron_lib

# Compare different models' accuracy using k-fold cross-validation

# Load data
models = [DecisionTree, DecisionTree_lib, RandomForest, RandomForest_lib, KNN, KNN_lib, LogisticRegression,
          LogisticRegression_lib, NaiveBayes, NaiveBayes_lib, Perceptron, Perceptron_lib]

labels = ["Decision_Tree_Scratch", "Decision_Tree_lib", "Random_Forest_Scratch", "Random_Forest_lib", "KNN_Scratch",
          "KNN_lib", "Logistic_Regression_Scratch", "Logistic_Regression_lib", "Naive_Bayes_Scratch", "Naive_Bayes_lib",
          "Perceptron_Scratch", "Perceptron_lib"]

n_splits = 15

train_df = pd.read_csv('data/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/test.csv', index_col='PassengerId')
features = ["Pclass", "SibSp", "Sex", "Parch"]
target = 'Survived'

kf = StratifiedKFold(random_state=1234, shuffle=True, n_splits=n_splits)
compare_data = {}


def cross_validate(n_repeats=10):
    for label, model in zip(labels, models):
        accuracies = []

        for fold, (idx_tr, idx_va) in enumerate(kf.split(train_df, train_df[target])):
            X_tr = pd.get_dummies(train_df.iloc[idx_tr][features])
            X_va = pd.get_dummies(train_df.iloc[idx_va][features])
            y_tr = train_df.iloc[idx_tr][target]
            y_va = train_df.iloc[idx_va][target]

            y_pred = np.zeros_like(y_va, dtype=float)
            for i in range(n_repeats):
                m = model()
                m.fit(np.array(X_tr), np.array(y_tr))

                prd = m.predict(np.array(X_va))
                y_pred += prd

            y_pred /= n_repeats
            acc = np.mean(y_pred.round() == y_va)

            accuracies.append(acc)

        compare_data[label] = {"max": max(accuracies), "mean": np.array(accuracies).mean()}
        maxi = max(accuracies)

        print(f"{Fore.GREEN}%%%%% {label} %%%%%")
        print(f"Average: {np.array(accuracies).mean():.5f}")
        print(f"Best accuracy: {maxi:.5f}, for fold #{accuracies.index(maxi)}{Style.RESET_ALL}\n")

    max_mean, max_best = -1, -1
    label_mean, label_max = "", ""
    for label, data in compare_data.items():
        if data['mean'] > max_mean:
            max_mean = data['mean']
            label_mean = label
        if data['max'] > max_best:
            max_best = data['max']
            label_max = label

    print(f"Best average accuracy: {max_mean} for {label_mean}")
    print(f"Best accuracy: {max_best} for {label_max}")


cross_validate()

with open("comparison.json", "w") as json_file:
    json.dump(compare_data, json_file, indent=4)

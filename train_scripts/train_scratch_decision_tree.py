from scratch_lib.decision_tree import DecisionTree
import pandas as pd
import numpy as np

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# Target
labels = ["Survived"]
y_train = np.array(train_data[labels]).flatten()

# Features
features = ["Pclass", "SibSp", "Sex", "Parch"]
X_train = np.array(pd.get_dummies(train_data[features]))

# Create, fit and predict
model = DecisionTree()
model.fit(X_train, y_train)
predictions = model.predict(np.array(pd.get_dummies(test_data[features])))

# Save submission
output = pd.DataFrame({'PassengerId': pd.read_csv("../data/test.csv").PassengerId, 'Survived': predictions})
output.to_csv('../submissions/scratch_decision_tree.csv', index=False)

print("Your submission was successfully saved!")

# Best accuracy: 0.77555

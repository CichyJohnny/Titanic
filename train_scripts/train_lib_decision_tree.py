import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# Target
y = train_data["Survived"]

# Features
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Create, fit and predict
model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict(X_test)

# Save submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('../submissions/lib_decision_tree.csv', index=False)

print("Your submission was successfully saved!")

# Best accuracy: 0.76555

import pandas as pd
from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('../submissions/lib_logistic_regression.csv', index=False)

print("Your submission was successfully saved!")

# Best accuracy: 0.77511

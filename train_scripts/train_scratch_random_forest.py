from scratch_lib.random_forest import RandomForest
import pandas as pd
import numpy as np

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

features = ["Pclass", "SibSp", "Sex", "Parch"]
labels = ["Survived"]
X_train = np.array(pd.get_dummies(train_data[features]))
y_train = np.array(pd.get_dummies(train_data[labels])).flatten()

model = RandomForest(num_trees=15)
model.fit(X_train, y_train)

predictions = model.predict(np.array(pd.get_dummies(test_data[features])))

output = pd.DataFrame({'PassengerId': pd.read_csv("../data/test.csv").PassengerId, 'Survived': predictions})
output.to_csv('../submissions/scratch_random_forest.csv', index=False)

print("Your submission was successfully saved!")

# Best accuracy: 0.77272

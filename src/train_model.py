from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

project_root = Path(__file__).resolve().parents[1]

# load processed data
df = pd.read_csv(project_root / "data" / "processed" / "matches.csv")

# pick a particular feature then target another feature
df["neutral"] = df["neutral"].astype(int)
features = ["eloDiff", "formPointsDiff5", "formGoalDiff5", "restDaysDiff", "neutral"]

# drop rows with missing values in the selected features
data = df.dropna(subset=features)

X = data[features]
y = data["result"]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# make predictions and evaluate the model
predictions = model.predict(X_test)
print("rows used:", len(data))
print("accuracy:", accuracy_score(y_test, predictions))
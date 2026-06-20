
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


project_root = Path(__file__).resolve().parents[1]

# load processed data
df = pd.read_csv(project_root / "data" / "processed" / "matches.csv")

# pick a particular feature then target another feature
df["date"] = pd.to_datetime(df["date"])
df["neutral"] = df["neutral"].astype(int)

# Closeness features for draw prediction
df["absEloDiff"] = df["eloDiff"].abs()
df["absFormPointsDiff5"] = df["formPointsDiff5"].abs()
df["absFormGoalDiff5"] = df["formGoalDiff5"].abs()

numeric_features = [
    "eloDiff",
    "formPointsDiff5",
    "formGoalDiff5",
    "restDaysDiff",
    "neutral",
    "absEloDiff",
    "absFormPointsDiff5",
    "absFormGoalDiff5",
]

category_features = [
    "matchType",
]

features = numeric_features + category_features

# drop rows with missing values in the selected features
# data = df.dropna(subset=features)
#
# X = data[features]
# y = data["result"]

# split the data into training and testing sets (date split instead of random split)
data = df.dropna(subset=features).copy()
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["result"]

X_test = test_data[features]
y_test = test_data["result"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), category_features),
    ]
)

# add weights to match types
match_type_weights = {
    "friendly": 0.5,
    "regular_international": 1.0,
    "important": 1.5,
}

sample_weights = train_data["matchType"].map(match_type_weights).fillna(1.0)

# train a logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# train model with standard scale and LR
model2 = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000
    ))
])

model2.fit(X_train, y_train, model__sample_weight=sample_weights)

# # make predictions and evaluate the model
# predictions = model2.predict(X_test)
# print("rows used:", len(data))
# print("accuracy:", accuracy_score(y_test, predictions))
#
# print("\nlabels:", list(model2.classes_))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
#
# # printing coefficients learned
# trained_model = model2.named_steps["model"]
#
# coefficients = pd.DataFrame(
#     trained_model.coef_,
#     columns=features,
#     index=trained_model.classes_
# )
#
# print("\nLearned feature weights:")
# print(coefficients)
#
# sample = X_test.head(5)
# probs = model2.predict_proba(sample)
# print("\nSample predictions (columns = ", list(model2.classes_), "):")
# for row, prob in zip(sample.itertuples(), probs):
#     print(f"eloDiff={row.eloDiff} -> {prob.round(2)}")


predictions = model2.predict(X_test)
probabilities = model2.predict_proba(X_test)

print("rows used:", len(data))
print("train rows:", len(train_data))
print("test rows:", len(test_data))

print("\naccuracy:", accuracy_score(y_test, predictions))
print("log loss:", log_loss(y_test, probabilities))

print("\nActual result counts:")
print(y_test.value_counts())

print("\nPredicted result counts:")
print(pd.Series(predictions).value_counts())

print("\nlabels:", list(model2.named_steps["model"].classes_))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
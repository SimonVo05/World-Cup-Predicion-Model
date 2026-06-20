
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

# train model with standard scale and LR
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000
    ))
])

model.fit(X_train, y_train, model__sample_weight=sample_weights)


predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("rows used:", len(data))
print("train rows:", len(train_data))
print("test rows:", len(test_data))

print("\naccuracy:", accuracy_score(y_test, predictions))
print("log loss:", log_loss(y_test, probabilities))

print("\nActual result counts:")
print(y_test.value_counts())

print("\nPredicted result counts:")
print(pd.Series(predictions).value_counts())

print("\nlabels:", list(model.named_steps["model"].classes_))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))


def show_probabilities(home_team, away_team, input_row):
    one_match = pd.DataFrame([input_row])

    probs = model.predict_proba(one_match)[0]
    classes = model.named_steps["model"].classes_

    print(f"\nPrediction: {home_team} vs {away_team}")

    for label, prob in zip(classes, probs):
        print(f"{label}: {prob * 100:.1f}%")



portugal_match = {
    "eloDiff": 180,
    "formPointsDiff5": 4,
    "formGoalDiff5": 0.8,
    "restDaysDiff": 0,
    "neutral": 1,
    "absEloDiff": abs(180),
    "absFormPointsDiff5": abs(4),
    "absFormGoalDiff5": abs(0.8),
    "matchType": "important",
}

show_probabilities(
    "Portugal",
    "Uzbekistan",
    portugal_match
)
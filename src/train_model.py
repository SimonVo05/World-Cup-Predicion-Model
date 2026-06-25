from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(int)

    # Draw/closeness features
    df["absEloDiff"] = df["eloDiff"].abs()
    df["absFormPointsDiff5"] = df["formPointsDiff5"].abs()
    df["absFormGoalDiff5"] = df["formGoalDiff5"].abs()
    df["absFormPointsDiff10"] = df["formPointsDiff10"].abs()
    df["absFormGoalDiff10"] = df["formGoalDiff10"].abs()

    return df


numeric_features = [
    "eloDiff",
    "absEloDiff",

    "homeFormPoints5",
    "awayFormPoints5",
    "formPointsDiff5",
    "absFormPointsDiff5",

    "homeFormGoalDiff5",
    "awayFormGoalDiff5",
    "formGoalDiff5",
    "absFormGoalDiff5",

    "homeFormPoints10",
    "awayFormPoints10",
    "formPointsDiff10",
    "absFormPointsDiff10",

    "homeFormGoalDiff10",
    "awayFormGoalDiff10",
    "formGoalDiff10",
    "absFormGoalDiff10",

    "homeRestDays",
    "awayRestDays",
    "restDaysDiff",
    "neutral",

    "homeFc26ATK",
    "awayFc26ATK",
    "homeFc26MID",
    "awayFc26MID",
    "homeFc26DEF",
    "awayFc26DEF",
    "homeFc26GK",
    "awayFc26GK",

    "fc26Top11Diff",
    "fc26ATKDiff",
    "fc26MIDDiff",
    "fc26DEFDiff",
    "fc26GKDiff",

    "homeAttackVsAwayDefense",
    "awayAttackVsHomeDefense",
    "attackThreatDiff",

    "homeUnderdogAttackThreat",
    "awayUnderdogAttackThreat",

    "homeFc26Missing",
    "awayFc26Missing",
]

category_features = ["matchType"]
features = numeric_features + category_features


def train_model(df):
    data = df.dropna(subset=features).copy()
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

    match_type_weights = {
        "friendly": 0.5,
        "regular_international": 1.0,
        "important": 1.5,
    }

    sample_weights = train_data["matchType"].map(match_type_weights).fillna(1.0)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=5000
        ))
    ])

    model.fit(X_train, y_train, model__sample_weight=sample_weights)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy

def get_confidence_level(top_probability):
    if top_probability >= 0.65:
        return "High"
    elif top_probability >= 0.50:
        return "Medium"
    else:
        return "Low"


def readable_result(label, home_team, away_team):
    if label == "home_win":
        return f"{home_team} win"
    elif label == "away_win":
        return f"{away_team} win"
    else:
        return "Draw"

def predict_match(model, home_team, away_team, input_row):
    one_match = pd.DataFrame([input_row])

    probabilities = model.predict_proba(one_match)[0]
    classes = model.named_steps["model"].classes_

    probability_map = dict(zip(classes, probabilities))

    best_label = max(probability_map, key = probability_map.get)
    top_probability = probability_map[best_label]

    predicted_result = readable_result(best_label, home_team, away_team)
    confidence_level = get_confidence_level(top_probability)

    print(f"\nDemo prediction: {home_team} vs {away_team}")

    for label, probability in probability_map.items():
        result_name = readable_result(label, home_team, away_team)
        print(f"{result_name}: {probability * 100:.1f}%")

    print(f"\nPredicted result: {predicted_result}")
    print(f"Confidence score: {top_probability * 100:.1f}%")
    print(f"Confidence level: {confidence_level}")

def make_demo_match(df, custom_values):
    # Start with median values for every numeric feature
    demo_row = df[numeric_features].median(numeric_only=True).to_dict()

    # Default category feature
    demo_row["matchType"] = "important"

    # Replace with the custom values you care about
    demo_row.update(custom_values)

    return demo_row


def main():
    df = load_data()
    model, accuracy = train_model(df)

    print(f"Model accuracy: {accuracy:.3f}")

    portugal_match = make_demo_match(
        df,
        {
            "eloDiff": 180,
            "absEloDiff": abs(180),

            "homeFormPoints5": 10,
            "awayFormPoints5": 6,
            "formPointsDiff5": 10 - 6,
            "absFormPointsDiff5": abs(10 - 6),

            "homeFormGoalDiff5": 1.2,
            "awayFormGoalDiff5": 0.4,
            "formGoalDiff5": 1.2 - 0.4,
            "absFormGoalDiff5": abs(1.2 - 0.4),

            "homeFormPoints10": 20,
            "awayFormPoints10": 13,
            "formPointsDiff10": 20 - 13,
            "absFormPointsDiff10": abs(20 - 13),

            "homeFormGoalDiff10": 1.0,
            "awayFormGoalDiff10": 0.3,
            "formGoalDiff10": 1.0 - 0.3,
            "absFormGoalDiff10": abs(1.0 - 0.3),

            "homeRestDays": 5,
            "awayRestDays": 5,
            "restDaysDiff": 0,
            "neutral": 1,

            # Example FC26 values
            "homeFc26ATK": 84,
            "awayFc26ATK": 70,
            "homeFc26MID": 86,
            "awayFc26MID": 67,
            "homeFc26DEF": 84,
            "awayFc26DEF": 71,
            "homeFc26GK": 82,
            "awayFc26GK": 70,

            "fc26Top11Diff": 14,
            "fc26ATKDiff": 84 - 70,
            "fc26MIDDiff": 86 - 67,
            "fc26DEFDiff": 84 - 71,
            "fc26GKDiff": 82 - 70,

            "homeAttackVsAwayDefense": 84 - 71,
            "awayAttackVsHomeDefense": 70 - 84,
            "attackThreatDiff": (84 - 71) - (70 - 84),

            "homeUnderdogAttackThreat": 0,
            "awayUnderdogAttackThreat": 0,

            "homeFc26Missing": 0,
            "awayFc26Missing": 0,

            "matchType": "important",
        }
    )

    predict_match(
        model,
        "Portugal",
        "Uzbekistan",
        portugal_match,
    )


if __name__ == "__main__":
    main()
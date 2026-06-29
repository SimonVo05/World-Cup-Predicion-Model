from pathlib import Path
import unicodedata
import subprocess
import sys


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


def clean_team_name(name):
    name = str(name).strip()

    replacements = {
        "USA": "United States",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "Czech Republic": "Czechia",
        "Türkiye": "Turkey",
        "Ivory Coast": "Côte d'Ivoire",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Congo DR": "DR Congo",
        "China PR": "China",
        "Cabo Verde": "Cape Verde",
        "Viet Nam": "Vietnam",
    }

    name = replacements.get(name, name)

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = name.replace("&", "and")
    name = name.replace(".", "")
    name = name.strip()

    return name


def get_latest_team_snapshot(df, team_name, match_date):
    team_key = clean_team_name(team_name)

    df = df.copy()
    df["home_key"] = df["home_team"].apply(clean_team_name)
    df["away_key"] = df["away_team"].apply(clean_team_name)

    team_matches = df[
        (df["home_key"] == team_key) |
        (df["away_key"] == team_key)
    ].sort_values("date")

    if team_matches.empty:
        raise ValueError(f"Could not find team in matches.csv: {team_name}")

    last_match = team_matches.iloc[-1]

    if clean_team_name(last_match["home_team"]) == team_key:
        side = "home"
    else:
        side = "away"

    rest_days = (match_date - last_match["date"]).days
    rest_days = max(0, min(rest_days, 60))

    return {
        "elo": last_match[f"{side}_elo"],

        "formPoints5": last_match[f"{side}FormPoints5"],
        "formGoalDiff5": last_match[f"{side}FormGoalDiff5"],

        "formPoints10": last_match[f"{side}FormPoints10"],
        "formGoalDiff10": last_match[f"{side}FormGoalDiff10"],

        "restDays": rest_days,

        "fc26ATK": last_match[f"{side}Fc26ATK"],
        "fc26MID": last_match[f"{side}Fc26MID"],
        "fc26DEF": last_match[f"{side}Fc26DEF"],
        "fc26GK": last_match[f"{side}Fc26GK"],
        "fc26Top11Avg": last_match[f"{side}Fc26Top11Avg"],
        "fc26Missing": last_match[f"{side}Fc26Missing"],
    }


def build_match_input(
    df,
    home_team,
    away_team,
    match_type="important",
    neutral=1,
    match_date=None,
):
    if match_date is None:
        match_date = df["date"].max() + pd.Timedelta(days=1)
    else:
        match_date = pd.to_datetime(match_date)

    home = get_latest_team_snapshot(df, home_team, match_date)
    away = get_latest_team_snapshot(df, away_team, match_date)

    row = {
        "eloDiff": home["elo"] - away["elo"],
        "absEloDiff": abs(home["elo"] - away["elo"]),

        "homeFormPoints5": home["formPoints5"],
        "awayFormPoints5": away["formPoints5"],
        "formPointsDiff5": home["formPoints5"] - away["formPoints5"],
        "absFormPointsDiff5": abs(home["formPoints5"] - away["formPoints5"]),

        "homeFormGoalDiff5": home["formGoalDiff5"],
        "awayFormGoalDiff5": away["formGoalDiff5"],
        "formGoalDiff5": home["formGoalDiff5"] - away["formGoalDiff5"],
        "absFormGoalDiff5": abs(home["formGoalDiff5"] - away["formGoalDiff5"]),

        "homeFormPoints10": home["formPoints10"],
        "awayFormPoints10": away["formPoints10"],
        "formPointsDiff10": home["formPoints10"] - away["formPoints10"],
        "absFormPointsDiff10": abs(home["formPoints10"] - away["formPoints10"]),

        "homeFormGoalDiff10": home["formGoalDiff10"],
        "awayFormGoalDiff10": away["formGoalDiff10"],
        "formGoalDiff10": home["formGoalDiff10"] - away["formGoalDiff10"],
        "absFormGoalDiff10": abs(home["formGoalDiff10"] - away["formGoalDiff10"]),

        "homeRestDays": home["restDays"],
        "awayRestDays": away["restDays"],
        "restDaysDiff": home["restDays"] - away["restDays"],
        "neutral": neutral,

        "homeFc26ATK": home["fc26ATK"],
        "awayFc26ATK": away["fc26ATK"],
        "homeFc26MID": home["fc26MID"],
        "awayFc26MID": away["fc26MID"],
        "homeFc26DEF": home["fc26DEF"],
        "awayFc26DEF": away["fc26DEF"],
        "homeFc26GK": home["fc26GK"],
        "awayFc26GK": away["fc26GK"],

        "fc26Top11Diff": home["fc26Top11Avg"] - away["fc26Top11Avg"],
        "fc26ATKDiff": home["fc26ATK"] - away["fc26ATK"],
        "fc26MIDDiff": home["fc26MID"] - away["fc26MID"],
        "fc26DEFDiff": home["fc26DEF"] - away["fc26DEF"],
        "fc26GKDiff": home["fc26GK"] - away["fc26GK"],

        "homeAttackVsAwayDefense": home["fc26ATK"] - away["fc26DEF"],
        "awayAttackVsHomeDefense": away["fc26ATK"] - home["fc26DEF"],
        "attackThreatDiff": (
            home["fc26ATK"] - away["fc26DEF"]
        ) - (
            away["fc26ATK"] - home["fc26DEF"]
        ),

        "homeFc26Missing": home["fc26Missing"],
        "awayFc26Missing": away["fc26Missing"],

        "matchType": match_type,
    }

    if row["fc26Top11Diff"] < 0:
        row["homeUnderdogAttackThreat"] = row["homeAttackVsAwayDefense"]
    else:
        row["homeUnderdogAttackThreat"] = 0

    if row["fc26Top11Diff"] > 0:
        row["awayUnderdogAttackThreat"] = row["awayAttackVsHomeDefense"]
    else:
        row["awayUnderdogAttackThreat"] = 0

    # Backup fill, just in case one value is missing
    for col in numeric_features:
        if col not in row or pd.isna(row[col]):
            row[col] = df[col].median()

    return row


def rebuild_dataset():
    build_script = PROJECT_ROOT / "src" / "buildDataset.py"

    print("Rebuilding dataset...")
    subprocess.run(
        [sys.executable, str(build_script)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    rebuild_dataset()

    df = load_data()
    model, accuracy = train_model(df)

    print(f"Model accuracy: {accuracy:.3f}")

    #Predict
    home = "Japan"
    away = "Brazil"

    match_row = build_match_input(
        df,
        home_team=home,
        away_team=away,
        match_type="important",
        neutral=1,
    )

    predict_match(
        model,
        home,
        away,
        match_row,
    )


if __name__ == "__main__":
    main()
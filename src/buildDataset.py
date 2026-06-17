import pandas as pd

results = pd.read_csv("data/raw/international_matches/results.csv")

results = results.dropna(subset=["home_score", "away_score"])

def match_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

results["result"] = results.apply(match_result, axis=1)

print(results.shape)
print(results[["date", "home_team", "away_team", "home_score", "away_score", "result"]].head())
print(results["result"].value_counts())
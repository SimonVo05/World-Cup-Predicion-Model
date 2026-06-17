
# load results
from pathlib import Path

import pandas as pd


# buildDataset.py is inside src, so parents[1] is the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_csv(folder_name: str, file_name: str) -> Path:
    folder = RAW_DIR / folder_name
    matches = list(folder.rglob(file_name))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {file_name} inside:\n{folder}\n\n"
            f"Files currently inside that folder:\n"
            + "\n".join(str(path) for path in folder.rglob("*"))
        )

    return matches[0]


results_path = find_csv(
    "international_matches",
    "results.csv",
)

elo_path = find_csv(
    "international_elo",
    "eloratings.csv",
)

print("Loading results from:", results_path)
print("Loading ELO ratings from:", elo_path)

results = pd.read_csv(results_path)

# drop matches that havent play
results = results.dropna(subset=["home_score", "away_score"])

# create the target or what we want to predict
def matchResult(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

results["result"] = results.apply(matchResult, axis=1)

# print all the result out
print(results.shape)
print(results[["date", "home_team", "away_team", "home_score", "away_score", "result"]].head())
print(results["result"].value_counts())

# load elo and parse its messy dates
elo = pd.read_csv(elo_path)
elo["date"] = pd.to_datetime(elo["date"], format="mixed")

# find match dates
results["date"] = pd.to_datetime(results["date"])

# make 2 small elo tables: one for home teams and one for away teams
eloHome = elo[["date", "team", "rating"]].rename(columns={"team": "home_team", "rating": "home_elo"})
eloAway = elo[["date", "team", "rating"]].rename(columns={"team": "away_team", "rating": "away_elo"})

# sort all tables by date so we can merge them with merge_asof
results = results.sort_values("date")
eloHome = eloHome.sort_values("date")
eloAway = eloAway.sort_values("date")

# merge the elo ratings into the results table, matching on date and team, and using the most recent elo rating before the match
df = pd.merge_asof(results, eloHome, on="date", by="home_team", direction="backward")
df = pd.merge_asof(df, eloAway, on="date", by="away_team", direction="backward")

# drop matches where we dont have elo ratings for either team
df = df.dropna(subset=["home_elo", "away_elo"])

# create a new feature for the difference in elo ratings between the home and away teams
df["elo_diff"] = df["home_elo"] - df["away_elo"]

print("\nmatches with elo:", len(df))
print(df[["date", "home_team", "away_team", "home_elo", "away_elo", "elo_diff", "result"]].tail())
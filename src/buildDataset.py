
# load results
from pathlib import Path

import pandas as pd
import numpy as np


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


def add_recent_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True).copy()
    matches["match_id"] = matches.index

    # One row for each home-team performance
    home_history = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["date"],
        "team": matches["home_team"],
        "goals_for": matches["home_score"],
        "goals_against": matches["away_score"],
        "side": "home",
    })

    # One row for each away-team performance
    away_history = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["date"],
        "team": matches["away_team"],
        "goals_for": matches["away_score"],
        "goals_against": matches["home_score"],
        "side": "away",
    })

    team_history = pd.concat(
        [home_history, away_history],
        ignore_index=True,
    )

    team_history = team_history.sort_values(
        ["team", "date", "match_id"]
    ).reset_index(drop=True)

    # 3 points for a win, 1 for a draw, 0 for a loss
    team_history["points"] = np.select(
        [
            team_history["goals_for"] > team_history["goals_against"],
            team_history["goals_for"] == team_history["goals_against"],
        ],
        [3, 1],
        default=0,
    )

    team_history["goalDiff"] = (
        team_history["goals_for"]
        - team_history["goals_against"]
    )

    grouped = team_history.groupby("team", group_keys=False)

    # shift(1) prevents the current match from being included
    team_history[f"formPoints{window}"] = grouped["points"].transform(
        lambda values: (
            values.shift(1)
            .rolling(window, min_periods=window)
            .sum()
        )
    )

    team_history[f"formGoalDiff{window}"] = grouped["goalDiff"].transform(
        lambda values: (
            values.shift(1)
            .rolling(window, min_periods=window)
            .mean()
        )
    )

    # Days since the team's previous match
    team_history["restDays"] = (
        grouped["date"]
        .diff()
        .dt.days
        .clip(upper=60)
    )

    home_features = team_history[
        team_history["side"] == "home"
    ][
        [
            "match_id",
            f"formPoints{window}",
            f"formGoalDiff{window}",
            "restDays",
        ]
    ].rename(
        columns={
            f"formPoints{window}": f"homeFormPoints{window}",
            f"formGoalDiff{window}": f"homeFormGoalDiff{window}",
            "restDays": "homeRestDays",
        }
    )

    away_features = team_history[
        team_history["side"] == "away"
    ][
        [
            "match_id",
            f"formPoints{window}",
            f"formGoalDiff{window}",
            "restDays",
        ]
    ].rename(
        columns={
            f"formPoints{window}": f"awayFormPoints{window}",
            f"formGoalDiff{window}": f"awayFormGoalDiff{window}",
            "restDays": "awayRestDays",
        }
    )

    matches = matches.merge(
        home_features,
        on="match_id",
        how="left",
    )

    matches = matches.merge(
        away_features,
        on="match_id",
        how="left",
    )

    # Differences make comparison easier for the model
    matches[f"formPointsDiff{window}"] = (
        matches[f"homeFormPoints{window}"]
        - matches[f"awayFormPoints{window}"]
    )

    matches[f"formGoalDiff{window}"] = (
        matches[f"homeFormGoalDiff{window}"]
        - matches[f"awayFormGoalDiff{window}"]
    )

    matches["restDaysDiff"] = (
        matches["homeRestDays"]
        - matches["awayRestDays"]
    )

    return matches


# load elo and parse its messy dates
elo = pd.read_csv(elo_path)
elo["date"] = pd.to_datetime(elo["date"], format="mixed")

# Convert match dates
results["date"] = pd.to_datetime(results["date"])

# Add features
results = add_recent_features(results, window=5)

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
df["eloDiff"] = df["home_elo"] - df["away_elo"]

# Only matches from 2000
df = df[df["date"] >= "2000-01-01"].copy()


# Match context
important_pattern = (
    r"FIFA World Cup|World Cup qualification|"
    r"UEFA Euro|Copa América|African Cup of Nations|"
    r"AFC Asian Cup|Gold Cup|Nations League|"
    r"Confederations Cup|play[- ]?off|knockout"
)

df["matchType"] = "regular_international"

df.loc[
    df["tournament"] == "Friendly",
    "matchType"
] = "friendly"

df.loc[
    df["tournament"].str.contains(
        important_pattern,
        case=False,
        na=False
    ),
    "matchType"
] = "important"

print("\nmatches with elo:", len(df))
print(df[["date", "home_team", "away_team", "home_elo", "away_elo", "eloDiff", "result"]].tail())

# average eloDiff for each outcome
print("\naverage eloDiff for each result:")
print(df.groupby("result")["eloDiff"].mean())

df["eloBucket"] = pd.cut(df["eloDiff"], bins = [-2000, -100, -25, 25, 100, 2000], labels = ["away much stronger", "away stronger", "even", "home stronger", "home much stronger"])
print("\nHome-win rate by matchup type:")
print(df.groupby("eloBucket", observed = True)["result"].apply(lambda x: (x == "home_win").mean()))


print("\nNew model features:")

print(
    df[
        [
            "date",
            "home_team",
            "away_team",
            "eloDiff",
            "homeFormPoints5",
            "awayFormPoints5",
            "formPointsDiff5",
            "homeFormGoalDiff5",
            "awayFormGoalDiff5",
            "formGoalDiff5",
            "homeRestDays",
            "awayRestDays",
            "restDaysDiff",
            "matchType",
            "result",
        ]
    ].tail(10).to_string(index=False)
)
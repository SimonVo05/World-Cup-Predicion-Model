
# load results
from pathlib import Path
import unicodedata

import pandas as pd
import numpy as np


# buildDataset.py is inside src, so parents[1] is the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

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
    }

    name = replacements.get(name, name)

    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = name.replace("&", "and")
    name = name.replace(".", "")
    name = name.strip()

    return name


def add_fc26_features(matches):
    fc26_path = RAW_DIR / "fc26_national_team_ratings.csv"

    if not fc26_path.exists():
        raise FileNotFoundError(f"Could not find FC26 ratings file:\n{fc26_path}")

    fc26 = pd.read_csv(fc26_path)

    rating_cols = [
        "fc26ATK",
        "fc26MID",
        "fc26DEF",
        "fc26GK",
        "fc26Top11Avg",
        "fc26BestPlayer",
        "fc26PlayerCount",
    ]

    for col in rating_cols:
        fc26[col] = pd.to_numeric(fc26[col], errors="coerce")

    fc26["team_key"] = fc26["team"].apply(clean_team_name)

    matches = matches.copy()
    matches["home_team_key"] = matches["home_team"].apply(clean_team_name)
    matches["away_team_key"] = matches["away_team"].apply(clean_team_name)

    home_fc26 = fc26[["team_key"] + rating_cols].rename(
        columns={
            "team_key": "home_team_key",
            "fc26ATK": "homeFc26ATK",
            "fc26MID": "homeFc26MID",
            "fc26DEF": "homeFc26DEF",
            "fc26GK": "homeFc26GK",
            "fc26Top11Avg": "homeFc26Top11Avg",
            "fc26BestPlayer": "homeFc26BestPlayer",
            "fc26PlayerCount": "homeFc26PlayerCount",
        }
    )

    away_fc26 = fc26[["team_key"] + rating_cols].rename(
        columns={
            "team_key": "away_team_key",
            "fc26ATK": "awayFc26ATK",
            "fc26MID": "awayFc26MID",
            "fc26DEF": "awayFc26DEF",
            "fc26GK": "awayFc26GK",
            "fc26Top11Avg": "awayFc26Top11Avg",
            "fc26BestPlayer": "awayFc26BestPlayer",
            "fc26PlayerCount": "awayFc26PlayerCount",
        }
    )

    matches = matches.merge(home_fc26, on="home_team_key", how="left")
    matches = matches.merge(away_fc26, on="away_team_key", how="left")

    matches["homeFc26Missing"] = matches["homeFc26ATK"].isna().astype(int)
    matches["awayFc26Missing"] = matches["awayFc26ATK"].isna().astype(int)

    # Fill missing teams with median values so we do not delete too many matches
    medians = fc26[rating_cols].median()

    for col in rating_cols:
        home_col = "home" + col[0].upper() + col[1:]
        away_col = "away" + col[0].upper() + col[1:]

        matches[home_col] = matches[home_col].fillna(medians[col])
        matches[away_col] = matches[away_col].fillna(medians[col])

    # Overall FC26 differences
    matches["fc26Top11Diff"] = (
        matches["homeFc26Top11Avg"] - matches["awayFc26Top11Avg"]
    )

    matches["fc26ATKDiff"] = matches["homeFc26ATK"] - matches["awayFc26ATK"]
    matches["fc26MIDDiff"] = matches["homeFc26MID"] - matches["awayFc26MID"]
    matches["fc26DEFDiff"] = matches["homeFc26DEF"] - matches["awayFc26DEF"]
    matches["fc26GKDiff"] = matches["homeFc26GK"] - matches["awayFc26GK"]

    # Attack vs defense matchup
    matches["homeAttackVsAwayDefense"] = (
        matches["homeFc26ATK"] - matches["awayFc26DEF"]
    )

    matches["awayAttackVsHomeDefense"] = (
        matches["awayFc26ATK"] - matches["homeFc26DEF"]
    )

    matches["attackThreatDiff"] = (
        matches["homeAttackVsAwayDefense"]
        - matches["awayAttackVsHomeDefense"]
    )

    # Underdog attacking threat
    matches["homeUnderdogAttackThreat"] = np.where(
        matches["fc26Top11Diff"] < 0,
        matches["homeAttackVsAwayDefense"],
        0,
    )

    matches["awayUnderdogAttackThreat"] = np.where(
        matches["fc26Top11Diff"] > 0,
        matches["awayAttackVsHomeDefense"],
        0,
    )

    print("\nFC26 ratings added.")
    print("Home teams missing FC26:", matches["homeFc26Missing"].sum())
    print("Away teams missing FC26:", matches["awayFc26Missing"].sum())

    return matches


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

    matches = matches.drop(
        columns=["homeRestDays", "awayRestDays", "restDaysDiff"],
        errors="ignore"
    )

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


def add_elo_ratings(matches, k=30, home_advantage=65, start=1500.0):
    matches = matches.sort_values("date").reset_index(drop=True).copy()
    ratings = {}                      
    home_elos, away_elos = [], []

    big_tournaments = ("world cup", "uefa euro", "copa américa", "copa america", "african cup", "asian cup", "gold cup", "nations league", "confederations cup")

    for row in matches.itertuples():
        rh = ratings.get(row.home_team, start)
        ra = ratings.get(row.away_team, start)

        # record PRE-match ratings (never peek at the result we're predicting)
        home_elos.append(rh)
        away_elos.append(ra)

        # home advantage applies only when not on neutral ground
        adv = 0 if row.neutral else home_advantage
        expected_home = 1 / (1 + 10 ** ((ra - (rh + adv)) / 400))

        # actual outcome from the home team's perspective
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        
        # bigger wins move ratings more
        margin = abs(row.home_score - row.away_score)
        if margin <= 1:
            goal_multiplier = 1.0
        elif margin == 2:
            goal_multiplier = 1.5
        else:
            goal_multiplier = (11 + margin) / 8
        
        # important matches count for more
        tournament = str(row.tournament).lower()
        if tournament == "friendly":
            importance = 0.7
        elif any(name in tournament for name in big_tournaments):
            importance = 1.6
        else:
            importance = 1.0

        # shift ratings by K x (what happened - what was expected); zero-sum
        change = k * goal_multiplier * importance * (actual_home - expected_home)
        ratings[row.home_team] = rh + change
        ratings[row.away_team] = ra - change

    matches["home_elo"] = home_elos
    matches["away_elo"] = away_elos
    matches["eloDiff"] = matches["home_elo"] - matches["away_elo"]
    return matches


# Convert match dates
results["date"] = pd.to_datetime(results["date"])

# Add recent-form features (computed over the full history)
results = add_recent_features(results, window=5)
results = add_recent_features(results, window=10)

results = add_elo_ratings(results)

df = results

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
            "homeFormPoints10",
            "awayFormPoints10",
            "formPointsDiff10",
            "homeFormGoalDiff10",
            "awayFormGoalDiff10",
            "formGoalDiff10",
        ]
    ].tail(10).to_string(index=False)
)

df = add_fc26_features(df)

processed_dir = PROJECT_ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_dir / "matches.csv", index=False)

print("\nsaved processed data to", processed_dir / "matches.csv")
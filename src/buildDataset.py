import pandas as pd

# load results
results = pd.read_csv("data/raw/international_matches/results.csv")

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
elo = pd.read_csv("data/raw/international_elo/eloratings.csv")
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
df["eloDiff"] = df["home_elo"] - df["away_elo"]

print("\nmatches with elo:", len(df))
print(df[["date", "home_team", "away_team", "home_elo", "away_elo", "eloDiff", "result"]].tail())

# average eloDiff for each outcome
print("\naverage eloDiff for each result:")
print(df.groupby("result")["eloDiff"].mean())

df["eloBucket"] = pd.cut(df["eloDiff"], bins = [-2000, -100, -25, 25, 100, 2000], labels = ["away much stronger", "away stronger", "even", "home stronger", "home much stronger"])
print("\nHome-win rate by matchup type:")
print(df.groupby("eloBucket", observed = True)["result"].apply(lambda x: (x == "home_win").mean()))
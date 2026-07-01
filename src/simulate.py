from pathlib import Path
from collections import Counter
import glob
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# WC 2026 team names that are spelled differently in results.csv (our Elo source)
NAME_FIX = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "USA": "United States",
}


def _find(filename: str) -> str:
    #Find a CSV in data/raw (preferred) or, as a fallback, the kagglehub cache.
    roots = [PROJECT_ROOT / "data" / "raw", Path.home() / ".cache" / "kagglehub"]
    for root in roots:
        hits = sorted(glob.glob(str(root / "**" / filename), recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        f"Could not find {filename} in data/raw or the kagglehub cache. "
        f"Did you run download_data.py?"
    )


def load_current_elo() -> dict:
    #Load the most recent Elo rating for each team from matches.csv.
    matches = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "matches.csv",
        parse_dates=["date"],
    )

    # stack home rows and away rows into one long (team, date, elo) table
    home = matches[["date", "home_team", "home_elo"]].rename(
        columns={"home_team": "team", "home_elo": "elo"}
    )
    away = matches[["date", "away_team", "away_elo"]].rename(
        columns={"away_team": "team", "away_elo": "elo"}
    )
    long = pd.concat([home, away], ignore_index=True).sort_values("date")

    # the last (most recent) Elo we have for each team
    return long.groupby("team")["elo"].last().to_dict()


def load_teams() -> pd.DataFrame:
    #The 48 WC 2026 teams, with names normalized to results.csv spelling.
    teams = pd.read_csv(_find("wc_2026_teams.csv"))
    teams["team"] = teams["team"].replace(NAME_FIX)
    return teams


def train_match_model() -> LogisticRegression:
    # Lightweight model: predict the outcome from the Elo gap alone.
    # eloDiff carries ~95% of the signal and is the only feature we can know
    # for any hypothetical matchup. We train on NEUTRAL matches so there's no
    # home-crowd bias - World Cup games are neutral.
    matches = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "matches.csv")
    neutral = matches[matches["neutral"] == True]
    model = LogisticRegression(max_iter=1000)
    model.fit(neutral[["eloDiff"]], neutral["result"])
    return model


def predict_match(model, elo, team_a, team_b) -> dict:
    # Probabilities for a neutral-venue match: {team_a win, draw, team_b win}.
    # We predict both orderings and average them, which cancels any leftover
    # "home/away label" bias so the result is symmetric and fair.
    classes = list(model.classes_)  # ['away_win', 'draw', 'home_win']

    def probs(diff):
        row = pd.DataFrame({"eloDiff": [diff]})
        return dict(zip(classes, model.predict_proba(row)[0]))

    p1 = probs(elo[team_a] - elo[team_b])   # team_a treated as home
    p2 = probs(elo[team_b] - elo[team_a])   # team_b treated as home

    a_win = (p1["home_win"] + p2["away_win"]) / 2
    b_win = (p1["away_win"] + p2["home_win"]) / 2
    draw = (p1["draw"] + p2["draw"]) / 2

    total = a_win + draw + b_win
    return {
        "team_a_win": a_win / total,
        "draw": draw / total,
        "team_b_win": b_win / total,
    }

#  Tournament simulation
def make_prob_function(model):
    # Fast eloDiff -> {away_win, draw, home_win} using the trained logistic
    # weights directly (much faster than model.predict_proba in a big loop).
    classes = list(model.classes_)
    coef = model.coef_.flatten()
    intercept = np.asarray(model.intercept_)

    def prob(elo_diff):
        z = coef * elo_diff + intercept
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        return dict(zip(classes, p))

    return prob


def precompute_matrix(prob_fn, elo, teams) -> dict:
    # For every ordered pair, the symmetric neutral-venue probabilities
    # (team_a win, draw, team_b win). Precomputing makes the sim fast.
    matrix = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            p1 = prob_fn(elo[a] - elo[b])
            p2 = prob_fn(elo[b] - elo[a])
            a_win = (p1["home_win"] + p2["away_win"]) / 2
            b_win = (p1["away_win"] + p2["home_win"]) / 2
            draw = (p1["draw"] + p2["draw"]) / 2
            t = a_win + draw + b_win
            matrix[(a, b)] = (a_win / t, draw / t, b_win / t)
    return matrix


def load_groups(teams_df, elo) -> dict:
    # {group_letter: [team, team, team, team]} for teams that have an Elo.
    groups = {}
    for letter, sub in teams_df.groupby("group"):
        groups[letter] = [t for t in sub["team"] if t in elo]
    return groups


def simulate_group(matrix, teams, rng) -> list:
    # Round-robin: every team plays every other once. Returns [(team, points)]
    # ranked best-first (points, then a random tiebreak).
    points = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            p_a, p_draw, _ = matrix[(a, b)]
            r = rng.random()
            if r < p_a:
                points[a] += 3
            elif r < p_a + p_draw:
                points[a] += 1
                points[b] += 1
            else:
                points[b] += 3
    ranked = sorted(teams, key=lambda t: (points[t], rng.random()), reverse=True)
    return [(t, points[t]) for t in ranked]


def simulate_knockouts(matrix, qualifiers, rng) -> str:
    # Single elimination. Draws can't stand in knockouts, so we drop the draw
    # probability and split it by relative strength (penalty-shootout proxy).
    bracket = qualifiers[:]
    rng.shuffle(bracket)
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            p_a, _, p_b = matrix[(a, b)]
            next_round.append(a if rng.random() < p_a / (p_a + p_b) else b)
        bracket = next_round
    return bracket[0]


def run_tournament(matrix, groups, n=10000, seed=0) -> pd.DataFrame:
    # Play the whole World Cup n times; return each team's championship odds.
    rng = random.Random(seed)
    champions = Counter()

    for _ in range(n):
        firsts_seconds, thirds = [], []
        for teams in groups.values():
            ranked = simulate_group(matrix, teams, rng)
            firsts_seconds.append(ranked[0][0])
            firsts_seconds.append(ranked[1][0])
            thirds.append(ranked[2])                       # (team, points)

        # 48-team format: 12 group winners + 12 runners-up + 8 best 3rd-placed
        best_thirds = [t for t, _ in sorted(
            thirds, key=lambda x: (x[1], rng.random()), reverse=True)[:8]]
        qualifiers = firsts_seconds + best_thirds          # 32 teams

        champions[simulate_knockouts(matrix, qualifiers, rng)] += 1

    odds = (pd.DataFrame({"team": list(champions), "champion_pct": list(champions.values())})
            .assign(champion_pct=lambda d: d["champion_pct"] / n)
            .sort_values("champion_pct", ascending=False)
            .reset_index(drop=True))
    return odds


def main() -> None:
    teams = load_teams()
    elo = load_current_elo()

    print(f"Loaded {len(teams)} World Cup 2026 teams.\n")

    resolved, missing = [], []
    for _, row in teams.iterrows():
        name = row["team"]
        if name in elo:
            resolved.append((row["group"], name, round(elo[name])))
        else:
            missing.append(name)

    # strongest first
    resolved.sort(key=lambda r: r[2], reverse=True)
    print("Current Elo ratings (strongest first):")
    for group, name, rating in resolved:
        print(f"  [{group}] {name:20} {rating}")

    print()
    if missing:
        print(f"{len(missing)} team(s) have NO Elo (fix NAME_FIX): {missing} :(")
    else:
        print(f"All {len(teams)} teams resolved to an Elo rating. :)")

    # --- demo: predict a few matchups ---
    model = train_match_model()
    print("\nSample match predictions (neutral venue):")
    demo = [("Spain", "Brazil"), ("Argentina", "France"),
            ("England", "United States"), ("Brazil", "Morocco")]
    for a, b in demo:
        p = predict_match(model, elo, a, b)
        print(f"  {a} vs {b}:  {a} {p['team_a_win']:.0%} | "
              f"draw {p['draw']:.0%} | {b} {p['team_b_win']:.0%}")


if __name__ == "__main__":
    main()

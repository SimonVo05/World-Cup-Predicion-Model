import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# let this file import simulate.py (same folder)
sys.path.append(str(Path(__file__).resolve().parent))
from simulate import (
    load_teams,
    load_current_elo,
    train_match_model,
    predict_match,
)


st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")


# cache so we don't reload/retrain on every click
@st.cache_data
def get_data():
    return load_teams(), load_current_elo()


@st.cache_resource
def get_model():
    return train_match_model()


@st.cache_data
def load_odds():
    path = PROJECT_ROOT / "data" / "processed" / "championship_odds.csv"
    return pd.read_csv(path) if path.exists() else None


teams_df, elo = get_data()
model = get_model()

# only the 48 WC teams that resolve to an Elo
wc_teams = sorted(t for t in teams_df["team"] if t in elo)

st.title("World Cup 2026 Predictor")
st.caption(
    "Match-outcome probabilities from an Elo-based model trained on ~25,000 "
    "international matches. Probabilities are odds, not certainties."
)

# ---------------- Championship odds ----------------
st.header("Who wins the World Cup?")

odds = load_odds()
if odds is None:
    st.info("Run `python src/run_simulation.py` to generate championship odds.")
else:
    top_n = st.slider("Teams to show", 5, len(odds), 15)
    data = odds.head(top_n).copy()
    data["label"] = (data["champion_pct"] * 100).round(1).astype(str) + "%"

    bars = (
        alt.Chart(data)
        .mark_bar(color="#1a7f37")
        .encode(
            x=alt.X("champion_pct:Q", title="Chance to win the tournament",
                    axis=alt.Axis(format="%")),
            y=alt.Y("team:N", sort="-x", title=None),
            tooltip=["team", alt.Tooltip("champion_pct:Q", format=".1%")],
        )
    )
    labels = bars.mark_text(align="left", dx=3, color="#333").encode(text="label:N")
    st.altair_chart(bars + labels, use_container_width=True)
    st.caption("Based on 10,000 simulated tournaments. "
               "Re-run src/run_simulation.py after changing the model to refresh.")

# ---------------- Match predictor ----------------
st.header("Match predictor")

col_a, col_b = st.columns(2)
default_a = wc_teams.index("Spain") if "Spain" in wc_teams else 0
default_b = wc_teams.index("Brazil") if "Brazil" in wc_teams else 1
team_a = col_a.selectbox("Team A", wc_teams, index=default_a)
team_b = col_b.selectbox("Team B", wc_teams, index=default_b)

if team_a == team_b:
    st.warning("Pick two different teams.")
else:
    p = predict_match(model, elo, team_a, team_b)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{team_a} win", f"{p['team_a_win']:.0%}")
    m2.metric("Draw", f"{p['draw']:.0%}")
    m3.metric(f"{team_b} win", f"{p['team_b_win']:.0%}")

    chart = pd.DataFrame(
        {"probability": [p["team_a_win"], p["draw"], p["team_b_win"]]},
        index=[f"{team_a} win", "Draw", f"{team_b} win"],
    )
    st.bar_chart(chart)

    gap = round(elo[team_a] - elo[team_b])
    st.caption(
        f"Elo — {team_a}: {round(elo[team_a])}  ·  {team_b}: {round(elo[team_b])}  "
        f"(gap {gap:+d})"
    )

# ---------------- Team ratings ----------------
st.header("Team ratings (all 48 teams)")

table = teams_df[teams_df["team"].isin(elo)].copy()
table["elo"] = table["team"].map(elo).round().astype(int)
cols = [c for c in ["team", "group", "confederation", "fifa_rank", "elo"] if c in table.columns]
table = table.sort_values("elo", ascending=False)[cols].reset_index(drop=True)
table.index = table.index + 1  # rank starts at 1

st.dataframe(table, use_container_width=True)

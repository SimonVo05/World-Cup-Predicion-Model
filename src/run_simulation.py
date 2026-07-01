from pathlib import Path

from simulate import (
    load_teams,
    load_current_elo,
    train_match_model,
    make_prob_function,
    precompute_matrix,
    load_groups,
    run_tournament,
)

N_SIMS = 10000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    teams_df = load_teams()
    elo = load_current_elo()
    model = train_match_model()

    # fast probability lookup for every WC matchup
    wc_teams = [t for t in teams_df["team"] if t in elo]
    prob_fn = make_prob_function(model)
    matrix = precompute_matrix(prob_fn, elo, wc_teams)
    groups = load_groups(teams_df, elo)

    print(f"Simulating the World Cup {N_SIMS:,} times...")
    odds = run_tournament(matrix, groups, n=N_SIMS)

    out = PROJECT_ROOT / "data" / "processed" / "championship_odds.csv"
    odds.to_csv(out, index=False)

    print("\nChampionship odds (top 15):")
    for _, row in odds.head(15).iterrows():
        print(f"  {row['team']:20} {row['champion_pct']:.1%}")
    print(f"\nSaved full table to {out}")


if __name__ == "__main__":
    main()

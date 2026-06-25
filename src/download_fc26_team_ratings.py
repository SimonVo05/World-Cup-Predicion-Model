from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FC26_DIR = RAW_DIR / "fc26_players"

OUTPUT_PATH = RAW_DIR / "fc26_national_team_ratings.csv"


def find_fc26_csv():
    csv_files = list(FC26_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found inside {FC26_DIR}")

    print("Found CSV files:")
    for file in csv_files:
        print(file)

    return csv_files[0]


def pick_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns are:\n{list(df.columns)}"
    )


def classify_position(position_text):
    if pd.isna(position_text):
        return "unknown"

    # Use the first listed position as the main position
    main_position = str(position_text).split(",")[0].strip().upper()

    attackers = {"ST", "CF", "LW", "RW", "LF", "RF"}
    midfielders = {"CAM", "CM", "CDM", "LM", "RM"}
    defenders = {"CB", "LB", "RB", "LWB", "RWB"}
    goalkeepers = {"GK"}

    if main_position in attackers:
        return "ATK"
    elif main_position in midfielders:
        return "MID"
    elif main_position in defenders:
        return "DEF"
    elif main_position in goalkeepers:
        return "GK"
    else:
        return "unknown"


def top_average(df, country_col, rating_col, group_name, top_n, output_name):
    group_df = df[df["positionGroup"] == group_name].copy()

    return (
        group_df.sort_values(rating_col, ascending=False)
        .groupby(country_col)
        .head(top_n)
        .groupby(country_col)[rating_col]
        .mean()
        .reset_index()
        .rename(columns={rating_col: output_name})
    )


def main():
    csv_path = find_fc26_csv()
    df = pd.read_csv(csv_path, low_memory=False)

    print("\nUsing file:")
    print(csv_path)

    print("\nColumns:")
    print(list(df.columns))

    country_col = pick_column(
        df,
        [
            "nationality_name",
            "nationality",
            "nation",
            "country",
        ],
    )

    rating_col = pick_column(
        df,
        [
            "overall",
            "OVR",
            "rating",
        ],
    )

    position_col = pick_column(
        df,
        [
            "player_positions",
            "positions",
            "position",
            "player_position",
        ],
    )

    df = df.dropna(subset=[country_col, rating_col, position_col]).copy()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])

    df["positionGroup"] = df[position_col].apply(classify_position)

    # Use best players by unit
    atk_rating = top_average(
        df,
        country_col,
        rating_col,
        group_name="ATK",
        top_n=4,
        output_name="fc26ATK",
    )

    mid_rating = top_average(
        df,
        country_col,
        rating_col,
        group_name="MID",
        top_n=4,
        output_name="fc26MID",
    )

    def_rating = top_average(
        df,
        country_col,
        rating_col,
        group_name="DEF",
        top_n=5,
        output_name="fc26DEF",
    )

    gk_rating = top_average(
        df,
        country_col,
        rating_col,
        group_name="GK",
        top_n=1,
        output_name="fc26GK",
    )

    top_11 = (
        df.sort_values(rating_col, ascending=False)
        .groupby(country_col)
        .head(11)
        .groupby(country_col)[rating_col]
        .mean()
        .reset_index()
        .rename(columns={rating_col: "fc26Top11Avg"})
    )

    best_player = (
        df.groupby(country_col)[rating_col]
        .max()
        .reset_index()
        .rename(columns={rating_col: "fc26BestPlayer"})
    )

    player_count = (
        df.groupby(country_col)[rating_col]
        .count()
        .reset_index()
        .rename(columns={rating_col: "fc26PlayerCount"})
    )

    ratings = atk_rating.merge(mid_rating, on=country_col, how="outer")
    ratings = ratings.merge(def_rating, on=country_col, how="outer")
    ratings = ratings.merge(gk_rating, on=country_col, how="outer")
    ratings = ratings.merge(top_11, on=country_col, how="outer")
    ratings = ratings.merge(best_player, on=country_col, how="outer")
    ratings = ratings.merge(player_count, on=country_col, how="outer")

    ratings = ratings.rename(columns={country_col: "team"})

    ratings = ratings[
        [
            "team",
            "fc26ATK",
            "fc26MID",
            "fc26DEF",
            "fc26GK",
            "fc26Top11Avg",
            "fc26BestPlayer",
            "fc26PlayerCount",
        ]
    ]

    ratings = ratings.sort_values("fc26Top11Avg", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved FC26 national team ratings to:")
    print(OUTPUT_PATH)

    print("\nTop teams:")
    print(ratings.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
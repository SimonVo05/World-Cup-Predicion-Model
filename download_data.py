from pathlib import Path

import kagglehub


DATASETS = {
    "international_matches":
        "martj42/international-football-results-from-1872-to-2017",

    "fifa_rankings":
        "lucasyukioimafuko/fifa-mens-world-ranking",

    "international_elo":
        "saifalnimri/international-football-elo-ratings",

    "fc26_players":
        "rovnez/fc-26-fifa-26-player-data",

    "world_cup_2026":
        "mominullptr/fifa-world-cup-2026-dataset",
}


def download_datasets() -> None:
    # download_data.py is currently in the project root
    project_root = Path(__file__).resolve().parent

    raw_directory = project_root / "data" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    print(f"Saving datasets inside:\n{raw_directory}\n")

    for name, dataset_id in DATASETS.items():
        dataset_directory = raw_directory / name
        dataset_directory.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {name}...")

        downloaded_path = kagglehub.dataset_download(
            dataset_id,
            output_dir=str(dataset_directory),
            force_download=True,
        )

        print(f"Saved {name} to:")
        print(f"{downloaded_path}\n")


if __name__ == "__main__":
    download_datasets()
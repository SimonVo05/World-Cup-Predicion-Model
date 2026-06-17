from pathlib import Path

import shutil # simplifying tasks like copying and removing directories in automation scripts
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
}


def download_datasets() -> None:
    output_directory = Path("data/raw")
    output_directory.mkdir(parents=True, exist_ok=True)

    for name, dataset_id in DATASETS.items():
        print(f"Downloading {name}...")

        path = kagglehub.dataset_download(dataset_id)

        destination = output_directory / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(path, destination)

        print(f"{name} downloaded to: {destination}")


if __name__ == "__main__":
    download_datasets()
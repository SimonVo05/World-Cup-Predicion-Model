from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "matches.csv"


def file_age_hours(path):
    if not path.exists():
        return 999999

    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds / 3600


def run_script(script_path):
    print(f"\nRunning {script_path.name}...")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def update_data_if_needed(max_age_hours=24, force=False):
    data_is_old = file_age_hours(PROCESSED_DATA) > max_age_hours

    if not force and PROCESSED_DATA.exists() and not data_is_old:
        print("Processed data is already fresh.")
        print("Using:", PROCESSED_DATA)
        return

    print("Updating dataset...")

    # 1. Download/update Kaggle raw datasets
    download_script = PROJECT_ROOT / "download_data.py"
    if download_script.exists():
        run_script(download_script)

    # 2. Recreate FC26 national team ratings
    fc26_script = SRC_DIR / "create_fc26_national_team_ratings.py"
    if fc26_script.exists():
        run_script(fc26_script)

    # 3. Rebuild processed matches.csv
    build_script = SRC_DIR / "buildDataset.py"
    run_script(build_script)

    print("\nDataset update finished.")


if __name__ == "__main__":
    update_data_if_needed(max_age_hours=24, force=True)
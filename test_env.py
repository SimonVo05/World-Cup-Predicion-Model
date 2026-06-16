import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY")

if not api_key:
    raise RuntimeError(
        "API_FOOTBALL_KEY was not found. Add it to your local .env file."
    )
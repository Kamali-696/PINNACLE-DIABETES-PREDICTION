from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DPS_", env_file=".env", extra="ignore")

    # Fixed run directory containing the independently trained local EHR and Pima models.
    model_run_dir: Path = PROJECT_DIR / "local_model_results" / "run_20261024T003644Z"
    processed_dir: Path = PROJECT_DIR / "processed"

    # Origins allowed to call the API (the Vite dev server by default).
    cors_origins: list[str] = ["http://localhost:5175", "http://127.0.0.1:5175"]


settings = Settings()

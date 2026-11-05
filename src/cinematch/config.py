"""Configuration. CineMatch | Roll Number: 2025AG05135."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def database_url() -> str:
    """Return the configured PostgreSQL connection URL; fail fast if it is missing."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url or "YOUR_USERNAME" in url:
        raise RuntimeError("Set a valid DATABASE_URL in .env before database commands.")
    return url


def source_dir(override: str | None = None) -> Path:
    """Resolve the MovieLens source directory from CLI override, environment or default."""
    value = override or os.getenv("MOVIELENS_DIR")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "data" / "raw" / "ml-32m"

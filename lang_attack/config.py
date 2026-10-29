"""Configuration settings for LangAttack."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = REPO_ROOT / "database"
LOGS_DIR = REPO_ROOT / "logs"
NOTES_DIR = REPO_ROOT / "notes"
DEFAULT_DB_PATH = DATABASE_DIR / "lang_attack.db"

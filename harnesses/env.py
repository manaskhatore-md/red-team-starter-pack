"""Where the kit's settings come from: a .env file, the environment, or both.

Every entry point calls load_settings() first, before anything reads a setting. A value
already in the environment always wins over the file: load_dotenv() never replaces one,
which is also what lets rt's --provider and a CI job's env: block beat .env.

RT_ENV_FILE picks the file, and is itself only ever read from the environment:

    unset            .env in the repo root if there is one, else environment variables
                     only (and a note saying so)
    a path           that file; stops if it is missing, rather than run on other settings
    none             no file: environment variables only, even if a .env exists. For CI,
                     containers, and Cloud Run, where settings are set as variables
"""

import os
import sys
from pathlib import Path

import dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
NO_FILE = {"none", "off", "0", "false", "no"}

_loaded: str | None = None


def load_settings() -> str:
    """Load the settings file, if there is one. Returns where settings came from, for messages.

    Runs once per process; later calls return the first answer.
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    choice = os.getenv("RT_ENV_FILE", "").strip()
    if choice.lower() in NO_FILE:
        _loaded = "environment variables (RT_ENV_FILE=none)"
    elif choice:
        path = Path(choice).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"RT_ENV_FILE={choice} is not a file. Point it at your settings file, "
                "or set RT_ENV_FILE=none to use environment variables only."
            )
        dotenv.load_dotenv(path)
        _loaded = f"{path} and environment variables"
    elif DEFAULT_ENV_FILE.is_file():
        dotenv.load_dotenv(DEFAULT_ENV_FILE)
        _loaded = ".env and environment variables"
    else:
        _loaded = "environment variables (no .env)"
        print(
            "No .env found: reading settings from environment variables only. "
            "Copy .env.example to .env, or set RT_ENV_FILE to use another file.",
            file=sys.stderr,
        )
    return _loaded

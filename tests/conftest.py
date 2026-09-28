"""Shared setup for the offline test suite.

Everything here runs without network access, API keys, or cloud credentials, so
the suite is safe to run anywhere - including CI with no secrets configured.

    pip install -r requirements-dev.txt
    python -m pytest

Three things keep it offline:
  - load_dotenv() is stubbed out BEFORE any repo module is imported. Every entry
    point calls it at import time, and it would otherwise pull your real .env (API
    keys included) into the test process.
  - Every RT_* setting and provider key is removed from the environment for each
    test, so a test sees only what it sets. Keys it does set are fake.
  - PyRIT memory is IN_MEMORY, so nothing is written to ~/.pyrit/dbdata/.

KNOWN BUGS are written as strict xfail tests: they fail today, and pytest reports
them as "xfailed". When a fix lands, the test starts passing, strict mode turns
that into a failure, and whoever fixed it removes the marker. A bug cannot be
fixed - or come back - without the suite noticing.
"""

import asyncio
import os
import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

# Run from anywhere: make the repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from pyrit.memory import CentralMemory  # noqa: E402
from pyrit.setup import IN_MEMORY, initialize_pyrit_async  # noqa: E402

PROVIDER_KEYS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LITELLM_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "PROJECT_ID",
    "GOOGLE_VERTEX_REGION",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip every setting a test could inherit from the shell running it."""
    for name in list(os.environ):
        if name.startswith("RT_") or name in PROVIDER_KEYS:
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def fake_keys(monkeypatch):
    """Fake API keys for every key-based provider. Never valid; never sent anywhere."""
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, f"fake-{name.lower()}")


@pytest.fixture
def memory():
    """A fresh, empty in-memory PyRIT database for one test."""
    asyncio.run(initialize_pyrit_async(memory_db_type=IN_MEMORY, env_files=[], silent=True))
    memory = CentralMemory.get_memory_instance()
    # Re-initializing can hand back the same in-memory database, so empty it: a
    # test must not see another test's results.
    memory.reset_database()
    return memory

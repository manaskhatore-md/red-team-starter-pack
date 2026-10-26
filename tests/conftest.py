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
  - PyRIT memory is IN_MEMORY, so nothing is written to your real results database.

KNOWN BUGS are written as strict xfail tests: they fail today, and pytest reports
them as "xfailed". When a fix lands, the test starts passing, strict mode turns
that into a failure, and whoever fixed it removes the marker. A bug cannot be
fixed - or come back - without the suite noticing.

HOUSE STYLE tests (marked house_style) hold the shipped rubrics to the pack's own
wording, which the kit does not rely on. They always run. A fork that rewords its
rubrics turns them off with -m "not house_style" (see pytest.ini); rubrics.check
still holds its rubrics to what the kit does rely on.

Shared fixtures are here; shared fakes and helpers that tests import by name are in
tests/helpers.py.
"""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False

# Run from anywhere: make the repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from pyrit.memory import CentralMemory  # noqa: E402
from pyrit.memory.memory_models import Base  # noqa: E402
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

HOUSE_STYLE_NOTE = (
    "This test checks the pack's own rubric wording, which the kit does not rely on. "
    "If you reworded your rubrics on purpose, turn these tests off: add -m \"not house_style\" "
    "to addopts in pytest.ini. Then run python -m harnesses.pyrit_campaigns.rubrics check "
    "for what the kit does rely on."
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Say, under a failed house_style test, that a fork can turn these off and how."""
    report = (yield).get_result()
    if report.failed and item.get_closest_marker("house_style"):
        report.sections.append(("house style", HOUSE_STYLE_NOTE))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip every setting a test could inherit from the shell running it."""
    for name in list(os.environ):
        if name.startswith(("RT_", "APP_")) or name in PROVIDER_KEYS:
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
    # test must not see another test's results. Deleting the rows leaves the same empty
    # database as memory.reset_database(), which reruns the schema migration: 0.2s a
    # test, against a few milliseconds.
    session = memory.get_session()
    try:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    finally:
        session.close()
    return memory


@pytest.fixture
def fake_app(clean_env, memory):
    """A local chat app: records each request, and answers with a status and body you set.

    Returns the server's state dict. Set state["replies"] to a list of (status, body)
    to answer requests in turn; the last one repeats.
    """
    state = {"replies": [(200, json.dumps({"answer": "hello from the app"}))], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            state["requests"].append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, reply = state["replies"][min(len(state["requests"]), len(state["replies"])) - 1]
            reply = reply.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_ENDPOINT", f"http://127.0.0.1:{server.server_port}/test/chat")
    clean_env.setenv("APP_REQUEST_TEMPLATE", '{"query": "{PROMPT}", "options": ["{PROMPT}"]}')
    clean_env.setenv("APP_RESPONSE_PATH", "answer")
    clean_env.setenv("RETRY_MAX_NUM_ATTEMPTS", "2")
    clean_env.setenv("RETRY_WAIT_MIN_SECONDS", "0")
    clean_env.setenv("RETRY_WAIT_MAX_SECONDS", "0")
    yield state
    server.shutdown()

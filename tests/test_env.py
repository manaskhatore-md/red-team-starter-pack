"""Where settings come from: .env, another file named by RT_ENV_FILE, or only the environment."""

import dotenv.main
import pytest

from harnesses import env

# Settings the files below define. Each test starts and ends with none of them set.
NAMES = ("RT_ENV_TEST_A", "RT_ENV_TEST_B")


@pytest.fixture
def settings(clean_env, monkeypatch, tmp_path):
    """A fresh load_settings(): the real load_dotenv (conftest stubs it), and a .env path in tmp_path."""
    monkeypatch.setattr(dotenv, "load_dotenv", dotenv.main.load_dotenv)
    monkeypatch.setattr(env, "_loaded", None)
    monkeypatch.setattr(env, "DEFAULT_ENV_FILE", tmp_path / ".env")
    for name in NAMES:
        # Recorded as unset, so monkeypatch removes whatever load_dotenv sets.
        clean_env.setenv(name, "")
        clean_env.delenv(name)
    return tmp_path


def _write(path, **values):
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    return path


def test_without_a_env_file_settings_come_from_the_environment(settings, clean_env, capsys):
    clean_env.setenv("RT_ENV_TEST_A", "from-shell")
    assert env.load_settings() == "environment variables (no .env)"
    assert env.os.environ["RT_ENV_TEST_A"] == "from-shell"
    assert "No .env found: reading settings from environment variables only" in capsys.readouterr().err


def test_a_env_file_is_loaded_and_the_environment_wins(settings, clean_env, capsys):
    _write(settings / ".env", RT_ENV_TEST_A="from-file", RT_ENV_TEST_B="from-file")
    clean_env.setenv("RT_ENV_TEST_A", "from-shell")
    assert env.load_settings() == ".env and environment variables"
    assert env.os.environ["RT_ENV_TEST_A"] == "from-shell"
    assert env.os.environ["RT_ENV_TEST_B"] == "from-file"
    assert capsys.readouterr().err == ""


def test_rt_env_file_names_another_file(settings, clean_env):
    _write(settings / ".env", RT_ENV_TEST_A="from-default")
    other = _write(settings / "pilot.env", RT_ENV_TEST_A="from-other")
    clean_env.setenv("RT_ENV_FILE", str(other))
    assert env.load_settings() == f"{other} and environment variables"
    assert env.os.environ["RT_ENV_TEST_A"] == "from-other"


def test_a_missing_rt_env_file_stops(settings, clean_env):
    # Running on the wrong settings is worse than not running.
    clean_env.setenv("RT_ENV_FILE", str(settings / "missing.env"))
    with pytest.raises(SystemExit, match="missing.env is not a file.*RT_ENV_FILE=none"):
        env.load_settings()


@pytest.mark.parametrize("value", ["none", "None", "off", "0"])
def test_rt_env_file_none_skips_the_env_file(settings, clean_env, capsys, value):
    _write(settings / ".env", RT_ENV_TEST_A="from-file")
    clean_env.setenv("RT_ENV_FILE", value)
    assert env.load_settings() == "environment variables (RT_ENV_FILE=none)"
    assert "RT_ENV_TEST_A" not in env.os.environ
    # Asked for on purpose, so no note.
    assert capsys.readouterr().err == ""


def test_settings_load_once_per_process(settings, capsys):
    env.load_settings()
    capsys.readouterr()
    _write(settings / ".env", RT_ENV_TEST_A="too-late")
    assert env.load_settings() == "environment variables (no .env)"
    assert "RT_ENV_TEST_A" not in env.os.environ
    assert capsys.readouterr().err == ""


def test_every_entry_point_loads_settings_through_load_settings():
    # A module calling load_dotenv() itself would skip RT_ENV_FILE.
    offenders = [
        str(path.relative_to(env.REPO_ROOT))
        for folder in ("harnesses", "scripts", "redteam")
        for path in (env.REPO_ROOT / folder).rglob("*.py")
        if path.name != "env.py" and "load_dotenv(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

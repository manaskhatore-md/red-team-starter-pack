"""The linter setup: checks that do not need GitHub or ruff to run."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _pinned_in_requirements():
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    pins = re.findall(r"^ruff==(\S+)\s*$", text, flags=re.MULTILINE)
    assert len(pins) == 1, "requirements-dev.txt should pin ruff exactly once, as ruff==X.Y.Z"
    return pins[0]


def _pre_commit_rev():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    revs = [repo["rev"] for repo in config["repos"] if repo["repo"].endswith("/ruff-pre-commit")]
    assert len(revs) == 1, ".pre-commit-config.yaml should have one ruff-pre-commit entry"
    return revs[0].removeprefix("v")


def _lint_job_version():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8"))
    versions = [
        step["with"]["version"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("astral-sh/ruff-action@")
    ]
    assert len(versions) == 1, "lint.yml should run astral-sh/ruff-action once, with a version"
    return str(versions[0])


def test_the_hook_and_the_lint_job_run_the_ruff_version_that_requirements_pin():
    # Ruff adds rules and changes findings between versions. If one of these is bumped
    # alone, a commit can pass the hook on your machine and then fail the Lint job on
    # the pull request, or the other way round.
    pinned = _pinned_in_requirements()
    assert _pre_commit_rev() == pinned, "bump the ruff-pre-commit rev in .pre-commit-config.yaml to match"
    assert _lint_job_version() == pinned, "bump version: in .github/workflows/lint.yml to match"

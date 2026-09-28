"""The CI gate workflow: checks that do not need GitHub to run."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "redteam-ci-gate.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps():
    for job in _workflow()["jobs"].values():
        yield from job.get("steps", [])


def test_piped_steps_keep_the_exit_code():
    # GitHub's default run shell is `bash -e {0}`: no pipefail, so a pipeline exits
    # with its LAST command's status. `scan | tee log` then reports tee's success
    # even when the scan exits 1 on a finding. `shell: bash` runs
    # `bash --noprofile --norc -eo pipefail {0}` instead.
    piped = [s for s in _steps() if "|" in s.get("run", "")]
    assert piped, "expected the scan step to pipe through tee"
    for step in piped:
        assert step.get("shell") == "bash", f"step {step.get('name')!r} pipes output without shell: bash"


def test_workflow_asks_only_for_the_permissions_it_uses():
    assert _workflow()["permissions"] == {"contents": "read"}

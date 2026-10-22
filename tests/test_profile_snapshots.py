"""What each profile runs today, pinned, so making RT_PROFILE optional cannot change it by accident.

RT_PROFILE is becoming optional: unset, a run will use a built-in default instead of
internal_productivity, and the named profiles stay as presets. Each named profile has
to keep running what it runs today: the same probes, numbered the same way for
--probes 1-15, scored by the same rubrics and checks, the same pairs compared, the same
planted records and stand-in system prompt, and the same multi-turn objectives.

The snapshots are taken only through what a user runs: RT_PROFILE, --list-probes, and
each campaign's main(), on fake models. A PR should not need to touch this file or
the snapshots it checks. If one does, that is the thing to review.

Text is pinned by hash, so a snapshot stays small enough to read and an edit to a
probe shows up as one changed line. When a change is intended, regenerate and review:

    UPDATE_PROFILE_SNAPSHOTS=1 python -m pytest tests/test_profile_snapshots.py
    git diff tests/fixtures/profile_snapshots/

A changed probe_numbering means --probes numbers now pick different probes; compare
`python -m harnesses.pyrit_campaigns.single_turn_scan --list-probes` before and after.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harnesses.pyrit_campaigns import multi_turn_crescendo, multi_turn_red_team, single_turn_scan
from harnesses.pyrit_campaigns.profiles import PROFILES, get_profile
from scripts.export_finding_report import full_conversation_scores
from helpers import FakeModel, nothing
from test_judge_context import FakeJudge

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = REPO_ROOT / "tests" / "fixtures" / "profile_snapshots"
# Read from the file, not through the scan's loader, which the refactor changes.
RUBRICS = set(yaml.safe_load(
    (REPO_ROOT / "harnesses" / "pyrit_campaigns" / "state_policy_rubric.yaml").read_text(encoding="utf-8")
)["rubrics"])
UPDATE = os.getenv("UPDATE_PROFILE_SNAPSHOTS") == "1"
AUTHORIZATION_BANNER = "This profile requires a non-production instance"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@pytest.fixture
def fixed_randomness(monkeypatch):
    """The same canary and planted records every run, so the stand-in prompt hashes the same."""
    monkeypatch.setattr(random, "SystemRandom", lambda: random.Random(0))
    monkeypatch.setattr(secrets, "token_hex", lambda n=None: "0123456789abcdef"[: 2 * (n or 16)])


class RecordingModel(FakeModel):
    """The fake model, keeping the system prompt it was sent."""

    def __init__(self):
        super().__init__()
        self.system_prompts = set()

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        pieces = [p for m in normalized_conversation for p in m.message_pieces]
        self.system_prompts.update(p.converted_value for p in pieces if p.role == "system")
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


def run_labels(labels: dict) -> dict:
    keep = ("rt_profile", "rt_system_prompt", "rt_system_prompt_chosen_by", "rt_planted_records", "rt_placeholders")
    return {k: labels.get(k) for k in keep}


def probe_numbering(capsys) -> str:
    """The numbers --list-probes gives this profile's probes, which --probes 1-15 selects by."""
    single_turn_scan.list_probes(get_profile(), None)
    numbered = re.findall(r"^\[(\d+)\] (\S+) / (\S+)", capsys.readouterr().out, re.MULTILINE)
    assert numbered and [int(n) for n, _, _ in numbered] == list(range(1, len(numbered) + 1))
    return digest("\n".join(f"{n} {dataset}/{name}" for n, dataset, name in numbered))


def scan_snapshot(profile: str | None, monkeypatch, clean_env, memory, capsys) -> dict:
    """Run the scan's main() on fake models with RT_PROFILE set to `profile` (None: unset)."""
    if profile is None:
        clean_env.delenv("RT_PROFILE", raising=False)
    else:
        clean_env.setenv("RT_PROFILE", profile)
    target = RecordingModel()
    runs = []
    for name in ("initialize_pyrit_async", "check_models", "close_target"):
        monkeypatch.setattr(single_turn_scan, name, nothing)
    monkeypatch.setattr(single_turn_scan, "build_target", lambda: target)
    monkeypatch.setattr(single_turn_scan, "build_scoring_target", lambda: FakeJudge())
    monkeypatch.setattr(single_turn_scan, "model_name", lambda t: "fake-model")
    monkeypatch.setattr(single_turn_scan, "write_after_run", runs.append)
    asyncio.run(single_turn_scan.main())
    out = capsys.readouterr().out

    results = [r for r in memory.get_attack_results() if r.labels.get("rt_run_id") == runs[0]]
    scored = {id(r): {c for s in full_conversation_scores(memory, r) for c in s.score_category} for r in results}
    # The code checks run on every reply; listed once, with any probe that differs showing its own.
    checks = set.intersection(*(categories - RUBRICS for categories in scored.values()))
    probes, compared = [], set()
    for r in results:
        rubrics = sorted(scored[id(r)] & RUBRICS)
        extra = sorted(scored[id(r)] - RUBRICS - checks)
        if "disparate_treatment" in rubrics:
            compared.add(r.labels["rt_pair_id"])
        line = f"{r.labels['rt_dataset']}/{r.labels['rt_probe']}  graded: {' + '.join(rubrics) or '-'}"
        line += f"  checks also: {' + '.join(extra)}" if extra else ""
        line += f"  pair: {r.labels['rt_pair_id']}" if r.labels.get("rt_pair_id") else ""
        probes.append(line + f"  text: {digest(r.objective)}")
    assert len(target.system_prompts) <= 1
    return {
        "labels": run_labels(results[0].labels),
        "probe_numbering": probe_numbering(capsys),
        "system_prompt": digest(target.system_prompts.pop()) if target.system_prompts else None,
        "authorization_banner": AUTHORIZATION_BANNER in out,
        "code_checks_on_every_reply": sorted(checks),
        "pairs_compared": sorted(compared),
        "probes": sorted(probes),
    }


def campaign_snapshot(campaign, key: str, monkeypatch, clean_env) -> dict:
    """Run a multi-turn campaign's main() on fake models; snapshot what its executor was handed."""
    handed = {}

    class Executor:
        def __init__(self, **kwargs):
            pass

        async def execute_attack_async(self, **kwargs):
            handed.update(kwargs)
            return SimpleNamespace(completed_results=[], incomplete_objectives=[])

    clean_env.setenv("RT_PROFILE", key)
    clean_env.setenv("RT_JUDGE_PROVIDER", "gemini")
    for name in ("initialize_pyrit_async", "check_models", "close_target"):
        monkeypatch.setattr(campaign, name, nothing)
    monkeypatch.setattr(campaign, "build_target", lambda role="target", **kw: FakeModel())
    monkeypatch.setattr(campaign, "build_scoring_target", lambda: FakeModel())
    monkeypatch.setattr(campaign, "model_name", lambda t: "fake")
    monkeypatch.setattr(campaign, "write_after_run", lambda run_id: None)
    monkeypatch.setattr(campaign, "AttackExecutor", Executor)
    asyncio.run(campaign.main())

    labels = [o["memory_labels"] for o in handed["field_overrides"]]
    prepended = handed.get("prepended_conversation") or []
    system = [p.converted_value for m in prepended for p in m.message_pieces if p.role == "system"]
    return {
        "labels": {**run_labels(labels[0]), "rt_objectives_file": labels[0]["rt_objectives_file"]},
        "system_prompt": digest(system[0]) if system else None,
        # Goals are short and few, so they are kept whole.
        "objectives": [
            {"rubric": l["rt_rubric"], "needs_system_prompt": l["rt_needs_system_prompt"], "goal": goal}
            for goal, l in zip(handed["objectives"], labels)
        ],
    }


def check(key: str, actual: dict) -> None:
    path = SNAPSHOTS / f"{key}.json"
    text = json.dumps(actual, indent=2, ensure_ascii=False) + "\n"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.is_file(), f"No snapshot at {path}. Generate it: UPDATE_PROFILE_SNAPSHOTS=1 python -m pytest {__file__}"
    assert json.loads(path.read_text(encoding="utf-8")) == actual, (
        f"{path.name} no longer matches. If the change is intended, regenerate with "
        "UPDATE_PROFILE_SNAPSHOTS=1 and review the diff."
    )


# --- what each profile runs --------------------------------------------------

def test_every_profile_has_a_snapshot_and_every_snapshot_a_profile():
    assert UPDATE or {p.stem for p in SNAPSHOTS.glob("*.json")} == set(PROFILES)


@pytest.mark.parametrize("key", sorted(PROFILES))
def test_what_each_profile_runs_is_unchanged(key, monkeypatch, clean_env, memory, capsys, fixed_randomness):
    snapshot = {"profile": key, "single_turn_scan": scan_snapshot(key, monkeypatch, clean_env, memory, capsys)}
    for name, campaign in (("multi_turn_crescendo", multi_turn_crescendo), ("multi_turn_red_team", multi_turn_red_team)):
        snapshot[name] = campaign_snapshot(campaign, key, monkeypatch, clean_env)
    check(key, snapshot)


# --- how RT_PROFILE is read ----------------------------------------------------

def saved_scan(key: str) -> dict:
    return json.loads((SNAPSHOTS / f"{key}.json").read_text(encoding="utf-8"))["single_turn_scan"]


def test_an_unset_rt_profile_runs_internal_productivity(monkeypatch, clean_env, memory, capsys, fixed_randomness):
    # CHANGES ON PURPOSE when RT_PROFILE becomes optional: unset then runs the built-in default.
    assert scan_snapshot(None, monkeypatch, clean_env, memory, capsys) == saved_scan("internal_productivity")


def test_rt_profile_is_read_in_any_case(monkeypatch, clean_env, memory, capsys, fixed_randomness):
    # An existing .env must keep working.
    actual = scan_snapshot("Public_Conversational", monkeypatch, clean_env, memory, capsys)
    assert actual == saved_scan("public_conversational")

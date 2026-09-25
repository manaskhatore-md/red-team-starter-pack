"""Single-turn scan - every probe in the profile's datasets, graded by your rubrics.

This is the broad, cheap pass: one prompt, one response, one judgment per probe. It
catches the failures that do not need patience, and it is the thing you re-run on a
schedule to catch regressions. The multi-turn campaigns
(pyrit_campaigns/multi_turn_crescendo.py) catch what this misses.

    RT_PROFILE=public_conversational python -m pyrit_campaigns.single_turn_scan

What runs is decided by the profile, not by this file: the profile names which
datasets/*.yaml probes to send and which judges/state_policy_rubric.yaml rubrics grade
them. See pyrit_campaigns/profiles.py.

POLARITY, because it inverts twice and trips everyone up: the rubrics return
true = policy violation = A FINDING. So a "true" score means your system failed, not
that the test passed. Scores are written to PyRIT's memory database alongside the
transcript, so reporting/export_finding_report.py can pick them up by conversation_id.

COST: one target call plus one judge call per probe per rubric. A profile with 13
probes and 3 rubrics is ~52 calls. Start with one dataset while you tune the rubrics.
"""

import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Fix for corporate TLS-inspecting proxies
import truststore
truststore.inject_into_ssl()

import yaml

from pyrit.executor.attack import AttackExecutor, AttackScoringConfig, PromptSendingAttack
from pyrit.memory import CentralMemory
from pyrit.models import SeedDataset
from pyrit.score import SelfAskGeneralTrueFalseScorer
from pyrit.setup import SQLITE, initialize_pyrit_async

from pyrit_campaigns.profiles import PAIR_RUBRICS, Profile, describe, get_profile
from pyrit_campaigns.target_factory import build_scoring_target, build_target, close_target

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RUBRIC_FILE = REPO_ROOT / "judges" / "state_policy_rubric.yaml"

# TODO: raise once you know your provider's rate limit; lower it if you are being
# throttled. Judge calls count against the same quota.
MAX_CONCURRENCY = int(os.getenv("RT_MAX_CONCURRENCY", "3"))

# Off by default: in a red team run, findings are the expected output, so a non-zero
# exit would look like a broken script. Set RT_FAIL_ON_FINDING=1 in CI, where you do
# want a finding to fail the job.
FAIL_ON_FINDING = os.getenv("RT_FAIL_ON_FINDING") == "1"


def load_probes(profile: Profile) -> list[tuple[str, str]]:
    """Return (dataset_name, rendered_prompt) for every seed the profile selects."""
    probes: list[tuple[str, str]] = []
    for name in profile.datasets:
        path = DATASETS_DIR / f"{name}.yaml"
        if not path.exists():
            raise SystemExit(f"Profile {profile.key!r} names a missing dataset: {path}")
        dataset = SeedDataset.from_yaml_file(path)
        for seed in dataset.seeds:
            # _silent leaves unrecognized placeholders in place rather than raising,
            # so adding a token to a dataset does not break the run before you have
            # filled it in on the profile.
            probes.append((name, seed.render_template_value_silent(**profile.placeholders)))
    return probes


def build_rubric_scorers(profile: Profile, judge) -> list[tuple[str, SelfAskGeneralTrueFalseScorer]]:
    """Turn the profile's named rubrics into scorers.

    Skips the pair rubrics: they compare two responses to matched-pair inputs, and
    there is nothing to compare against in a single-response scan.
    """
    rubrics = yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8"))["rubrics"]

    scorers = []
    for key in profile.rubrics:
        if key in PAIR_RUBRICS:
            print(f"  skipping rubric {key!r}: needs a matched pair, not a single response")
            continue
        if key not in rubrics:
            raise SystemExit(f"Profile {profile.key!r} names rubric {key!r}, absent from {RUBRIC_FILE.name}")
        scorers.append(
            (
                key,
                SelfAskGeneralTrueFalseScorer(
                    system_prompt_format_string=rubrics[key]["system_prompt"],
                    chat_target=judge,
                    # Lands in Score.score_category so findings can be grouped by
                    # rubric in the report.
                    category=key,
                ),
            )
        )
    return scorers


UNRENDERED = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def warn_on_placeholders(profile: Profile, probes: list[tuple[str, str]]) -> None:
    """Generic probes find generic problems. Say so, loudly, but do not block.

    Two different failures, and the second is the one that wastes a whole run: a
    placeholder left at its TODO default, and a dataset token the profile has no value
    for at all. The latter has to be checked on the RENDERED probes - the profile dict
    cannot tell you about a token it is missing.
    """
    unfilled = sorted(k for k, v in profile.placeholders.items() if str(v).startswith("TODO"))
    if unfilled:
        print(
            f"\n!! Profile {profile.key!r} still has placeholder values for: {', '.join(unfilled)}\n"
            "!! The scan will run, but probes naming your real program, tools, and record\n"
            "!! types find far more than generic ones. Fill them in pyrit_campaigns/profiles.py.\n"
        )

    missing = sorted({m for _, prompt in probes for m in UNRENDERED.findall(prompt)})
    if missing:
        print(
            f"\n!! {len(missing)} dataset token(s) have no value in profile {profile.key!r}:\n"
            f"!!   {', '.join(missing)}\n"
            "!! Those probes will go out with the literal {{ token }} text in them, which\n"
            "!! tests nothing useful. Add them to the profile's placeholders in\n"
            "!! pyrit_campaigns/profiles.py before you trust this run.\n"
        )


async def main() -> int:
    profile = get_profile()

    # SQLITE so the transcript and scores survive the run - they are the evidence
    # for the finding report. Writes to ~/.pyrit/dbdata/ by default, and inherits the
    # classification of whatever you sent.
    await initialize_pyrit_async(memory_db_type=SQLITE)

    print(describe(profile))

    probes = load_probes(profile)
    warn_on_placeholders(profile, probes)

    target = build_target()
    judge = build_scoring_target()
    scorers = build_rubric_scorers(profile, judge)

    if not scorers:
        raise SystemExit(f"Profile {profile.key!r} has no single-response rubrics to score with.")

    print(f"\nSending {len(probes)} probes, scored by {len(scorers)} rubric(s)...\n")

    attack = PromptSendingAttack(
        objective_target=target,
        attack_scoring_config=AttackScoringConfig(
            # Auxiliary rather than objective: there is no single "did the attack
            # win" question here, there are N independent policy questions. Auxiliary
            # scores are still written to memory against the response.
            auxiliary_scorers=[scorer for _, scorer in scorers],
        ),
    )

    executor_result = await AttackExecutor(max_concurrency=MAX_CONCURRENCY).execute_attack_async(
        attack=attack,
        objectives=[prompt for _, prompt in probes],
    )

    memory = CentralMemory.get_memory_instance()
    findings: list[tuple[str, str, str]] = []   # (rubric, probe, rationale)
    errors: list[str] = []
    by_rubric: Counter[str] = Counter()

    for objective, exception in executor_result.incomplete_objectives:
        errors.append(f"{objective[:70]!r} did not complete: {exception}")

    for result in executor_result.completed_results:
        response = result.last_response
        probe = result.objective

        if response is None or getattr(response, "response_error", "none") != "none":
            error = "no response" if response is None else response.response_error
            errors.append(f"{probe[:70]!r} -> {error}")
            continue

        for score in memory.get_prompt_scores(conversation_id=result.conversation_id):
            # score_category is a list - a scorer can carry several. Flatten it so it
            # can be used as a grouping key.
            categories = score.score_category or []
            rubric = ", ".join(categories) if isinstance(categories, list) else str(categories)
            if score.get_value() is True:
                findings.append((rubric, probe, score.score_rationale or ""))
                by_rubric[rubric] += 1

        print(f"--- [{result.conversation_id}]")
        print(f"    probe: {probe[:100]}")
        print(f"    reply: {(response.converted_value or '')[:200]}")

    print("\n" + "=" * 78)
    print(f"SCAN COMPLETE - {len(probes)} probes, {len(findings)} finding(s)")
    print("=" * 78)

    for rubric, count in by_rubric.most_common():
        print(f"  {rubric}: {count}")

    for rubric, probe, rationale in findings:
        print(f"\n[FINDING: {rubric}]")
        print(f"  probe:     {probe[:150]}")
        print(f"  rationale: {rationale[:300]}")

    if errors:
        print(f"\n{len(errors)} probe(s) produced no usable response:")
        for error in errors:
            print(f"  - {error}")
        print("  (a blocked or empty response is not a pass - it means nothing was tested)")

    if profile.checklist:
        print(f"\nBefore calling {profile.key!r} covered:")
        for item in profile.checklist:
            print(f"  [ ] {item}")

    # TODO: turn these into report drafts automatically -
    #   python -m reporting.export_finding_report --conversation-id <id> --out findings/
    print("\nNext: reporting/export_finding_report.py takes any conversation_id above.")
    print("LLM output is stochastic - re-run a finding 5-10 times before you report it.")

    for t in (target, judge):
        await close_target(t)

    if errors:
        return 1
    if findings and FAIL_ON_FINDING:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

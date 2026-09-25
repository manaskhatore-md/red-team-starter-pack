"""Single-turn scan - every probe in the profile's datasets, graded by your rubrics.

This is the broad, cheap pass: one prompt, one response, one judgment per probe. It
catches the failures that do not need patience, and it is the thing you re-run on a
schedule to catch regressions. The multi-turn campaigns
(pyrit_campaigns/multi_turn_crescendo.py) catch what this misses.

    RT_PROFILE=public_conversational python -m pyrit_campaigns.single_turn_scan

What runs is decided by the profile, not by this file: the profile names which
datasets/*.yaml probes to send and which judges/state_policy_rubric.yaml rubrics grade
them. See pyrit_campaigns/profiles.py.

MATCHED PAIRS (datasets/algorithmic_bias.yaml): bias only shows when two replies are
compared, so after the scan, the two replies to each pair are handed to the judge
together under the disparate_treatment rubric. The verdict is recorded on the second
reply of the pair, so it reports and exports like any other finding.

POLARITY, because it inverts twice and trips everyone up: the rubrics return
true = policy violation = A FINDING. So a "true" score means your system failed, not
that the test passed. Scores are written to PyRIT's memory database alongside the
transcript, so reporting/export_finding_report.py can pick them up by conversation_id.

COST: one target call plus one judge call per probe per rubric, plus one judge call
per matched pair. A profile with 13 probes and 3 rubrics is ~52 calls. Start with one dataset while you tune the rubrics.
"""

import asyncio
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from pyrit.models import MessageScorable, SeedDataset
from pyrit.score import SelfAskGeneralTrueFalseScorer
from pyrit.setup import SQLITE, initialize_pyrit_async

from pyrit_campaigns.profiles import PAIR_RUBRICS, Profile, describe, env_var, get_profile
from pyrit_campaigns.target_factory import build_scoring_target, build_target, close_target, model_name

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


@dataclass(frozen=True)
class Probe:
    dataset: str
    name: str
    prompt: str
    harm_categories: tuple[str, ...]
    # Set only for matched-pair probes (datasets/algorithmic_bias.yaml).
    pair_id: str = ""
    variant: str = ""
    compare_on: str = ""


def load_probes(profile: Profile) -> list[Probe]:
    """Return a Probe for every seed the profile selects."""
    probes: list[Probe] = []
    for name in profile.datasets:
        path = DATASETS_DIR / f"{name}.yaml"
        if not path.exists():
            raise SystemExit(f"Profile {profile.key!r} names a missing dataset: {path}")
        dataset = SeedDataset.from_yaml_file(path)
        for seed in dataset.seeds:
            metadata = seed.metadata or {}
            probes.append(
                Probe(
                    dataset=name,
                    name=seed.name or "",
                    # _silent leaves unrecognized placeholders in place rather than
                    # raising, so adding a token to a dataset does not break the run
                    # before you have filled it in on the profile.
                    prompt=seed.render_template_value_silent(**profile.placeholders),
                    harm_categories=tuple(seed.harm_categories or ()),
                    pair_id=str(metadata.get("pair_id", "")),
                    variant=str(metadata.get("variant", "")),
                    compare_on=str(metadata.get("compare_on", "")),
                )
            )
    return probes


def load_rubrics() -> dict:
    return yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8"))["rubrics"]


def build_rubric_scorers(profile: Profile, judge) -> list[tuple[str, SelfAskGeneralTrueFalseScorer]]:
    """Turn the profile's named rubrics into scorers.

    Leaves out the pair rubrics: they compare two responses rather than grade one,
    and compare_pairs() runs them.
    """
    rubrics = load_rubrics()

    scorers = []
    for key in profile.rubrics:
        if key in PAIR_RUBRICS:
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


def _escape_braces(text: str) -> str:
    # The scorer runs str.format() over its system prompt, so literal braces in a
    # reply (JSON, code) would break it.
    return text.replace("{", "{{").replace("}", "}}")


async def compare_pairs(probes: list[Probe], results_by_prompt: dict, memory, judge, rubric_prompt: str):
    """Judge each matched pair: did the substance change when one attribute changed?

    Returns (pair_id, probe_a, probe_b, score) for every pair where both replies came back.
    The score is recorded against the second reply, so it lands in that
    conversation's scores like any rubric verdict.
    """
    pairs: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        if probe.pair_id:
            pairs[probe.pair_id].append(probe)

    verdicts = []
    for pair_id, members in pairs.items():
        if len(members) != 2:
            print(f"  pair {pair_id!r} has {len(members)} probe(s), not 2 - not compared")
            continue
        a, b = members
        result_a, result_b = results_by_prompt.get(a.prompt), results_by_prompt.get(b.prompt)
        if not (result_a and result_b and result_a.last_response and result_b.last_response):
            print(f"  pair {pair_id!r}: a reply is missing - not compared")
            continue

        # Pair ids look like "dialect_01"; the prefix names the attribute.
        attribute = f"{pair_id.rsplit('_', 1)[0]} ({a.variant} vs. {b.variant})"
        if a.compare_on:
            attribute += f". Compare especially: {a.compare_on}"
        filled = (
            rubric_prompt.replace("[[RESPONSE_A]]", result_a.last_response.converted_value or "")
            .replace("[[RESPONSE_B]]", result_b.last_response.converted_value or "")
            .replace("[[ATTRIBUTE]]", attribute)
        )
        scorer = SelfAskGeneralTrueFalseScorer(
            system_prompt_format_string=_escape_braces(filled),
            chat_target=judge,
            category="disparate_treatment",
        )
        reply_b = memory.get_conversation_messages(conversation_id=result_b.conversation_id)[-1]
        scores = await scorer.score_async(scorable=MessageScorable.from_message(reply_b))
        verdicts.append((pair_id, a, b, scores[0]))
    return verdicts


UNRENDERED = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def check_placeholders(profile: Profile, probes: list[Probe]) -> None:
    """Stop before sending probes that still say "TODO Program".

    A model asked about "TODO Program" answers about a program that does not
    exist, and the judges then grade that - so the findings describe the
    placeholder, not your system. RT_ALLOW_PLACEHOLDERS=1 runs anyway, for trying
    out the tooling.

    Also warns about a dataset token the profile has no value for at all. That has
    to be checked on the RENDERED probes - the profile dict cannot tell you about a
    token it is missing.
    """
    unfilled = sorted(
        key
        for key, value in profile.placeholders.items()
        if str(value).startswith("TODO") and any(str(value) in probe.prompt for probe in probes)
    )
    if unfilled and os.getenv("RT_ALLOW_PLACEHOLDERS") != "1":
        lines = "\n".join(f"    {env_var(key)}=" for key in unfilled)
        raise SystemExit(
            f"\nStopped before sending anything: {len(unfilled)} value(s) the probes use are still\n"
            "placeholders, so the model would be asked about \"TODO Program\" instead of yours.\n"
            "Add these lines to .env with your own values (.env.example explains each one):\n\n"
            f"{lines}\n\n"
            "To run anyway, only to try out the tooling, also add RT_ALLOW_PLACEHOLDERS=1."
        )
    if unfilled:
        print(
            f"\n!! Running with placeholder values for: {', '.join(unfilled)} (RT_ALLOW_PLACEHOLDERS=1).\n"
            "!! Findings from this run describe the placeholders, not your system.\n"
        )

    missing = sorted({m for probe in probes for m in UNRENDERED.findall(probe.prompt)})
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
    check_placeholders(profile, probes)

    target = build_target()
    judge = build_scoring_target()
    scorers = build_rubric_scorers(profile, judge)

    if not scorers:
        raise SystemExit(f"Profile {profile.key!r} has no single-response rubrics to score with.")

    compare = "disparate_treatment" in profile.rubrics and any(p.pair_id for p in probes)
    pair_note = ", then matched pairs compared" if compare else ""
    print(f"\nSending {len(probes)} probes, scored by {len(scorers)} rubric(s){pair_note}...\n")

    attack = PromptSendingAttack(
        objective_target=target,
        attack_scoring_config=AttackScoringConfig(
            # Auxiliary rather than objective: there is no single "did the attack
            # win" question here, there are N independent policy questions. Auxiliary
            # scores are still written to memory against the response.
            auxiliary_scorers=[scorer for _, scorer in scorers],
        ),
    )

    # Labels are stored with each result, so the finding report can say what ran
    # without anyone copying it over by hand.
    run_labels = {
        "rt_campaign": "single_turn_scan",
        # Ties a run's results together - the report uses it to find a pair's partner.
        "rt_run_id": str(uuid.uuid4()),
        "rt_profile": profile.key,
        "rt_provider": os.getenv("RT_PROVIDER", "gemini"),
        "rt_target": model_name(target),
        "rt_judge": model_name(judge),
    }
    executor_result = await AttackExecutor(max_concurrency=MAX_CONCURRENCY).execute_attack_async(
        attack=attack,
        objectives=[probe.prompt for probe in probes],
        field_overrides=[
            {
                "memory_labels": {
                    **run_labels,
                    "rt_dataset": probe.dataset,
                    "rt_probe": probe.name,
                    "rt_harm_categories": ", ".join(probe.harm_categories),
                    "rt_pair_id": probe.pair_id,
                    "rt_variant": probe.variant,
                }
            }
            for probe in probes
        ],
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

    if compare:
        print("\nComparing matched pairs...")
        results_by_prompt = {r.objective: r for r in executor_result.completed_results}
        rubric_prompt = load_rubrics()["disparate_treatment"]["system_prompt"]
        for pair_id, a, b, score in await compare_pairs(probes, results_by_prompt, memory, judge, rubric_prompt):
            flagged = score.get_value() is True
            label = f"pair {pair_id} ({a.variant} vs. {b.variant})"
            print(f"  {label}: {'DIFFERENT TREATMENT' if flagged else 'equivalent'}")
            if flagged:
                findings.append(("disparate_treatment", label, score.score_rationale or ""))
                by_rubric["disparate_treatment"] += 1

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

    print("\nNext: python -m reporting.export_finding_report --rubric any --out findings/")
    print("      writes a report for every finding above (--conversation-id exports one).")
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

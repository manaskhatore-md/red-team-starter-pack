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

WHICH RUBRIC GRADES WHICH PROBE: each rubric in judges/state_policy_rubric.yaml lists
the datasets it `grades`, and a probe is graded only by the profile's rubrics that
list its dataset. A probe no rubric grades is not sent, and the scan says so.
The judge is also shown the probe's pass_criteria from its dataset, what a safe
reply to that probe looks like, where the rubric has a [[PASS_CRITERIA]] slot. A
rubric with a [[SYSTEM_PROMPT]] slot (injection_compliance) is shown the system
prompt too, so a reply that does what its instructions say is not mistaken for one
that was talked out of them.

THE SYSTEM PROMPT: each probe is sent after a system prompt, as a deployed app would
send it - by default the stand-in in system_prompts/ for the profile, or the file
RT_SYSTEM_PROMPT_FILE names (RT_SYSTEM_PROMPT_FILE=none for no system prompt). It
carries a random canary token, and every reply is checked for it: a reply containing it has
leaked the system prompt, and counts as a system_prompt_leak finding whatever the
judges said. See pyrit_campaigns/system_prompt.py.

REPEATS: model output varies from run to run, so one reply per probe is one sample.
RT_REPEATS=5 sends every probe 5 times, each in a new conversation, and the run
summary reports how often each one was flagged ("2 of 5"). Pairs are compared
within each repeat. The default is 1.

COST: one target call per probe, one judge call per probe per rubric that grades it,
plus one judge call per matched pair - all times RT_REPEATS. public_conversational
is 16 + 16 + 5 = 37 calls per repeat. The scan prints the total before it sends
anything. Start with one dataset while you tune the rubrics.
"""

import asyncio
import os
import re
import sys
import time
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
from pyrit.models import ComponentIdentifier, MessagePiece, MessageScorable, Score, SeedDataset
from pyrit.score import MessageTrueFalseScorer, SelfAskGeneralTrueFalseScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.setup import SQLITE, initialize_pyrit_async

from pyrit_campaigns.profiles import PAIR_RUBRICS, Profile, describe, env_var, get_profile
from pyrit_campaigns.code_checks import code_scorers, fill_pattern_check
from pyrit_campaigns.planted_records import PlantedRecordScorer, records_for, records_section
from pyrit_campaigns.system_prompt import (
    CanaryLeakScorer,
    load_system_prompt,
    new_canary,
    system_message,
    takes_system_prompt,
)
from reporting.run_summary import write_after_run
from pyrit_campaigns.target_factory import (
    build_scoring_target,
    build_target,
    check_models,
    close_target,
    model_name,
    resolve_provider,
)

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


class ProgressAttack(PromptSendingAttack):
    """PromptSendingAttack that prints one line as each probe finishes.

    A probe is a target call plus a judge call per rubric, so a scan can otherwise
    sit silent for minutes. `progress` is shared by every attack in the run, so the
    count runs across them: {"done": 0, "total": N, "repeats": RT_REPEATS}.
    """

    def __init__(self, *, progress: dict[str, int], **kwargs):
        super().__init__(**kwargs)
        self._progress = progress

    async def execute_with_context_async(self, *, context):
        started = time.monotonic()
        try:
            result = await super().execute_with_context_async(context=context)
        except Exception as e:
            # PyRIT wraps the error twice; the innermost one says what went wrong.
            root = e
            while root.__cause__ is not None:
                root = root.__cause__
            self._report(context, started, f"ERROR {type(root).__name__}: {str(root)[:120]}")
            raise
        response = result.last_response
        if response is None or response.response_error != "none":
            status = f"ERROR {'no response' if response is None else response.response_error}"
        else:
            scores = CentralMemory.get_memory_instance().get_prompt_scores(conversation_id=result.conversation_id)
            flagged = sorted({
                ", ".join(s.score_category) if isinstance(s.score_category, list) else str(s.score_category)
                for s in scores if s.get_value() is True
            })
            status = f"FLAGGED {'; '.join(flagged)}" if flagged else "ok"
        self._report(context, started, status)
        return result

    def _report(self, context, started: float, status: str) -> None:
        self._progress["done"] += 1
        width = len(str(self._progress["total"]))
        labels = context.memory_labels
        name = labels.get("rt_probe") or context.objective[:40]
        if self._progress["repeats"] > 1:
            name += f" (repeat {labels.get('rt_repeat')})"
        print(
            f"  [{self._progress['done']:>{width}}/{self._progress['total']}] "
            f"{name}: {status} ({time.monotonic() - started:.1f}s)",
            flush=True,
        )


def get_repeats() -> int:
    """RT_REPEATS: how many times to send each probe. Default 1."""
    raw = os.getenv("RT_REPEATS", "1").strip()
    if not raw.isdigit() or int(raw) < 1:
        raise SystemExit(f"RT_REPEATS must be a whole number of 1 or more, not {raw!r}.")
    return int(raw)


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
    # What the probe's author says a safe reply looks like. The judge is shown it.
    pass_criteria: str = ""


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
                    pass_criteria=" ".join(str(metadata.get("pass_criteria", "")).split()),
                )
            )
    return probes


def load_rubrics() -> dict:
    return yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8"))["rubrics"]


NO_PASS_CRITERIA = "None were written for this probe."
NO_SYSTEM_PROMPT = "None. The assistant was sent no system prompt in this run."


class RubricScorer(MessageTrueFalseScorer):
    """A rubric from judges/state_policy_rubric.yaml, with each probe's pass criteria filled in.

    One scorer grades a whole group of probes, and a rubric's text is fixed when the
    scorer is built, but [[PASS_CRITERIA]] differs per probe. So this fills it in for
    the probe being scored and hands the result to PyRIT's own rubric scorer.
    """

    _DEFAULT_VALIDATOR: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["text"])

    def __init__(self, *, rubric_prompt: str, judge, category: str, pass_criteria: dict[str, str]) -> None:
        self._rubric_prompt = rubric_prompt
        self._judge = judge
        self._category = category
        # Probe text -> its pass criteria. The objective PyRIT scores against is the probe text.
        self._pass_criteria = pass_criteria
        super().__init__(validator=self._DEFAULT_VALIDATOR)

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={"system_prompt_template": self._rubric_prompt, "category": self._category},
            prompt_target=self._judge.get_identifier(),
        )

    def rubric_for(self, objective: str | None) -> str:
        criteria = self._pass_criteria.get(objective or "") or NO_PASS_CRITERIA
        return self._rubric_prompt.replace("[[PASS_CRITERIA]]", _escape_braces(criteria))

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: str | None = None) -> list[Score]:
        scorer = SelfAskGeneralTrueFalseScorer(
            # A rubric with a [[PATTERN_CHECK]] slot is told what the pattern check found.
            system_prompt_format_string=fill_pattern_check(self.rubric_for(objective), message_piece.converted_value or ""),
            chat_target=self._judge,
            # Lands in Score.score_category so findings can be grouped by
            # rubric in the report.
            category=self._category,
        )
        return await scorer._score_piece_async(message_piece, objective=objective)


def build_rubric_scorers(
    profile: Profile, judge, probes: list[Probe] = (), system_prompt: str | None = None
) -> dict[str, RubricScorer]:
    """Turn the profile's named rubrics into scorers, keyed by rubric name.

    A rubric with a [[SYSTEM_PROMPT]] slot is shown the system prompt the model was
    sent, so its judge can tell following instructions from breaking them.

    Leaves out the pair rubrics: they compare two responses rather than grade one,
    and compare_pairs() runs them.
    """
    rubrics = load_rubrics()
    pass_criteria = {probe.prompt: probe.pass_criteria for probe in probes if probe.pass_criteria}

    scorers = {}
    for key in profile.rubrics:
        if key not in rubrics:
            raise SystemExit(f"Profile {profile.key!r} names rubric {key!r}, absent from {RUBRIC_FILE.name}")
        if key in PAIR_RUBRICS:
            continue
        scorers[key] = RubricScorer(
            rubric_prompt=rubric_prompt(rubrics[key], system_prompt),
            judge=judge,
            category=key,
            pass_criteria=pass_criteria,
        )
    return scorers


def rubric_prompt(rubric: dict, system_prompt: str | None) -> str:
    """A rubric's judge prompt with the system prompt the model was sent filled in, where it has the slot."""
    return rubric["system_prompt"].replace("[[SYSTEM_PROMPT]]", _escape_braces(system_prompt or NO_SYSTEM_PROMPT))


def group_by_rubrics(probes: list[Probe], rubric_keys, compare: bool) -> dict[tuple[str, ...], list[Probe]]:
    """Group probes by which of the given rubrics grade them (their `grades` list).

    A pair probe with no rubric of its own is still sent when pairs are compared,
    under the empty key: its reply is half of a comparison.
    """
    rubrics = load_rubrics()
    groups: dict[tuple[str, ...], list[Probe]] = defaultdict(list)
    for probe in probes:
        keys = tuple(k for k in rubric_keys if probe.dataset in rubrics[k].get("grades", []))
        if keys or (compare and probe.pair_id):
            groups[keys].append(probe)
    return groups


def _escape_braces(text: str) -> str:
    # The scorer runs str.format() over its system prompt, so literal braces in a
    # reply (JSON, code) would break it.
    return text.replace("{", "{{").replace("}", "}}")


def by_prompt_and_repeat(results) -> dict[tuple[str, int], object]:
    """Index results by (probe text, repeat number), so each repeat's pair halves find each other."""
    return {(r.objective, int((r.labels or {}).get("rt_repeat") or 1)): r for r in results}


async def compare_pairs(probes: list[Probe], results: dict, memory, judge, rubric_prompt: str, repeats: int = 1):
    """Judge each matched pair: did the substance change when one attribute changed?

    `results` is keyed by (probe text, repeat), as by_prompt_and_repeat builds it.
    Each repeat's two replies are compared with each other, never with another
    repeat's. Returns (pair_id, repeat, probe_a, probe_b, score) for every pair and
    repeat where both replies came back. The score is recorded against the second
    reply, so it lands in that conversation's scores like any rubric verdict.
    """
    pairs: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        if probe.pair_id:
            pairs[probe.pair_id].append(probe)

    verdicts = []
    for (pair_id, members), repeat in ((p, r) for p in pairs.items() for r in range(1, repeats + 1)):
        name = f"pair {pair_id!r}" + (f" (repeat {repeat})" if repeats > 1 else "")
        if len(members) != 2:
            if repeat == 1:
                print(f"  pair {pair_id!r} has {len(members)} probe(s), not 2 - not compared")
            continue
        a, b = members
        result_a, result_b = results.get((a.prompt, repeat)), results.get((b.prompt, repeat))
        if not (result_a and result_b and result_a.last_response and result_b.last_response):
            print(f"  {name}: a reply is missing - not compared")
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
        verdicts.append((pair_id, repeat, a, b, scores[0]))
    return verdicts


def expand_repeats(group: list[Probe], repeats: int, run_labels: dict) -> tuple[list[str], list[dict]]:
    """The objectives and per-objective labels for one attack: every probe, `repeats` times.

    Each copy is its own conversation, labelled with its repeat number (from 1).
    """
    objectives, overrides = [], []
    for probe in group:
        for repeat in range(1, repeats + 1):
            objectives.append(probe.prompt)
            overrides.append({
                "memory_labels": {
                    **run_labels,
                    "rt_dataset": probe.dataset,
                    "rt_probe": probe.name,
                    "rt_harm_categories": ", ".join(probe.harm_categories),
                    "rt_pair_id": probe.pair_id,
                    "rt_variant": probe.variant,
                    "rt_repeat": str(repeat),
                }
            })
    return objectives, overrides


def planned_calls(groups: dict[tuple[str, ...], list[Probe]], pair_count: int, repeats: int) -> tuple[int, int]:
    """(target calls, judge calls) the scan will make, before any retries."""
    target = sum(len(group) for group in groups.values()) * repeats
    judge = (sum(len(group) * len(keys) for keys, group in groups.items()) + pair_count) * repeats
    return target, judge


UNRENDERED = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def check_placeholders(profile: Profile, probes: list[Probe], system_prompt: str = "") -> None:
    """Warn when the probes still say "TODO Program". The scan runs either way.

    A model asked about "TODO Program" answers about a program that does not
    exist, and the judges then grade that - so the findings describe the
    placeholder, not your system.

    Also warns about a dataset token the profile has no value for at all. That has
    to be checked on the RENDERED probes - the profile dict cannot tell you about a
    token it is missing.

    The system prompt is filled from the same values, so it is checked too.
    """
    texts = [probe.prompt for probe in probes] + ([system_prompt] if system_prompt else [])
    unfilled = sorted(
        key
        for key, value in profile.placeholders.items()
        if str(value).startswith("TODO") and any(str(value) in text for text in texts)
    )
    if unfilled:
        lines = "\n".join(f"!!     {env_var(key)}=" for key in unfilled)
        print(
            f"\n!! Running with placeholder values, so the model is asked about \"TODO Program\"\n"
            "!! instead of yours, and findings describe the placeholders. For a real run, add\n"
            "!! these lines to .env with your own values (.env.example explains each one):\n"
            f"{lines}\n"
        )

    missing = sorted({m for text in texts for m in UNRENDERED.findall(text)})
    if missing:
        print(
            f"\n!! {len(missing)} token(s) have no value in profile {profile.key!r}:\n"
            f"!!   {', '.join(missing)}\n"
            "!! The probes or system prompt will go out with the literal {{ token }} text\n"
            "!! in them, which tests nothing useful. Add them to the profile's placeholders in\n"
            "!! pyrit_campaigns/profiles.py before you trust this run.\n"
        )


async def main() -> int:
    started = time.monotonic()
    profile = get_profile()
    repeats = get_repeats()

    # SQLITE so the transcript and scores survive the run - they are the evidence
    # for the finding report. Writes to PyRIT's data folder, and inherits the
    # classification of whatever you sent. To find the file, print it with: python -c "from pyrit.common.path import DB_DATA_PATH; print(DB_DATA_PATH)"
    await initialize_pyrit_async(memory_db_type=SQLITE)

    print(describe(profile))

    probes = load_probes(profile)
    records = records_for(profile)
    system_prompt = load_system_prompt(
        profile.key, profile.placeholders, new_canary(), records_section(records) if records else ""
    )
    if not system_prompt:
        records = ()
    check_placeholders(profile, probes, system_prompt.text if system_prompt else "")

    target = build_target()
    if system_prompt:
        print(f"\nSystem prompt: {system_prompt.source} (chosen by {system_prompt.chosen_by}), "
              f"with canary token {system_prompt.canary}.")
        print("Every reply is checked for the canary token; a reply that contains it has leaked the system prompt.")
        if records:
            print(f"It also holds {len(records)} fake records of other people and a fake database credential "
                  "(pyrit_campaigns/planted_records.py). Every reply is checked for them.")
        if not takes_system_prompt(target):
            print(
                "\n!! This target does not take a system prompt, so PyRIT sends it as the start of\n"
                "!! the user's message instead. The model sees the instructions as coming from the\n"
                "!! user, which makes extraction and override probes easier than in a real deployment.\n"
            )
    else:
        print("\nSystem prompt: none. The model gets each probe with no instructions.")
    judge = build_scoring_target()
    await check_models(target=target, judge=judge)
    scorers = build_rubric_scorers(profile, judge, probes, system_prompt.text if system_prompt else None)

    compare = "disparate_treatment" in profile.rubrics and any(p.pair_id for p in probes)
    groups = group_by_rubrics(probes, scorers, compare)
    sent = [probe for group in groups.values() for probe in group]

    ungraded = [probe for probe in probes if probe not in sent]
    if ungraded:
        datasets = sorted({probe.dataset for probe in ungraded})
        print(
            f"\n!! {len(ungraded)} probe(s) from {', '.join(datasets)} are NOT sent: no rubric in profile\n"
            f"!! {profile.key!r} grades them. Add a rubric whose `grades` list names the dataset\n"
            "!! (judges/state_policy_rubric.yaml) to the profile, or drop the dataset.\n"
        )
    if not sent:
        raise SystemExit(f"Profile {profile.key!r} has no rubric that grades any of its probes.")

    pair_count = len({p.pair_id for p in sent if p.pair_id}) if compare else 0
    graded = [f"{len(group)} graded by {' + '.join(keys)}" for keys, group in groups.items() if keys]
    ungraded_pairs = sum(len(group) for keys, group in groups.items() if not keys)
    if ungraded_pairs:
        graded.append(f"{ungraded_pairs} sent only for pair comparison")
    pair_note = f"; then {pair_count} matched pairs compared" if pair_count else ""
    target_calls, judge_calls = planned_calls(groups, pair_count, repeats)
    repeat_note = f", each {repeats} times (RT_REPEATS)" if repeats > 1 else ""
    print(f"\nSending {len(sent)} probes{repeat_note}: {', '.join(graded)}{pair_note}.")
    print(f"That is about {target_calls} target calls and {judge_calls} judge calls, before retries.\n")

    # Labels are stored with each result, so the finding report can say what ran
    # without anyone copying it over by hand.
    run_labels = {
        "rt_campaign": "single_turn_scan",
        # Ties a run's results together - the report uses it to find a pair's partner.
        "rt_run_id": str(uuid.uuid4()),
        "rt_profile": profile.key,
        "rt_provider": resolve_provider("target"),
        "rt_target": model_name(target),
        "rt_judge_provider": resolve_provider("judge"),
        "rt_judge": model_name(judge),
        # Where the system prompt came from, so a report can say what the model was told.
        "rt_system_prompt": system_prompt.source if system_prompt else "none",
        "rt_system_prompt_chosen_by": system_prompt.chosen_by if system_prompt else "",
        "rt_prompt_canary": system_prompt.canary if system_prompt else "",
        "rt_planted_records": str(len(records)),
        "rt_repeats": str(repeats),
    }

    # Every reply is checked for the canary, whichever rubrics grade it. The check
    # needs no judge call, so it adds nothing to the cost.
    canary_scorers = [CanaryLeakScorer(canary=system_prompt.canary)] if system_prompt else []
    canary_scorers += [PlantedRecordScorer(records=records)] if records else []
    # Credentials, markdown exfiltration, and realistic SSNs or card numbers: code_checks.py.
    canary_scorers += code_scorers()
    # PyRIT puts these messages ahead of each probe, in a new conversation per probe.
    prepended = {"prepended_conversation": [system_message(system_prompt)]} if system_prompt else {}

    # One attack per group, because an attack applies its scorers to every probe
    # it sends.
    completed, incomplete = [], []
    progress = {"done": 0, "total": sum(len(group) for group in groups.values()) * repeats, "repeats": repeats}
    for keys, group in groups.items():
        attack = ProgressAttack(
            progress=progress,
            objective_target=target,
            attack_scoring_config=AttackScoringConfig(
                # Auxiliary rather than objective: there is no single "did the attack
                # win" question here, there are N independent policy questions.
                # Auxiliary scores are still written to memory against the response.
                auxiliary_scorers=[scorers[key] for key in keys] + canary_scorers,
            ),
        )
        objectives, overrides = expand_repeats(group, repeats, run_labels)
        executor_result = await AttackExecutor(max_concurrency=MAX_CONCURRENCY).execute_attack_async(
            attack=attack,
            objectives=objectives,
            **prepended,
            field_overrides=overrides,
        )
        completed += executor_result.completed_results
        incomplete += executor_result.incomplete_objectives

    memory = CentralMemory.get_memory_instance()
    findings: list[tuple[str, str, str]] = []   # (rubric, probe, rationale)
    errors: list[str] = []
    by_rubric: Counter[str] = Counter()

    for objective, exception in incomplete:
        errors.append(f"{objective[:70]!r} did not complete: {exception}")

    for result in completed:
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
        rubric_prompt = load_rubrics()["disparate_treatment"]["system_prompt"]
        verdicts = await compare_pairs(sent, by_prompt_and_repeat(completed), memory, judge, rubric_prompt, repeats)
        for pair_id, repeat, a, b, score in verdicts:
            flagged = score.get_value() is True
            label = f"pair {pair_id} ({a.variant} vs. {b.variant})"
            shown = label + (f", repeat {repeat}" if repeats > 1 else "")
            print(f"  {shown}: {'DIFFERENT TREATMENT' if flagged else 'equivalent'}")
            if flagged:
                findings.append(("disparate_treatment", label, score.score_rationale or ""))
                by_rubric["disparate_treatment"] += 1

    print("\n" + "=" * 78)
    minutes, seconds = divmod(round(time.monotonic() - started), 60)
    times = f" x {repeats} repeats" if repeats > 1 else ""
    print(f"SCAN COMPLETE - {len(sent)} probes{times}, {len(findings)} finding(s), in {minutes}m {seconds:02d}s")
    print("=" * 78)

    for rubric, count in by_rubric.most_common():
        print(f"  {rubric}: {count}")

    # With repeats, one probe can be flagged several times: print it once, with the
    # count and the first rationale.
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rubric, probe, rationale in findings:
        grouped[(rubric, probe)].append(rationale)
    for (rubric, probe), rationales in grouped.items():
        print(f"\n[FINDING: {rubric}]" + (f" flagged in {len(rationales)} of {repeats} repeats" if repeats > 1 else ""))
        print(f"  probe:     {probe[:150]}")
        print(f"  rationale: {rationales[0][:300]}")

    if ungraded:
        print(f"\n{len(ungraded)} probe(s) were not sent because no rubric in this profile grades them (see above).")

    if errors:
        print(f"\n{len(errors)} probe(s) produced no usable response:")
        for error in errors:
            print(f"  - {error}")
        print("  (a blocked or empty response is not a pass - it means nothing was tested)")

    if profile.checklist:
        print(f"\nBefore calling {profile.key!r} covered:")
        for item in profile.checklist:
            print(f"  [ ] {item}")

    write_after_run(run_labels["rt_run_id"])
    print(f"Next: python -m reporting.export_finding_report --rubric any --run-id {run_labels['rt_run_id'][:8]} --out findings/")
    print("      writes a report for each finding you agree with (--conversation-id exports one).")
    if repeats == 1:
        print("LLM output is stochastic - re-run with RT_REPEATS=5 or more before you report a finding.")
    else:
        print("The run summary's Failure rates table shows how often each probe was flagged.")

    for t in (target, judge):
        await close_target(t)

    if errors:
        return 1
    if findings and FAIL_ON_FINDING:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

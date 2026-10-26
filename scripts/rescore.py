"""Re-judge a past scan's replies with the rubrics as they are now, and show which verdicts changed.

Changing a rubric, you want to know what it does to real replies before you trust it.
This re-runs only the judge: every reply a single-turn scan stored in PyRIT's
database is judged again with the current rubric file, the way the scan judged it,
and the report says which verdicts changed, with the old and new rationale side by
side. Nothing is sent to the system under test, and the run's own scores - the
evidence its finding reports are built from - are left as they were.

    # the newest scan, every rubric
    python -m scripts.rescore

    # one run (the id the scan printed; the first 8 characters are enough), one rubric
    python -m scripts.rescore --run 937e73ef --rubric ungrounded_claim

    # only some probes or pairs
    python -m scripts.rescore --run 937e73ef --probes fabricated_form_number,dialect_01

It writes findings/<date>_rescore_<run id>.md. Each reply costs one judge call per
rubric, and each matched pair one more, with RT_JUDGE_PROVIDER / RT_JUDGE_MODEL as set
now. RT_MAX_CALLS applies.

READ IT AGAINST THE NOISE. The judge does not give the same verdict every time, so
some verdicts change even with the rubric untouched. Re-score once before you edit
anything: what changes then is the noise, and an edit's effect is what changes beyond
it. When the rubric file is the one the run was judged with, the report says so.

What is judged again, and how:
  - Each rubric verdict, with the reply, the probe as it was sent, the system prompt
    the run showed its judge, and the reply's trace. The pass criteria and reference
    facts come from the suite files as they are now; if a probe's text has changed
    since the run, the report says so.
  - A rubric that now grades a probe's suite but did not then is judged too, as new.
  - Each matched pair, in the order the judge read it in the run. Control pairs are
    counted on their own, as in the run summary.
  - Not the code checks (canary, planted records, credentials, ...): they are code,
    and give the same answer every time.

Multi-turn campaigns are not covered: their judge steers the attack while it runs, so
judging the last reply again does not repeat the run. Preview their judge with
--show-judge-prompt instead.
"""

import argparse
import asyncio
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Fix for corporate TLS-inspecting proxies
import truststore
truststore.inject_into_ssl()

from pyrit.memory import CentralMemory
from pyrit.setup import SQLITE, initialize_pyrit_async

from harnesses.pyrit_campaigns import single_turn_scan as scan
from harnesses.pyrit_campaigns.calibration import CALIBRATION_LABEL
from harnesses.pyrit_campaigns.code_checks import CODE_CHECKS
from harnesses.pyrit_campaigns.profiles import PAIR_RUBRICS, dataset_path
from harnesses.pyrit_campaigns.rubrics import describe_version, require_valid, version_labels
from harnesses.pyrit_campaigns.target_factory import (
    build_scoring_target,
    check_models,
    close_target,
    model_name,
    unverified_rubric,
)
from scripts.export_finding_report import select_run
from scripts.run_summary import DEFAULT_OUT, cell, fence
from scripts.scrub import encodings_from_labels, scrub_report

PAIR_RUBRIC = next(iter(PAIR_RUBRICS))


@dataclass
class Rejudged:
    """One verdict judged again: a rubric on one reply, or the pair rubric on one pair."""

    rubric: str
    shown: str                      # the probe or pair, as the report names it
    reply: str                      # what was judged; both replies for a pair
    old: bool | None = None         # None: not judged in the run
    old_rationale: str = ""
    new: bool | None = None         # None: the judge gave no verdict (see error)
    new_rationale: str = ""
    error: str = ""
    control: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.old is not None and self.new is not None and self.old != self.new


def verdict(value: bool | None) -> str:
    return {True: "true (flagged)", False: "false (passed)", None: "none"}[value]


def last_reply(messages):
    """The run's reply to the probe: the last assistant piece, or None."""
    for message in reversed(messages):
        for piece in message.message_pieces:
            if piece.role == "assistant":
                return piece
    return None


def system_prompt_of(messages, labels: dict) -> tuple[str | None, str]:
    """The system prompt the run's judge was shown, and a note when it cannot be the same."""
    for message in messages:
        for piece in message.message_pieces:
            if piece.role == "system":
                return piece.converted_value, ""
    app_file = labels.get("rt_app_prompt_file")
    if app_file:
        path = Path(app_file)
        if path.is_file():
            return path.read_text(encoding="utf-8"), f"the app's prompt was read again from {app_file}"
        return None, f"the app's prompt file {app_file} is gone, so the judge is told there was none"
    return None, ""


def judge_verdicts(scores) -> dict[str, tuple[bool, str]]:
    """The run's rubric verdicts on a reply: rubric -> (value, rationale). Not the code checks or the pair rubric."""
    verdicts = {}
    for score in scores:
        for category in score.score_category or []:
            # An unverified flag is still the judge's verdict on its rubric: the judge is what is re-run.
            category = unverified_rubric(category) or category
            if category not in CODE_CHECKS and category != PAIR_RUBRIC:
                verdicts[category] = (score.get_value() is True, score.score_rationale or "")
    return verdicts


def wanted(token_sets: set[str] | None, *names: str) -> bool:
    return token_sets is None or any(name and name in token_sets for name in names)


def plan(results, memory, rubrics: dict, only_rubrics: set[str] | None, only_probes: set[str] | None):
    """What to judge again: (single-reply tasks, pair tasks, things left out)."""
    probes = {(p.dataset, p.name): p for p in scan.load_probes()}
    order = {key: i for i, key in enumerate(probes)}
    singles, pairs, skipped = [], {}, []
    # Probes from a suite that is not here at all, by suite: said once per suite, not once per probe.
    missing_suites: Counter[str] = Counter()

    for result in results:
        labels = result.labels or {}
        dataset, name = labels.get("rt_dataset", ""), labels.get("rt_probe", "")
        repeat, encoding = int(labels.get("rt_repeat") or 1), labels.get("rt_encoding", "")
        shown = f"{dataset} / {name}"
        if encoding:
            shown = f"[{encoding}] {shown}"
        if int(labels.get("rt_repeats") or 1) > 1:
            shown += f" (repeat {repeat})"
        probe = probes.get((dataset, name))
        if probe is None:
            if not dataset_path(dataset).exists():
                missing_suites[dataset] += 1
            else:
                skipped.append(f"{shown}: no longer in {dataset}, so its pass criteria and facts are unknown.")
            continue
        messages = memory.get_conversation_messages(conversation_id=result.conversation_id)
        reply = last_reply(messages)
        if reply is None or getattr(reply, "response_error", "none") != "none":
            skipped.append(f"{shown}: the run got no usable reply.")
            continue
        scores = memory.get_prompt_scores(conversation_id=result.conversation_id)
        notes = []
        if probe.prompt != result.objective:
            notes.append("the probe's text has changed since the run; it is judged as sent, with today's criteria")
        system_prompt, prompt_note = system_prompt_of(messages, labels)
        if prompt_note:
            notes.append(prompt_note)

        # Each encoding pass compares its own pairs, as the scan did.
        if probe.pair_id and wanted(only_rubrics, PAIR_RUBRIC) and wanted(only_probes, name, probe.pair_id, dataset):
            pairs.setdefault((probe.pair_id, encoding, repeat), []).append((order[(dataset, name)], probe, reply, scores))

        if not wanted(only_probes, name, probe.pair_id, dataset):
            continue
        old = judge_verdicts(scores)
        now = {key for key, rubric in rubrics.items() if dataset in rubric.get("grades", []) and key not in PAIR_RUBRICS}
        for key in sorted(set(old) | now):
            if not wanted(only_rubrics, key):
                continue
            if key not in rubrics:
                skipped.append(f"{shown}: rubric {key!r} is no longer in the rubric file.")
                continue
            singles.append((key, probe, result.objective, reply, system_prompt, old.get(key), shown, notes))

    pair_tasks = []
    for (pair_id, encoding, repeat), members in pairs.items():
        tag = f" [{encoding}]" if encoding else ""
        if PAIR_RUBRIC not in rubrics:
            skipped.append(f"pair {pair_id}{tag}: the pair rubric is no longer in the rubric file.")
            continue
        if len(members) != 2:
            skipped.append(f"pair {pair_id}{tag} (repeat {repeat}): only {len(members)} half had a reply.")
            continue
        (_, a, reply_a, _), (_, b, reply_b, scores_b) = sorted(members, key=lambda m: m[0])
        old = next((s for s in scores_b if PAIR_RUBRIC in (s.score_category or [])), None)
        first_variant = (old.score_metadata or {}).get("first") if old else None
        first = b if first_variant == b.variant and a.variant != b.variant else a
        pair_tasks.append((pair_id, encoding, repeat, a, b, first, reply_a, reply_b, old))

    reasons = [
        f"{count} probe(s) from {dataset}: that suite is not here (no attack_suites/*/{dataset}.yaml), so their "
        "pass criteria and facts are unknown. Re-score from the branch or copy the run was made with."
        for dataset, count in sorted(missing_suites.items())
    ]
    # A rubric asked for by name that judges nothing in this run: say what it judges, and what the run sent.
    judged = {task[0] for task in singles} | ({PAIR_RUBRIC} if pair_tasks else set())
    sent = Counter((r.labels or {}).get("rt_dataset", "") for r in results)
    for key in sorted((only_rubrics or set()) - judged):
        grades = ", ".join(rubrics[key].get("grades", [])) or "no suite"
        reasons.append(
            f"No probe in this run is judged by {key}: it grades {grades}, and the run sent "
            + ", ".join(f"{dataset} ({n})" for dataset, n in sorted(sent.items())) + "."
        )
    return singles, pair_tasks, reasons + skipped


async def rejudge(singles, pair_tasks, rubrics: dict, judge) -> list[Rejudged]:
    """Judge every task again, MAX_CONCURRENCY at a time. A failed call is reported, not raised."""
    gate = asyncio.Semaphore(scan.MAX_CONCURRENCY)

    async def one(key, probe, objective, reply, system_prompt, old, shown, notes) -> Rejudged:
        item = Rejudged(key, shown, reply.converted_value or "", notes=list(notes))
        if old is not None:
            item.old, item.old_rationale = old
        scorer = scan.RubricScorer(
            rubric_prompt=scan.rubric_prompt(rubrics[key], system_prompt),
            judge=judge,
            category=key,
            judge_context={objective: probe.judge_slots()},
        )
        async with gate:
            try:
                # The scorer's own judging, without saving the score: the run's scores stay as they were.
                [score] = await scorer._score_piece_async(reply, objective=objective)
                item.new, item.new_rationale = score.get_value() is True, score.score_rationale or ""
            except Exception as e:
                item.error = f"{type(e).__name__}: {str(e)[:200]}"
        return item

    async def pair(pair_id, encoding, repeat, a, b, first, reply_a, reply_b, old) -> Rejudged:
        reply_first, reply_second = (reply_b, reply_a) if first is b else (reply_a, reply_b)
        shown = f"pair {pair_id}" + (f" [{encoding}]" if encoding else "") + f" ({a.variant} vs. {b.variant})" + (f", repeat {repeat}" if repeat > 1 else "")
        both = (f"{first.variant}, read first:\n{reply_first.converted_value or ''}\n\n"
                f"{(b if first is a else a).variant}, read second:\n{reply_second.converted_value or ''}")
        item = Rejudged(PAIR_RUBRIC, shown, both, control=a.control and b.control)
        if old is not None:
            item.old, item.old_rationale = old.get_value() is True, old.score_rationale or ""
        filled = scan.pair_prompt(
            rubrics[PAIR_RUBRIC]["system_prompt"], a, b, first,
            reply_first.converted_value or "", reply_second.converted_value or "",
        )
        scorer = scan.PairScorer(
            system_prompt_format_string=scan._escape_braces(filled),
            chat_target=judge,
            category=PAIR_RUBRIC,
            pair_metadata={"first": first.variant, "second": (b if first is a else a).variant},
        )
        async with gate:
            try:
                [score] = await scorer._score_piece_async(reply_b)
                item.new, item.new_rationale = score.get_value() is True, score.score_rationale or ""
            except Exception as e:
                item.error = f"{type(e).__name__}: {str(e)[:200]}"
        return item

    return list(await asyncio.gather(*(one(*t) for t in singles), *(pair(*t) for t in pair_tasks)))


def build_report(run_id: str, labels: dict, judge_shown: str, items: list[Rejudged], skipped: list[str]) -> str:
    now = version_labels()
    same_rubrics = labels.get("rt_rubric_sha") == now["rt_rubric_sha"]
    lines = [
        f"# Re-score: run {run_id[:8]}",
        "",
        f"Generated {date.today().isoformat()} by `scripts/rescore.py`. The replies this run stored were judged "
        "again with the rubrics as they are now. Nothing was sent to the system under test, and the run's own "
        "scores are unchanged.",
        "",
        "## At a glance",
        "",
        "| | |",
        "|---|---|",
        f"| Run id | `{run_id}` |",
        f"| Judge in the run | {labels.get('rt_judge') or 'not recorded'} (RT_JUDGE_PROVIDER={labels.get('rt_judge_provider') or 'not recorded'}) |",
        f"| Judge now | {judge_shown} |",
        f"| Rubrics in the run | {describe_version(labels)} |",
        f"| Rubrics now | {describe_version(now)} |",
        f"| Judged again | {len(items)} verdict(s), {sum(i.changed for i in items)} changed |",
        "",
    ]
    if same_rubrics:
        lines += [
            "The rubric file is the one the run was judged with, so every change below is the judge varying, not "
            "an edit. That is the noise to read later re-scores against.",
            "",
        ]
    if labels.get("rt_judge") and labels.get("rt_judge") not in judge_shown:
        lines += [
            "**The judge is not the one the run used,** so a change below may come from the judge rather than the "
            "rubric. Set RT_JUDGE_PROVIDER and RT_JUDGE_MODEL as in the run to compare rubrics alone.",
            "",
        ]

    lines += [
        "## By rubric",
        "",
        "| Rubric | Judged again | Unchanged | true → false | false → true | New | No verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[Rejudged]] = {}
    for item in items:
        groups.setdefault(item.rubric + (" (control pairs)" if item.control else ""), []).append(item)
    for name, group in sorted(groups.items()):
        unchanged = sum(i.old is not None and i.new is not None and i.old == i.new for i in group)
        lines.append(
            f"| {name} | {len(group)} | {unchanged} | {sum(i.changed and i.old for i in group)} | "
            f"{sum(i.changed and not i.old for i in group)} | {sum(i.old is None and i.new is not None for i in group)} | "
            f"{sum(i.new is None for i in group)} |"
        )
    if any(i.control for i in items):
        lines += ["", "Control pairs send the same text twice, so a control judged different is noise, never a finding."]
    lines.append("")

    def detail(item: Rejudged) -> list[str]:
        out = [f"### {item.shown} - {item.rubric}: {verdict(item.old)} → {verdict(item.new)}", ""]
        out += [f"- Note: {note}." for note in item.notes]
        out += ["", "Reply:", "", fence(item.reply), ""]
        if item.old is not None:
            out += [f"Rationale in the run: {item.old_rationale}", ""]
        out += [f"Rationale now: {item.new_rationale}" if not item.error else f"No verdict: {item.error}", ""]
        return out

    changed = [i for i in items if i.changed]
    lines += ["## Changed verdicts", ""]
    if not changed:
        lines += ["None.", ""]
    for item in changed:
        lines += detail(item)

    new = [i for i in items if i.old is None and i.new is not None]
    if new:
        lines += ["## New verdicts", "", "From rubrics that grade these probes now but did not in the run.", ""]
        lines += ["| Probe | Rubric | Verdict | Rationale |", "|---|---|---|---|"]
        lines += [f"| {cell(i.shown)} | {i.rubric} | {verdict(i.new)} | {cell(i.new_rationale, 160)} |" for i in new]
        lines.append("")

    failed = [i for i in items if i.new is None]
    if failed or skipped:
        lines += ["## Not judged again", ""]
        lines += [f"- {i.shown} - {i.rubric}: {i.error}" for i in failed]
        lines += [f"- {reason}" for reason in skipped]
        lines.append("")
    return "\n".join(lines)


def parse_tokens(raw: str | None) -> set[str] | None:
    return {t.strip() for t in raw.split(",") if t.strip()} if raw else None


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="The scan to re-score. The first 8 characters are enough. Default: the newest run.")
    parser.add_argument("--rubric", help="Only these rubrics, comma-separated, e.g. ungrounded_claim,disparate_treatment")
    parser.add_argument("--probes", help="Only these probes, pairs, or datasets, by name, comma-separated")
    parser.add_argument("--out", type=Path, help="Output directory (default: findings/)")
    parser.add_argument(
        "--no-scrub",
        action="store_true",
        help="Write the report without running the Rampart PII scrubber (or set RT_SCRUB=0)",
    )
    args = parser.parse_args(argv)

    rubrics = scan.load_rubrics()
    require_valid(rubrics)
    only_rubrics = parse_tokens(args.rubric)
    if only_rubrics and only_rubrics - set(rubrics):
        raise SystemExit(f"No rubric named {', '.join(sorted(only_rubrics - set(rubrics)))}. They are: {', '.join(rubrics)}.")

    # Must match the memory_db_type the scan used, or the run will not be here.
    await initialize_pyrit_async(memory_db_type=SQLITE)
    memory = CentralMemory.get_memory_instance()
    results, run_id = select_run(memory.get_attack_results(), args.run, latest=not args.run)
    results = [r for r in results if not (r.labels or {}).get(CALIBRATION_LABEL)]
    if not results:
        raise SystemExit(f"Run {run_id[:8]} has no probe results.")
    labels = results[0].labels or {}
    if labels.get("rt_campaign") != "single_turn_scan":
        raise SystemExit(
            f"Run {run_id[:8]} is a {labels.get('rt_campaign', 'unlabeled')} run. Re-scoring covers single-turn scans: "
            "a multi-turn judge steers its attack, so judging the last reply again does not repeat the run. "
            "Preview its judge with --show-judge-prompt."
        )

    singles, pair_tasks, skipped = plan(results, memory, rubrics, only_rubrics, parse_tokens(args.probes))
    calls = len(singles) + len(pair_tasks)
    if not calls:
        reasons = "".join(f"\n  - {reason}" for reason in skipped)
        raise SystemExit(f"Nothing to judge again in run {run_id[:8]}" + (f":{reasons}" if reasons else "."))
    print(f"Run {run_id[:8]}: {len(singles)} rubric verdict(s) and {len(pair_tasks)} pair(s) to judge again, "
          f"{calls} judge call(s). Nothing is sent to the system under test.")
    limit = scan.get_max_calls()
    if limit and calls > limit:
        raise SystemExit(f"That is over RT_MAX_CALLS={limit}. Narrow it with --rubric or --probes, or raise the limit.")

    judge = build_scoring_target()
    await check_models(judge=judge)
    judge_shown = f"{model_name(judge)} (RT_JUDGE_PROVIDER={scan.resolve_provider('judge')})"
    try:
        items = await rejudge(singles, pair_tasks, rubrics, judge)
    finally:
        await close_target(judge)

    out = args.out or DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{date.today().isoformat()}_rescore_{run_id[:8]}.md"
    encodings = encodings_from_labels(*(r.labels for r in results))
    report = scrub_report(
        build_report(run_id, labels, judge_shown, items, skipped),
        no_scrub=args.no_scrub,
        encodings=encodings,
    )
    path.write_text(report, encoding="utf-8")

    changed = [i for i in items if i.changed]
    print(f"{len(changed)} of {len(items)} verdict(s) changed.")
    for item in changed:
        print(f"  {item.shown} - {item.rubric}: {verdict(item.old)} -> {verdict(item.new)}")
    if labels.get("rt_rubric_sha") == version_labels()["rt_rubric_sha"]:
        print("The rubrics are the ones the run used, so these changes are the judge's noise.")
    print(f"\nIn full: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

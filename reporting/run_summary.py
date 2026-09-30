"""Write one Markdown summary of a whole campaign run: every result, in full.

The scan's console output cuts replies at 200 characters and rationales at 300,
and finding reports cover only what a judge flagged. Neither tells you why the
passes passed. This file lists every probe of one run with its full reply, each
judge's verdict and full rationale, and every matched-pair comparison, so a pass
can be checked as closely as a finding.

The scan and Crescendo write it themselves at the end of every run. Run this to
write one again, or for a run from before they did:

    # the newest run in the database
    python -m reporting.run_summary

    # one run - the id the campaign printed (the first 8 characters are enough)
    python -m reporting.run_summary --run-id 1a2b3c4d

    # somewhere other than findings/
    python -m reporting.run_summary --out reviews/

It writes a file to findings/ named for when the run started, the campaign, the
profile, and the run id, e.g. 2026-09-30_0951_scan_public_conversational_7f264395.md,
so the folder lists runs oldest to newest. The top section is written for program
staff; the per-result detail below it is for whoever reviews the verdicts.

    !! The summary holds full transcripts, like the finding reports. It inherits
    !! the classification of whatever was sent - findings/ is gitignored for this.
"""

import argparse
import re
from collections import Counter, defaultdict
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

from pyrit_campaigns.profiles import PROFILES
from pyrit_campaigns.system_prompt import CANARY_CHECK
from reporting.export_finding_report import (
    flagged_rubrics,
    format_scores,
    format_transcript,
    load_rubrics,
    select_run,
)

# Graded by comparing two replies, so its verdict belongs to the pair, not to the
# reply it happens to be recorded against.
PAIR_RUBRIC = "disparate_treatment"

# The repo's findings/ folder, wherever the command is run from.
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "findings"

FLAGGED = "FLAGGED"
PASSED = "passed"
NO_RESPONSE = "no usable response"
ERROR = "error"
NOT_SCORED = "not scored"
PAIR_ONLY = "pair comparison only"
ACHIEVED = "FLAGGED (objective achieved)"
HELD = "held"


def response_error(messages) -> str | None:
    """Why the conversation has no usable reply, or None if it has one."""
    replies = [p for m in messages for p in m.message_pieces if p.role == "assistant"]
    if not replies:
        return "no response"
    error = getattr(replies[-1], "response_error", "none")
    return None if error in (None, "none") else str(error)


def classify(result, messages, scores, graded_by: list[str]) -> str:
    """One verdict per result, in the words the summary uses."""
    if (result.labels or {}).get("rt_campaign") == "multi_turn_crescendo":
        # Crescendo's outcome is the verdict: success means the attack worked.
        if result.outcome.value == "success":
            return ACHIEVED
        if result.outcome.value == "failure":
            return HELD
    if result.outcome.value == "error":
        # The attack stopped partway, often in the judge after the reply came back.
        return ERROR
    if response_error(messages):
        return NO_RESPONSE
    if (result.labels or {}).get("rt_campaign") == "multi_turn_crescendo":
        return result.outcome.value
    if [r for r in flagged_rubrics(scores) if r != PAIR_RUBRIC]:
        return FLAGGED
    if graded_by:
        return PASSED
    if (result.labels or {}).get("rt_pair_id"):
        return PAIR_ONLY
    return NOT_SCORED


def rubrics_scored(scores) -> list[str]:
    """The rubric judges that returned a verdict on this reply.

    Not the pair rubric, whose verdict is the pair's, and not the canary check, which
    checks every reply: a reply only it checked was not graded against any policy.
    """
    known = set(load_rubrics())
    return sorted({c for s in scores for c in (s.score_category or [])} & known - {PAIR_RUBRIC, CANARY_CHECK})


def canary_verdict(scores) -> bool | None:
    """True if the reply leaked the canary, False if checked and clean, None if not checked."""
    checks = [s for s in scores if CANARY_CHECK in (s.score_category or [])]
    return any(s.get_value() is True for s in checks) if checks else None


def without_system(messages) -> list:
    """The conversation minus its system prompt, which the summary shows once rather than per result."""
    return [m for m in messages if any(p.role != "system" for p in m.message_pieces)]


def system_prompt_text(entries) -> str | None:
    for entry in entries:
        for message in entry["messages"]:
            for piece in message.message_pieces:
                if piece.role == "system":
                    return piece.converted_value
    return None


def fence(text: str) -> str:
    """A code fence longer than any run of backticks in the text, so replies with code stay inside it."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{text}\n{ticks}"


def cell(text: str, limit: int = 80) -> str:
    """Text safe for one Markdown table cell."""
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def repeat_of(result) -> int:
    """Which repeat of its probe a result is, from 1. Runs from before repeats are repeat 1."""
    return int((result.labels or {}).get("rt_repeat") or 1)


def zero_flag_bound(n: int) -> float:
    """The highest true failure rate still consistent (at 95%) with 0 flags in n tries.

    Solves (1 - p) ** n = 0.05 for p: a probe that fails at this rate would pass all
    n tries only 5% of the time. Roughly 3/n for larger n (the "rule of three").
    """
    return 1 - 0.05 ** (1 / n)


def failure_rates(entries) -> list[dict]:
    """One row per probe: how many of its repeats were flagged, passed, or have no verdict."""
    rows: dict[tuple[str, str], dict] = {}
    for e in entries:
        result_labels = e["result"].labels or {}
        key = (result_labels.get("rt_dataset", "-"), result_labels.get("rt_probe") or cell(e["result"].objective, 60))
        row = rows.setdefault(key, {"dataset": key[0], "probe": key[1], "flagged": 0, "passed": 0, "no_verdict": 0})
        if e["verdict"] == FLAGGED:
            row["flagged"] += 1
        elif e["verdict"] == PASSED:
            row["passed"] += 1
        elif e["verdict"] != PAIR_ONLY:
            row["no_verdict"] += 1
    # A probe sent only to be compared has no verdict of its own; its pair's row says how it did.
    return [r for r in rows.values() if r["flagged"] or r["passed"] or r["no_verdict"]]


def pair_verdicts(entries) -> list[dict]:
    """Each matched pair in the run, per repeat, with the judge's comparison if one was recorded."""
    pairs = defaultdict(list)
    for entry in entries:
        pair_id = (entry["result"].labels or {}).get("rt_pair_id")
        if pair_id:
            pairs[(pair_id, repeat_of(entry["result"]))].append(entry)
    out = []
    for (pair_id, repeat), members in sorted(pairs.items()):
        score = next(
            (s for m in members for s in m["scores"] if PAIR_RUBRIC in (s.score_category or [])),
            None,
        )
        if score is None:
            verdict = "not compared (a reply was missing, or the profile does not compare pairs)"
        elif score.get_value() is True:
            verdict = "DIFFERENT TREATMENT"
        else:
            verdict = "equivalent"
        out.append({"pair_id": pair_id, "repeat": repeat, "members": members, "score": score, "verdict": verdict})
    return out


def build_summary(results, memory, run_id: str) -> str:
    results = sorted(
        results,
        key=lambda r: ((r.labels or {}).get("rt_dataset", ""), (r.labels or {}).get("rt_probe", ""), r.objective, repeat_of(r)),
    )
    rubrics = load_rubrics()
    entries = []
    for number, result in enumerate(results, 1):
        messages = memory.get_conversation_messages(conversation_id=result.conversation_id)
        scores = memory.get_prompt_scores(conversation_id=result.conversation_id)
        graded_by = rubrics_scored(scores)
        entries.append({
            "number": number,
            "result": result,
            "messages": messages,
            "scores": scores,
            "graded_by": graded_by,
            "canary": canary_verdict(scores),
            "verdict": classify(result, messages, scores, graded_by),
        })
    pairs = pair_verdicts(entries)
    leaks = sum(e["canary"] is True for e in entries)
    canary_checked = sum(e["canary"] is not None for e in entries)
    prompt_text = system_prompt_text(entries)

    labels = results[0].labels or {}
    campaign = labels.get("rt_campaign", "unlabeled")
    repeats = int(labels.get("rt_repeats") or 1)

    def with_repeat(name: str, result) -> str:
        return f"{name} (repeat {repeat_of(result)})" if repeats > 1 else name

    counts = Counter(e["verdict"] for e in entries)
    flagged = counts[FLAGGED] + counts[ACHIEVED]
    pairs_flagged = sum(p["verdict"] == "DIFFERENT TREATMENT" for p in pairs)
    pairs_compared = sum(p["score"] is not None for p in pairs)
    started = min((r.timestamp for r in results if r.timestamp), default=None)

    lines = [f"# Run summary: {campaign}, run {run_id[:8]}", ""]
    lines += [
        f"Generated {date.today().isoformat()} by `reporting/run_summary.py` from the PyRIT database. "
        "Every result of the run is below, in full.",
        "",
        "## At a glance",
        "",
        "| | |",
        "|---|---|",
        f"| Run id | `{run_id}` |",
        f"| Campaign | {campaign} |",
    ]
    profile = PROFILES.get(labels.get("rt_profile", ""))
    if profile:
        lines.append(f"| Profile | {profile.key} - {cell(profile.description, 200)} |")
    provider = labels.get("rt_provider")

    def recorded(value) -> str:
        # Runs made before a label existed do not have it.
        return value or "not recorded"

    lines.append(
        f"| Model under test | {recorded(labels.get('rt_target'))} (RT_PROVIDER={recorded(provider)})"
        + ("" if provider == "app" else " - the model on its own, not a deployed application")
        + " |"
    )
    system_prompt = labels.get("rt_system_prompt")
    if system_prompt == "none":
        lines.append("| System prompt | none (RT_SYSTEM_PROMPT_FILE=none) |")
    elif system_prompt:
        how = (
            "chosen by RT_SYSTEM_PROMPT_FILE"
            if labels.get("rt_system_prompt_chosen_by") == "RT_SYSTEM_PROMPT_FILE"
            else "the profile's stand-in"
        )
        lines.append(
            f"| System prompt | `{system_prompt}`, {how} ([text](#system-prompt)); "
            f"canary token `{labels.get('rt_prompt_canary') or 'not recorded'}` |"
        )
    elif campaign == "single_turn_scan":
        lines.append("| System prompt | not recorded (the scan sent none before it recorded this) |")
    lines.append(f"| Judge | {recorded(labels.get('rt_judge'))} (RT_JUDGE_PROVIDER={recorded(labels.get('rt_judge_provider'))}) |")
    if labels.get("rt_adversarial"):
        lines.append(f"| Attacker | {labels['rt_adversarial']} (RT_ADVERSARIAL_PROVIDER={recorded(labels.get('rt_adversarial_provider'))}) |")
    if started:
        local = started.astimezone()
        lines.append(f"| Run started | {local:%Y-%m-%d %H:%M} {local.tzname()} ({started:%H:%M} UTC) |")
    if repeats > 1:
        lines.append(f"| Repeats | every probe sent {repeats} times (RT_REPEATS), each in a new conversation |")
    result_counts = ", ".join(f"{n} {v}" for v, n in counts.most_common())
    lines.append(f"| Results | {len(entries)}: {result_counts} |")
    if pairs:
        what = "pair comparisons (one per pair per repeat)" if repeats > 1 else "matched pairs"
        lines.append(f"| Matched pairs | {len(pairs)} {what}: {pairs_compared} compared, {pairs_flagged} flagged as different treatment |")
    lines.append("")

    lines += ["## What this means", ""]
    if campaign == "multi_turn_crescendo":
        lines.append(f"The attack reached its objective in **{flagged} of {len(entries)}** conversations in this run.")
    else:
        lines.append(
            ("The judge model and the canary token check" if canary_checked else "The judge model")
            + f" flagged **{flagged} of {len(entries)}** results"
            + (f" and **{pairs_flagged} of {pairs_compared}** matched pairs" if pairs_compared else "")
            + " in this run."
        )
    lines += [
        "",
        "- **A flag is a lead, not a finding.** It is the judge model's reading of a rubric in "
        "`judges/state_policy_rubric.yaml`. Someone has to read the reply and the judge's reasoning below "
        "and agree before it becomes a finding.",
    ]
    if repeats > 1:
        lines.append(
            f"- **Each probe ran {repeats} times.** [Failure rates](#failure-rates) shows how often each one was "
            f"flagged. A probe never flagged is not proven safe: 0 of {repeats} is still consistent with it "
            f"failing up to {zero_flag_bound(repeats):.0%} of the time. More repeats shrink that bound."
        )
    else:
        lines.append(
            "- **A pass is one sample.** Model output varies from run to run. A probe that passed once can "
            "fail on the next run, so re-run with RT_REPEATS=5 or more before you rely on either result."
        )
    if leaks:
        lines.append(
            f"- **{leaks} reply(ies) leaked the system prompt.** Each contains the canary token, a random code "
            "planted only in the system prompt, so this was found by matching the code, not by a judge. "
            "It counts as a flag whatever the judges said; their verdicts are beside it below."
        )
    if counts[ERROR]:
        lines.append(
            f"- **{counts[ERROR]} result(s) stopped with an error** and have no verdict. The reason "
            "PyRIT recorded is with each one below; a failed judge call is the usual cause."
        )
    if counts[NO_RESPONSE]:
        lines.append(
            f"- **{counts[NO_RESPONSE]} result(s) tested nothing.** An error, block, or empty reply is not a pass."
        )
    if counts[NOT_SCORED]:
        lines.append(
            f"- **{counts[NOT_SCORED]} result(s) have no verdict.** The reply came back but no judge score "
            "was recorded; the judge call may have failed. Check the campaign's console output."
        )
    if provider != "app":
        if system_prompt and system_prompt != "none" and labels.get("rt_system_prompt_chosen_by") == "RT_SYSTEM_PROMPT_FILE":
            lines.append(
                f"- **This tested a model with the system prompt in `{system_prompt}`, not your deployed "
                "application.** If that is your application's prompt, this tested your instructions on this "
                "model. Your retrieval index and tool permissions were still not in the loop."
            )
        elif system_prompt and system_prompt != "none":
            lines.append(
                "- **This tested a model with a stand-in system prompt, not your deployed application.** "
                f"`{system_prompt}` is a generic prompt for this kind of deployment. It shows how the model "
                "holds rules like yours, not how it holds yours: set RT_SYSTEM_PROMPT_FILE to your "
                "application's prompt for that. Your retrieval index and tool permissions were not in the loop."
            )
        else:
            lines.append(
                "- **This tested a bare model, not your deployed application.** It had no system prompt, and "
                "your retrieval index and tool permissions were not in the loop. They are where most "
                "of a deployment's risk lives."
            )
    lines.append("")

    flagged_by = Counter(r for e in entries for r in flagged_rubrics(e["scores"]) if r != PAIR_RUBRIC)
    graded = Counter(r for e in entries for r in e["graded_by"])
    if graded or pairs_compared or canary_checked:
        lines += ["## By rubric", "", "| Rubric | Flagged | Graded | What it checks |", "|---|---|---|---|"]
        for rubric in sorted(graded):
            description = cell(rubrics.get(rubric, {}).get("description", ""), 120)
            lines.append(f"| {rubric} | {flagged_by[rubric]} | {graded[rubric]} | {description} |")
        if canary_checked:
            lines.append(
                f"| {CANARY_CHECK} (canary token check) | {leaks} | {canary_checked} "
                "| Whether the reply contains the canary token planted in the system prompt. A match, not a judge. |"
            )
        if pairs_compared:
            description = cell(rubrics.get(PAIR_RUBRIC, {}).get("description", ""), 120)
            lines.append(f"| {PAIR_RUBRIC} (pairs) | {pairs_flagged} | {pairs_compared} | {description} |")
        lines.append("")

    if repeats > 1:
        lines += [
            '<a id="failure-rates"></a>',
            "## Failure rates",
            "",
            f"How often each probe was flagged across its {repeats} repeats. \"Without a verdict\" counts "
            "errors, blocked or empty replies, and missing judge scores; those repeats tested nothing, "
            "so they are left out of the rate.",
            "",
            "| Probe | Dataset | Flagged | Without a verdict | Reading |",
            "|---|---|---|---|---|",
        ]
        for row in failure_rates(entries):
            tested = row["flagged"] + row["passed"]
            if not tested:
                reading = "nothing tested"
            elif not row["flagged"]:
                reading = f"never flagged; could still fail up to {zero_flag_bound(tested):.0%} of the time"
            elif row["flagged"] == tested:
                reading = "flagged every time"
            else:
                reading = "flagged some of the time"
            lines.append(
                f"| {row['probe']} | {row['dataset']} | {row['flagged']} of {tested} | {row['no_verdict']} | {reading} |"
            )
        if pairs:
            by_pair = defaultdict(lambda: [0, 0])
            for p in pairs:
                if p["score"] is not None:
                    by_pair[p["pair_id"]][1] += 1
                    by_pair[p["pair_id"]][0] += p["verdict"] == "DIFFERENT TREATMENT"
            lines += ["", "| Pair | Different treatment | Compared |", "|---|---|---|"]
            for pair_id in sorted({p["pair_id"] for p in pairs}):
                different, compared = by_pair[pair_id]
                lines.append(f"| {pair_id} | {different} of {compared} | {compared} of {repeats} |")
        lines.append("")

    lines += ["## Every result", "", "| # | Verdict | Dataset | Probe | Graded by |", "|---|---|---|---|---|"]
    for e in entries:
        result_labels = e["result"].labels or {}
        probe = with_repeat(result_labels.get("rt_probe") or cell(e["result"].objective, 60), e["result"])
        checks = e["graded_by"] + {True: ["canary token LEAKED"], False: ["canary token"], None: []}[e["canary"]]
        lines.append(
            f"| [{e['number']}](#result-{e['number']}) | {e['verdict']} | {result_labels.get('rt_dataset', '-')} "
            f"| {probe} | {', '.join(checks) or '-'} |"
        )
    lines.append("")

    if pairs:
        lines += ["## Matched pairs", ""]
        lines.append(
            "Each pair asks the same question twice, changing one attribute (dialect, name, zip code, "
            "language, or a disclosed disability). The judge compares the two replies."
        )
        lines.append("")
        for p in pairs:
            members = ", ".join(
                f"[{m['number']}](#result-{m['number']}) ({(m['result'].labels or {}).get('rt_variant') or '?'})"
                for m in p["members"]
            )
            repeat = f", repeat {p['repeat']}" if repeats > 1 else ""
            lines.append(f"### Pair {p['pair_id']}{repeat}: {p['verdict']}")
            lines.append("")
            lines.append(f"Results {members}.")
            if p["score"] is not None and p["score"].score_rationale:
                lines += ["", f"Judge's reasoning: {p['score'].score_rationale}"]
            lines.append("")

    if prompt_text:
        lines += [
            '<a id="system-prompt"></a>',
            "## System prompt",
            "",
            "Sent ahead of every probe, as the start of each conversation. It is left out of the "
            "transcripts below.",
            "",
            fence(prompt_text),
            "",
        ]

    lines += ["## Results in full", ""]
    for e in entries:
        result = e["result"]
        result_labels = result.labels or {}
        name = with_repeat(result_labels.get("rt_probe") or cell(result.objective, 60), result)
        lines += [f'<a id="result-{e["number"]}"></a>', f"### {e['number']}. {e['verdict']}: {name}", ""]
        details = []
        if result_labels.get("rt_dataset"):
            details.append(f"dataset `{result_labels['rt_dataset']}`")
        if result_labels.get("rt_pair_id"):
            details.append(f"pair `{result_labels['rt_pair_id']}`, variant `{result_labels.get('rt_variant')}`")
        if campaign == "multi_turn_crescendo":
            details.append(f"{result.executed_turns} turn(s), outcome `{result.outcome.value}`")
        if result.outcome_reason and (campaign == "multi_turn_crescendo" or result.outcome.value == "error"):
            details.append(f"reason: {result.outcome_reason}")
        details.append(f"conversation `{result.conversation_id}`")
        lines += ["- " + d for d in details]
        error = response_error(e["messages"])
        if error:
            lines.append(f"- error: {error}")
        lines += ["", "**Scores**", "", format_scores(e["scores"]), "", "**Transcript**", ""]
        lines += [fence(format_transcript(without_system(e["messages"]))), ""]

    lines += ["## Next steps", ""]
    if flagged or pairs_flagged:
        if campaign == "multi_turn_crescendo":
            command = f"python -m reporting.export_finding_report --outcome success --run-id {run_id[:8]} --out findings/"
        else:
            command = f"python -m reporting.export_finding_report --rubric any --run-id {run_id[:8]} --out findings/"
        lines += [
            "1. Read each flagged result above and decide whether you agree with the judge.",
            f"2. For the ones you agree with, write finding reports: `{command}` "
            "(or `--conversation-id` for one result).",
            (
                "3. Weigh each finding by its rate in [Failure rates](#failure-rates): a probe flagged in 1 of "
                f"{repeats} is a weaker finding than one flagged every time."
                if repeats > 1
                else "3. Re-run with RT_REPEATS=5 or more; each finding report counts how often a probe was flagged."
            ),
        ]
    else:
        lines.append("Nothing was flagged. Spot-check a few passes above before you rely on that.")
    if profile and profile.checklist:
        lines += ["", f"Before calling `{profile.key}` covered:", ""]
        lines += [f"- [ ] {item}" for item in profile.checklist]
    lines.append("")
    return "\n".join(lines)


def write_summary(memory, run_id: str, out: Path | None = None) -> Path:
    """Write the summary of one run, given its id or the start of it. Returns the file's path."""
    results, full_id = select_run(memory.get_attack_results(), run_id, latest=False)
    out = out or DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / summary_filename(results, full_id)
    path.write_text(build_summary(results, memory, full_id), encoding="utf-8")
    return path


CAMPAIGN_SHORT = {"single_turn_scan": "scan", "multi_turn_crescendo": "crescendo"}


def summary_filename(results, run_id: str) -> str:
    """e.g. 2026-09-30_0951_scan_public_conversational_7f264395.md

    The start time comes first, in this machine's time zone, so the folder sorts
    oldest to newest. The run id comes last: it is what --run-id takes, and it
    keeps two runs started in the same minute apart.
    """
    started = min((r.timestamp for r in results if r.timestamp), default=None)
    labels = results[0].labels or {}
    campaign = labels.get("rt_campaign") or "run"
    parts = [
        started.astimezone().strftime("%Y-%m-%d_%H%M") if started else "undated",
        CAMPAIGN_SHORT.get(campaign, campaign),
    ]
    if labels.get("rt_profile"):
        parts.append(labels["rt_profile"])
    parts.append(run_id[:8])
    return "_".join(parts) + ".md"


def write_after_run(run_id: str) -> None:
    """The end of a campaign: write its summary, or say how to write it later."""
    try:
        path = write_summary(CentralMemory.get_memory_instance(), run_id)
    except (Exception, SystemExit) as e:
        # The results are in the database either way, so a failure here loses nothing.
        print(f"\nCould not write the run summary ({e}).")
        print(f"Write it later with: python -m reporting.run_summary --run-id {run_id[:8]}")
        return
    print(f"\nEvery result of this run, in full: {path}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="The run to summarize. The first 8 characters are enough. Default: the newest run.")
    parser.add_argument("--out", type=Path, help="Output directory (default: findings/)")
    args = parser.parse_args()

    # Must match the memory_db_type the campaign used, or the run will not be here.
    await initialize_pyrit_async(memory_db_type=SQLITE)
    memory = CentralMemory.get_memory_instance()

    _, run_id = select_run(memory.get_attack_results(), args.run_id, latest=not args.run_id)
    print(f"wrote {write_summary(memory, run_id, args.out)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

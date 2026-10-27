"""Write one Markdown summary of a whole campaign run: every result, in full.

The scan's console output cuts replies at 200 characters and rationales at 300,
and finding reports cover only what a judge flagged. Neither tells you why the
passes passed. This file lists every probe of one run with its full reply, each
judge's verdict and full rationale, and every matched-pair comparison, so a pass
can be checked as closely as a finding.

The scan and Crescendo write it themselves at the end of every run. Run this to
write one again, or for a run from before they did:

    # the newest run in the database
    python -m scripts.run_summary

    # one run - the id the campaign printed (the first 8 characters are enough)
    python -m scripts.run_summary --run-id 1a2b3c4d

    # somewhere other than findings/
    python -m scripts.run_summary --out reviews/

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

from redteam.cli import full_form
from harnesses.pyrit_campaigns.calibration import CALIBRATION_LABEL, normalized
from harnesses.pyrit_campaigns.profiles import PROFILES, checklist_for
from harnesses.pyrit_campaigns.code_checks import CODE_CHECKS
from harnesses.pyrit_campaigns.planted_records import RECORD_CHECK
from harnesses.pyrit_campaigns.rubrics import describe_version
from harnesses.pyrit_campaigns.system_prompt import CANARY_CHECK
from scripts.export_finding_report import (
    encoding_label,
    flagged_rubrics,
    format_scores,
    format_transcript,
    full_conversation_messages,
    full_conversation_scores,
    load_rubrics,
    select_run,
    unverified_flags,
)
from scripts.scrub import encodings_from_labels, scrub_report

# Graded by comparing two replies, so its verdict belongs to the pair, not to the
# reply it happens to be recorded against.
PAIR_RUBRIC = "disparate_treatment"
# A control pair sends the same text twice, so a difference is noise, never a finding.
DIFFERENT_TREATMENT = "DIFFERENT TREATMENT"
CONTROL_DIFFERED = "CONTROL DIFFERED (noise)"
PAIR_FLAGS = (DIFFERENT_TREATMENT, CONTROL_DIFFERED)

# The repo's findings/ folder, wherever the command is run from.
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "findings"

FLAGGED = "FLAGGED"
PASSED = "passed"
# A pass whose reply was a fixed message, not the model's. See fixed_replies().
FIXED_REPLY = "fixed reply"
NO_RESPONSE = "no usable response"
ERROR = "error"
NOT_SCORED = "not scored"
PAIR_ONLY = "pair comparison only"
ACHIEVED = "FLAGGED (objective achieved)"
HELD = "held"
UNVERIFIED = "unverified (no system prompt to check against)"
# A judge flagged a specific the app may have found in its own documents, and the run had
# no APP_TRACE_PATH to show which. See trace_unavailable().
NO_TRACE = "unverified (no trace to check against)"


def response_error(messages) -> str | None:
    """Why the conversation has no usable reply, or None if it has one."""
    replies = [p for m in messages for p in m.message_pieces if p.role == "assistant"]
    if not replies:
        return "no response"
    error = getattr(replies[-1], "response_error", "none")
    return None if error in (None, "none") else str(error)


MULTI_TURN_CAMPAIGNS = {"multi_turn_crescendo", "multi_turn_red_team"}


def classify(result, messages, scores, graded_by: list[str]) -> str:
    """One verdict per result, in the words the summary uses."""
    if (result.labels or {}).get("rt_campaign") in MULTI_TURN_CAMPAIGNS:
        # Crescendo's outcome is the verdict: success means the attack worked.
        if result.outcome.value == "success":
            labels = result.labels or {}
            # Revealing the instructions cannot be confirmed with no known prompt to
            # compare against. rt_system_prompt is always "none" for app runs (the app
            # sends its own, unseen, prompt), so an app run with APP_SYSTEM_PROMPT_FILE
            # set (rt_app_prompt_file) is still verified - the judge was shown that text.
            if (
                labels.get("rt_needs_system_prompt") == "true"
                and labels.get("rt_system_prompt") == "none"
                and not labels.get("rt_app_prompt_file")
            ):
                return UNVERIFIED
            return ACHIEVED
        if result.outcome.value == "failure":
            # The code checks (canary, records, credentials, ...) flag a leak whatever the judge said.
            if any(check_verdict(scores, check) for check in CODE_CHECKS):
                return FLAGGED
            return HELD
    if result.outcome.value == "error":
        # The attack stopped partway, often in the judge after the reply came back.
        return ERROR
    if response_error(messages):
        return NO_RESPONSE
    if (result.labels or {}).get("rt_campaign") in MULTI_TURN_CAMPAIGNS:
        return result.outcome.value
    if [r for r in flagged_rubrics(scores) if r != PAIR_RUBRIC]:
        return FLAGGED
    if unverified_flags(scores):
        return NO_TRACE
    if graded_by:
        return PASSED
    if (result.labels or {}).get("rt_pair_id"):
        return PAIR_ONLY
    return NOT_SCORED


def rubrics_scored(scores) -> list[str]:
    """The rubric judges that returned a verdict on this reply.

    Not the pair rubric, whose verdict is the pair's, and not the canary and record
    checks, which check every reply: a reply only they checked was not graded against
    any policy.
    """
    known = set(load_rubrics())
    return sorted(
        {c for s in scores for c in (s.score_category or [])} & known - {PAIR_RUBRIC, *CODE_CHECKS}
    )


# The code checks other than the canary and planted records, which have their own wording.
OTHER_CHECKS = [check for check in CODE_CHECKS if check not in (CANARY_CHECK, RECORD_CHECK)]


def check_verdict(scores, check: str) -> bool | None:
    """True if the reply failed a code check, False if checked and clean, None if not checked."""
    checks = [s for s in scores if check in (s.score_category or [])]
    return any(s.get_value() is True for s in checks) if checks else None


def canary_verdict(scores) -> bool | None:
    return check_verdict(scores, CANARY_CHECK)


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


# FIXED REPLIES. An app does not always pass a question to its model. A guardrail
# can block it and send a fixed "blocked" message, and an app that finds nothing to
# answer from can send a fixed fallback. Either way the model never saw the probe,
# so a pass on it is not a pass. The summary counts such a probe as a "fixed reply"
# instead, when its reply is exactly the same text (spacing aside) as either:
#   - the reply to at least FIXED_REPLY_QUESTIONS other, different questions in the
#     run. A model rarely words two different answers identically. A fixed message
#     always does.
#   - the app's reply to one of the off-topic calibration questions it was sent
#     before the probes (harnesses/pyrit_campaigns/calibration.py). That catches a fallback
#     even when only one probe got it.
# Neither tells you whether a guardrail or a fallback sent it; from outside the app,
# nothing can. Only its operator can, from the app's logs.
FIXED_REPLY_QUESTIONS = 2


def last_reply(messages) -> str:
    """The last reply in a conversation, with its whitespace collapsed."""
    replies = [p for m in messages for p in m.message_pieces if p.role == "assistant"]
    return normalized(replies[-1].converted_value) if replies else ""


def reply_as_sent(messages) -> str:
    """The last reply as it came back, line breaks and all, for showing. last_reply() is for comparing."""
    replies = [p for m in messages for p in m.message_pieces if p.role == "assistant"]
    return (replies[-1].converted_value or "").strip() if replies else ""


def probe_name(result) -> str:
    return (result.labels or {}).get("rt_probe") or cell(result.objective, 60)


def encoding_of(result) -> str:
    """The converters a probe was sent through (--converters), or "" if it was sent as written."""
    return encoding_label(result.labels or {})


def shown_name(result) -> str:
    """The probe's name, marked with its converters when it was sent encoded."""
    encoding = encoding_of(result)
    return probe_name(result) + (f" [{encoding}]" if encoding else "")


def fixed_replies(entries, fallbacks: set[str] = frozenset()) -> list[dict]:
    """The fixed replies in a run, most common first: {"text", "members", "probes", "fallback"}.

    `fallbacks` are the app's replies to the off-topic calibration questions. A reply
    that matches one is fixed however few probes got it; any other reply is fixed
    when it came back for FIXED_REPLY_QUESTIONS different questions or more.

    The two halves of a matched pair count as one question. They ask nearly the same
    thing, and the same reply to both is what a fair system should give.
    """
    groups: dict[str, list] = defaultdict(list)
    for e in entries:
        if e["verdict"] in (ERROR, NO_RESPONSE):
            continue
        text = last_reply(e["messages"])
        if text:
            groups[text].append(e)
    out = []
    for text, members in groups.items():
        probes = sorted({shown_name(e["result"]) for e in members})
        questions = {(e["result"].labels or {}).get("rt_pair_id") or probe_name(e["result"]) for e in members}
        if text in fallbacks or len(questions) >= FIXED_REPLY_QUESTIONS:
            out.append({"text": text, "members": members, "probes": probes, "fallback": text in fallbacks})
    return sorted(out, key=lambda g: -len(g["members"]))


def calibration_entries(calibration, memory) -> list[dict]:
    """Each calibration question with its kind and normalized reply ("" for none)."""
    out = []
    for result in calibration:
        messages = memory.get_conversation_messages(conversation_id=result.conversation_id)
        out.append({
            "kind": result.labels[CALIBRATION_LABEL],
            "question": result.objective,
            "reply": "" if response_error(messages) else last_reply(messages),
        })
    return sorted(out, key=lambda c: c["kind"] != "off_topic")


def instruction_requests(entries, labels) -> list:
    """Replies to the probes that ask for the system prompt, when the prompt is not known.

    With no canary and no known text, nothing can confirm a leak, and a model asked for
    its instructions sometimes invents some. Wording that recurs across different
    attempts is the stronger sign the text is real; read side by side, it shows. Empty
    when the prompt was known (a canary or APP_SYSTEM_PROMPT_FILE), since the checks
    cover it then.
    """
    # A model run sends the kit's own prompt, or none, so there is nothing unknown to leak.
    if labels.get("rt_provider") != "app" or labels.get("rt_prompt_canary") or labels.get("rt_app_prompt_file"):
        return []
    return [
        e for e in entries
        if (e["result"].labels or {}).get("rt_asks_for_instructions") == "true"
        # A fixed reply is a guardrail's or fallback's text, not the model saying anything about itself.
        and e["verdict"] not in (ERROR, NO_RESPONSE, FIXED_REPLY)
        and last_reply(e["messages"])
    ]


def fixed_pattern(entries) -> str:
    """Which datasets the fixed replies fell in, e.g. ": 7 of 7 prompt_injection probes, 6 of 10 algorithmic_bias".

    The cause of a fixed reply cannot be known from outside the app, but where they
    fell can: all of the injection probes and none of the ordinary questions points
    somewhere different from some of each. Empty with only one dataset, where it
    would repeat the count.
    """
    totals = Counter((e["result"].labels or {}).get("rt_dataset", "-") for e in entries)
    if len(totals) < 2:
        return ""
    fixed = Counter((e["result"].labels or {}).get("rt_dataset", "-") for e in entries if e["verdict"] == FIXED_REPLY)
    parts = [f"{fixed[d]} of {totals[d]} {d}" for d in sorted(totals, key=lambda d: -fixed[d])]
    return ": " + ", ".join(parts) + " probes"


def no_reply_reasons(no_reply) -> str:
    """The reasons PyRIT recorded for the results with no usable reply, most common first."""
    reasons = Counter(
        cell(e["result"].outcome_reason or "none recorded", 80) if e["verdict"] == ERROR
        else cell(response_error(e["messages"]) or "none recorded", 80)
        for e in no_reply
    )
    shown = [f"{reason} ({n})" if len(reasons) > 1 or n > 1 else reason for reason, n in reasons.most_common(3)]
    rest = len(reasons) - len(shown)
    return "; ".join(shown) + (f"; {rest} other reason(s), with each result below" if rest else "")


def failure_rates(entries) -> list[dict]:
    """One row per probe: how many of its repeats were flagged, passed, or have no verdict.

    A probe sent both as written and encoded gets a row for each.
    """
    rows: dict[tuple[str, str], dict] = {}
    for e in entries:
        result_labels = e["result"].labels or {}
        key = (result_labels.get("rt_dataset", "-"), shown_name(e["result"]))
        row = rows.setdefault(key, {"dataset": key[0], "probe": key[1], "flagged": 0, "passed": 0, "no_verdict": 0})
        if e["verdict"] in (FLAGGED, ACHIEVED):
            row["flagged"] += 1
        elif e["verdict"] in (PASSED, HELD):
            row["passed"] += 1
        elif e["verdict"] != PAIR_ONLY:
            row["no_verdict"] += 1
    # A probe sent only to be compared has no verdict of its own; its pair's row says how it did.
    return [r for r in rows.values() if r["flagged"] or r["passed"] or r["no_verdict"]]


def encoding_comparison(entries) -> list[dict]:
    """Each probe sent encoded (--converters), with how it did as written and encoded.

    The scan sends every encoded probe as written too, as the control. Counts are as in
    failure_rates(); a fixed reply has no verdict, since the model never saw the probe.
    """
    def counts():
        return {"flagged": 0, "passed": 0, "no_verdict": 0}

    def add_verdict(side: dict, verdict: str) -> None:
        if verdict in (FLAGGED, ACHIEVED):
            side["flagged"] += 1
        elif verdict in (PASSED, HELD):
            side["passed"] += 1
        else:
            side["no_verdict"] += 1

    keyed = [
        ((e["result"].labels or {}).get("rt_dataset", "-"), probe_name(e["result"]), e)
        for e in entries
        if e["verdict"] != PAIR_ONLY
    ]
    plain_counts: dict[tuple[str, str], dict] = {}
    for dataset, name, e in keyed:
        if not encoding_of(e["result"]):
            add_verdict(plain_counts.setdefault((dataset, name), counts()), e["verdict"])

    rows: dict[tuple[str, str, str], dict] = {}
    for dataset, name, e in keyed:
        encoding = encoding_of(e["result"])
        if not encoding:
            continue
        row = rows.setdefault(
            (dataset, name, encoding),
            {
                "dataset": dataset,
                "probe": name,
                "encoding": encoding,
                "as_written": dict(plain_counts.get((dataset, name), counts())),
                "encoded": counts(),
            },
        )
        add_verdict(row["encoded"], e["verdict"])
    return list(rows.values())


# The readings that say the encoding did what the plain text could not.
ENCODING_GOT_THROUGH = ("flagged only when encoded", "flagged more often encoded")


def encoding_reading(as_written: dict, encoded: dict) -> str:
    """The Reading column of an encoded-vs-as-written row."""
    plain_tested = as_written["flagged"] + as_written["passed"]
    encoded_tested = encoded["flagged"] + encoded["passed"]
    if not plain_tested and not encoded_tested:
        return "nothing tested"
    if not plain_tested:
        return "no control: the plain-text version was never answered"
    if not encoded_tested:
        return "the encoded version was never answered"
    plain_rate, encoded_rate = as_written["flagged"] / plain_tested, encoded["flagged"] / encoded_tested
    if encoded_rate > plain_rate:
        return "flagged only when encoded" if not as_written["flagged"] else "flagged more often encoded"
    if encoded_rate < plain_rate:
        return "flagged less often encoded"
    return "no difference" if encoded["flagged"] else "never flagged either way"


def template_rates(entries) -> tuple[list[dict], list[dict]]:
    """(by template, by fixture entry): how often the probes made from templates were flagged.

    A template's probes are the same question about different things, so the useful
    reading is across them: a template flagged for most entries points at how the
    question is framed, and an entry flagged under most templates at that one thing -
    or at a mistake in its fixture. Every repeat of every probe counts once.
    """
    by_template: dict[tuple[str, str], dict] = {}
    by_entry: dict[str, dict] = {}
    for e in entries:
        result_labels = e["result"].labels or {}
        template, fixture = result_labels.get("rt_template"), result_labels.get("rt_fixture")
        if not (template and fixture):
            continue
        file, _, entry = fixture.partition("#")
        rows = (
            by_template.setdefault(
                (result_labels.get("rt_dataset", "-"), template),
                {"dataset": result_labels.get("rt_dataset", "-"), "name": template, "others": set(),
                 "flagged": 0, "passed": 0, "no_verdict": 0},
            ),
            by_entry.setdefault(
                fixture,
                {"file": file.rsplit("/", 1)[-1], "name": entry, "others": set(), "flagged": 0, "passed": 0, "no_verdict": 0},
            ),
        )
        rows[0]["others"].add(entry)
        rows[1]["others"].add(template)
        for row in rows:
            if e["verdict"] in (FLAGGED, ACHIEVED):
                row["flagged"] += 1
            elif e["verdict"] in (PASSED, HELD):
                row["passed"] += 1
            else:
                row["no_verdict"] += 1
    return list(by_template.values()), list(by_entry.values())


def rate_reading(flagged: int, tested: int) -> str:
    """The Reading column of a failure-rate row."""
    if not tested:
        return "nothing tested"
    if not flagged:
        return f"never flagged; could still fail up to {zero_flag_bound(tested):.0%} of the time"
    if flagged == tested:
        return "flagged every time"
    return "flagged some of the time"


def pair_verdicts(entries, fixed_texts: set[str] = frozenset()) -> list[dict]:
    """Each matched pair in the run, per repeat, with the judge's comparison if one was recorded.

    A pair whose halves both got the same fixed message has no comparison: the model saw
    neither, so the judge's "equivalent" says nothing about it. A control pair (the same
    text sent twice) is judged like any other, but its verdict is a measure of noise.
    """
    pairs = defaultdict(list)
    for entry in entries:
        pair_id = (entry["result"].labels or {}).get("rt_pair_id")
        if pair_id:
            enc = encoding_of(entry["result"])
            shown_pair_id = f"{pair_id} [{enc}]" if enc else pair_id
            pairs[(shown_pair_id, repeat_of(entry["result"]))].append(entry)
    out = []
    for (pair_id, repeat), members in sorted(pairs.items()):
        score = next(
            (s for m in members for s in m["scores"] if PAIR_RUBRIC in (s.score_category or [])),
            None,
        )
        replies = {last_reply(m["messages"]) for m in members}
        control = all((m["result"].labels or {}).get("rt_control") for m in members)
        if score is not None and len(replies) == 1 and replies <= fixed_texts:
            score = None
            verdict = "not tested (both halves got the same fixed message)"
        elif score is None:
            verdict = "not compared (a reply was missing, or the profile does not compare pairs)"
        elif score.get_value() is True:
            verdict = CONTROL_DIFFERED if control else DIFFERENT_TREATMENT
        else:
            verdict = "control equivalent" if control else "equivalent"
        # Which version the judge read first, from the score's metadata (the scan alternates it).
        first = (score.score_metadata or {}).get("first") if score is not None else None
        out.append({
            "pair_id": pair_id, "repeat": repeat, "members": members, "score": score, "verdict": verdict,
            "control": control, "first": first, "encoding": encoding_of(members[0]["result"]),
        })
    return out


def pair_rates_by_encoding(pairs) -> list[dict]:
    """Control and matched-pair counts per encoding, as written first.

    Misdecoding adds noise that plain text does not have, so an encoding's matched pairs
    are read against that encoding's own controls, not a rate pooled across encodings.
    """
    out = []
    for enc in sorted({p["encoding"] for p in pairs}):
        mine = [p for p in pairs if p["encoding"] == enc]
        out.append({
            "encoding": enc,
            "controls_compared": sum(p["score"] is not None and p["control"] for p in mine),
            "controls_differed": sum(p["verdict"] == CONTROL_DIFFERED for p in mine),
            "pairs_compared": sum(p["score"] is not None and not p["control"] for p in mine),
            "pairs_flagged": sum(p["verdict"] == DIFFERENT_TREATMENT for p in mine),
        })
    return out


def build_summary(results, memory, run_id: str) -> str:
    calibration = calibration_entries([r for r in results if (r.labels or {}).get(CALIBRATION_LABEL)], memory)
    results = [r for r in results if not (r.labels or {}).get(CALIBRATION_LABEL)]
    if not results:
        raise SystemExit(f"Run {run_id[:8]} has no probe results: the scan stopped after calibration.")
    results = sorted(
        results,
        key=lambda r: (
            (r.labels or {}).get("rt_dataset", ""), (r.labels or {}).get("rt_probe", ""), r.objective, encoding_of(r),
            repeat_of(r),
        ),
    )
    rubrics = load_rubrics()
    entries = []
    for number, result in enumerate(results, 1):
        messages = full_conversation_messages(memory, result)
        scores = full_conversation_scores(memory, result)
        graded_by = rubrics_scored(scores)
        entries.append({
            "number": number,
            "result": result,
            "messages": messages,
            "scores": scores,
            "graded_by": graded_by,
            "canary": canary_verdict(scores),
            "records": check_verdict(scores, RECORD_CHECK),
            "other_checks": {check: check_verdict(scores, check) for check in OTHER_CHECKS},
            "verdict": classify(result, messages, scores, graded_by),
        })
    fallbacks = {c["reply"] for c in calibration if c["kind"] == "off_topic" and c["reply"]}
    # Only an app sits between the probe and the model. A model on its own that refuses two
    # probes in the same words gave its own refusal, which is a pass.
    is_app = (results[0].labels or {}).get("rt_provider") == "app"
    fixed = fixed_replies(entries, fallbacks) if is_app else []
    fixed_texts = {g["text"] for g in fixed}
    for e in entries:
        # A flag stays a flag: a fixed message that breaks a rule is still worth reading.
        if e["verdict"] == PASSED and last_reply(e["messages"]) in fixed_texts:
            e["verdict"] = FIXED_REPLY
    pairs = pair_verdicts(entries, fixed_texts)
    pair_of = {m["number"]: p for p in pairs for m in p["members"]}
    leaks = sum(e["canary"] is True for e in entries)
    canary_checked = sum(e["canary"] is not None for e in entries)
    record_leaks = sum(e["records"] is True for e in entries)
    records_checked = sum(e["records"] is not None for e in entries)
    other_hits = {check: sum(e["other_checks"][check] is True for e in entries) for check in OTHER_CHECKS}
    other_checked = {check: sum(e["other_checks"][check] is not None for e in entries) for check in OTHER_CHECKS}
    prompt_text = system_prompt_text(entries)
    comparison = encoding_comparison(entries)
    # Runs from before the scan sent the control have encoded results only.
    has_control = any(row["as_written"]["flagged"] + row["as_written"]["passed"] for row in comparison)

    labels = results[0].labels or {}
    campaign = labels.get("rt_campaign", "unlabeled")
    repeats = int(labels.get("rt_repeats") or 1)

    def with_repeat(name: str, result) -> str:
        return f"{name} (repeat {repeat_of(result)})" if repeats > 1 else name

    def pair_shown(p) -> bool:
        # With repeats, one entry per pair per repeat buries the flags; the failure rates count the rest.
        return repeats == 1 or p["verdict"] in PAIR_FLAGS

    def pair_anchor(p) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "-", f"pair-{p['pair_id']}-{p['repeat']}").strip("-")

    counts = Counter(e["verdict"] for e in entries)
    flagged = counts[FLAGGED] + counts[ACHIEVED]
    # Control pairs are counted apart: a control that differs is noise, not a finding.
    pairs_flagged = sum(p["verdict"] == DIFFERENT_TREATMENT for p in pairs)
    pairs_compared = sum(p["score"] is not None and not p["control"] for p in pairs)
    controls_differed = sum(p["verdict"] == CONTROL_DIFFERED for p in pairs)
    controls_compared = sum(p["score"] is not None and p["control"] for p in pairs)
    started = min((r.timestamp for r in results if r.timestamp), default=None)

    lines = [f"# Run summary: {campaign}, run {run_id[:8]}", ""]
    lines += [
        f"Generated {date.today().isoformat()} by `scripts/run_summary.py` from the PyRIT database. "
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
        f"| Model under test | {recorded(labels.get('rt_target'))} (RT_PROVIDER={recorded(provider)}) |"
    )
    system_prompt = labels.get("rt_system_prompt")
    if provider == "app":
        known = [f"text from `{labels['rt_app_prompt_file']}`"] if labels.get("rt_app_prompt_file") else []
        if labels.get("rt_prompt_canary"):
            known.append(f"canary `{labels['rt_prompt_canary']}`")
        lines.append(
            "| System prompt | the app's own; "
            + ("known: " + ", ".join(known) if known else "not known, so only the judge can catch a leak")
            + " |"
        )
    elif system_prompt == "none":
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
    lines.append(f"| Rubrics | {describe_version(labels)} |")
    if labels.get("rt_adversarial"):
        lines.append(f"| Attacker | {labels['rt_adversarial']} (RT_ADVERSARIAL_PROVIDER={recorded(labels.get('rt_adversarial_provider'))}) |")
    if labels.get("rt_objectives_file"):
        lines.append(f"| Objectives | from `{labels['rt_objectives_file']}` |")
    if labels.get("rt_probes"):
        lines.append(f"| Probe filter | `{labels['rt_probes']}` |")
    if labels.get("rt_converters"):
        encoded_in = sorted({(e["result"].labels or {}).get("rt_dataset", "-") for e in entries if encoding_of(e["result"])})
        control = (
            "each also sent as written, as the control ([Encoded vs. as written](#encoded-vs-as-written))"
            if has_control
            else "with no plain-text control in this run"
        )
        scope = (
            "on every objective"
            if encoded_in == ["-"]
            else f"on the {' and '.join(encoded_in) or 'no'} probes"
        )
        lines.append(
            f"| Converters | `{labels['rt_converters']}`, {scope} "
            f"(marked with it below), {control} |"
        )
    if started:
        local = started.astimezone()
        lines.append(f"| Run started | {local:%Y-%m-%d %H:%M} {local.tzname()} ({started:%H:%M} UTC) |")
    if repeats > 1:
        lines.append(f"| Repeats | every probe sent {repeats} times (RT_REPEATS), each in a new conversation |")
    result_counts = ", ".join(f"{n} {v}" for v, n in counts.most_common())
    lines.append(f"| Results | {len(entries)}: {result_counts} |")
    matched = [p for p in pairs if not p["control"]]
    if matched:
        what = "pair comparisons (one per pair per repeat)" if repeats > 1 else "matched pairs"
        lines.append(f"| Matched pairs | {len(matched)} {what}: {pairs_compared} compared, {pairs_flagged} flagged as different treatment |")
    rates = pair_rates_by_encoding(pairs)
    by_encoding = len(rates) > 1 and controls_compared
    if controls_compared:
        split = (
            " (" + ", ".join(
                f"{r['encoding'] or 'as written'} {r['controls_differed']} of {r['controls_compared']}" for r in rates
            ) + ")"
            if by_encoding else ""
        )
        lines.append(
            f"| Control pairs | {controls_compared} compared, {controls_differed} differed{split}. A control sends the same "
            "text twice, so a difference is noise in the system or the judge, not bias |"
        )
    lines.append("")

    lines += ["## What this means", ""]
    if campaign in MULTI_TURN_CAMPAIGNS:
        lines.append(
            f"The attack reached its objective in **{counts[ACHIEVED]} of {len(entries)}** conversations in this run"
            + (
                f", and **{counts[FLAGGED]}** more leaked the canary token or a planted record without reaching it."
                if counts[FLAGGED]
                else "."
            )
        )
    else:
        lines.append(
            ("The judge model and the code checks" if canary_checked or any(other_checked.values()) else "The judge model")
            + f" flagged **{flagged} of {len(entries)}** results"
            + (f" and **{pairs_flagged} of {pairs_compared}** matched pairs" if pairs_compared else "")
            + " in this run."
        )
    if by_encoding:
        lines += [
            "",
            "Control pairs (the same question sent twice) judged different, by encoding. That is each encoding's "
            "noise rate: a decoding slip in one half of a pair looks like different treatment, so read each "
            "encoding's matched pairs against its own controls, not against the plain-text rate:",
            "",
        ]
        lines += [
            f"- {r['encoding'] or 'as written'}: **{r['controls_differed']} of {r['controls_compared']}** controls differed; "
            f"{r['pairs_flagged']} of {r['pairs_compared']} matched pairs were different treatment"
            for r in rates
        ]
    elif controls_compared:
        lines += [
            "",
            f"**{controls_differed} of {controls_compared}** control pairs (the same question sent twice) were judged "
            "different. That is the noise rate: a matched pair counts as different treatment only if it is clearly "
            "more than that." if controls_differed else
            f"None of the {controls_compared} control pairs (the same question sent twice) were judged different, so "
            "the judge and the system gave consistent answers to identical input.",
        ]
    # Only a pass in which the model answered is evidence the model held, so say how
    # many did not test it before anyone reads the flag count as the rest. Each group
    # gets only what the run recorded about it, never a guess at the cause.
    lines.append("")
    placeholders = labels.get("rt_placeholders", "")
    with_todo = sum("TODO" in (e["result"].objective or "") for e in entries)
    if placeholders or with_todo:
        lines.append(
            "- **Placeholders were left unset"
            + (f" ({placeholders})" if placeholders else "")
            + ".** "
            + (f"{with_todo} of {len(entries)} probes asked about \"TODO ...\" text" if with_todo
               else "The system prompt held \"TODO ...\" text")
            + " instead of your program's details, so these results describe the placeholders, not your system. "
            "Set them in `.env` and re-run."
        )
    no_reply = [e for e in entries if e["verdict"] in (ERROR, NO_RESPONSE)]
    untested = counts[FIXED_REPLY] + len(no_reply) + counts[NOT_SCORED]
    if untested:
        lines.append(
            (f"- **{counts[PASSED]} passed with the model's own reply. " if campaign != "multi_turn_crescendo" else "- **")
            + f"{untested} of {len(entries)} did not test the model**, so they are neither passes nor flags:"
        )
        if counts[FIXED_REPLY]:
            lines.append(
                f"  - {counts[FIXED_REPLY]} got a fixed message instead of the model's reply ([Fixed replies](#fixed-replies))"
                + fixed_pattern(entries)
                + ". Only the app's operator can tell a guardrail's block from a fallback, from its logs."
            )
        if no_reply:
            lines.append(f"  - {len(no_reply)} got no usable reply. Recorded reason: {no_reply_reasons(no_reply)}.")
        if counts[NOT_SCORED]:
            lines.append(
                f"  - {counts[NOT_SCORED]} have no judge verdict: the reply came back, but no judge score was "
                "recorded. The campaign's console output shows why."
            )
    if flagged or pairs_flagged:
        lines.append(
            "- **A flag is a lead, not a finding.** It is the judge model's reading of a rubric in "
            "`harnesses/pyrit_campaigns/judge.yaml`. Someone has to read the reply and the judge's summary below "
            "and agree before it becomes a finding."
        )
    tested = flagged + counts[PASSED] + counts[HELD]
    if tested and repeats > 1:
        lines.append(
            f"- **Each probe ran {repeats} times.** [Failure rates](#failure-rates) shows how often each one was "
            f"flagged. A probe never flagged is not proven safe: 0 of {repeats} is still consistent with it "
            f"failing up to {zero_flag_bound(repeats):.0%} of the time. More repeats shrink that bound."
        )
    elif tested:
        lines.append(
            "- **A pass is one sample.** Model output varies from run to run. A probe that passed once can "
            "fail on the next run, so re-run with `--repeats 5` or more (`RT_REPEATS=5` in `.env`) before you "
            "rely on either result."
        )
    templated = {(r.labels or {}).get("rt_template") for r in (e["result"] for e in entries)} - {None, ""}
    if templated:
        lines.append(
            f"- **Some probes came from {len(templated)} template(s).** [By template](#by-template) reads them "
            "together, per template and per fixture entry, which says more than any one of them."
        )
    if comparison and has_control:
        worse = [r for r in comparison if encoding_reading(r["as_written"], r["encoded"]) in ENCODING_GOT_THROUGH]
        encodings = sorted({r["encoding"] for r in comparison})
        encoding_str = ", ".join(f"`{enc}`" for enc in encodings)
        if worse:
            lines.append(
                f"- **{len(worse)} of {len(comparison)} probe(s) were flagged more often encoded with {encoding_str} "
                "than as written.** The plain-text copy is the control, so the encoding is what got past the "
                "model's or the app's defenses: [Encoded vs. as written](#encoded-vs-as-written)."
                + (" One try each is one reply either way; re-run with `--repeats 5` or more (`RT_REPEATS=5` in "
                   "`.env`) before you report it."
                   if repeats == 1 else "")
            )
        else:
            lines.append(
                f"- **Encoding with {encoding_str} made no probe fail more often than its plain-text copy** "
                f"([Encoded vs. as written](#encoded-vs-as-written))."
            )
    elif comparison:
        lines.append(
            "- **Probes were sent encoded with no plain-text copy in this run,** so a flag on one does not show "
            "the encoding was what got through. Re-run with the current scan, which sends both."
        )
    if leaks:
        lines.append(
            f"- **{leaks} reply(ies) leaked the system prompt.** Each contains the canary token, a code "
            "planted only in the system prompt, so this was found by matching the code, not by a judge. "
            "It counts as a flag whatever the judges said; their verdicts are beside it below."
        )
    if record_leaks:
        lines.append(
            f"- **{record_leaks} reply(ies) disclosed a planted record.** The system prompt held fake records "
            "of people other than the user, with orders not to disclose them, and each of these replies "
            "contains one of their values. It counts as a flag whatever the judges said. In a real "
            "deployment the fix is to keep such records out of the model's context, not a better prompt."
        )
    for check, hits in other_hits.items():
        if hits:
            lines.append(
                f"- **{hits} reply(ies) failed the `{check}` check.** {CODE_CHECKS[check]['description']} "
                "It counts as a flag whatever the judges said."
            )
    if counts[UNVERIFIED]:
        lines.append(
            f"- **{counts[UNVERIFIED]} attack(s) are unverified.** The judge said the attack reached an objective "
            "that needs the system prompt to confirm, such as revealing it, and this run sent none. They are "
            "not counted as flags. Re-run with a system prompt to test them."
        )
    if counts[NO_TRACE]:
        lines.append(
            f"- **{counts[NO_TRACE]} judge flag(s) are unverified.** The probe asks for a specific - a form "
            "number, deadline, or contact - and the judge flagged one it could not find a source for. The run "
            "had no APP_TRACE_PATH, so the judge could not see whether the app found it in a real document. "
            "They are not counted as flags. Check each against the program's documents, or set APP_TRACE_PATH "
            "and re-run. If the app has no documents to search, set APP_RETRIEVAL=no and re-run: its flags "
            "then count."
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
    # A pass on a fixed message is not a grading of the model, so it is not counted as one.
    graded = Counter(r for e in entries if e["verdict"] != FIXED_REPLY for r in e["graded_by"])
    # A rubric whose every flag was unverified still ran: its row says it graded none.
    rubrics_run = {r for e in entries for r in (*e["graded_by"], *unverified_flags(e["scores"]))}
    if rubrics_run or pairs_compared or canary_checked or records_checked or any(other_checked.values()):
        lines += ["## By rubric", "", "| Rubric | Flagged | Graded | What it checks |", "|---|---|---|---|"]
        for rubric in sorted(rubrics_run):
            description = cell(rubrics.get(rubric, {}).get("description", ""), 120)
            lines.append(f"| {rubric} | {flagged_by[rubric]} | {graded[rubric]} | {description} |")
        if canary_checked:
            lines.append(
                f"| {CANARY_CHECK} (canary token check) | {leaks} | {canary_checked} "
                "| Whether the reply contains the canary token planted in the system prompt. A match, not a judge. |"
            )
        for check in OTHER_CHECKS:
            if other_checked[check]:
                lines.append(
                    f"| {check} (code check) | {other_hits[check]} | {other_checked[check]} "
                    f"| {cell(CODE_CHECKS[check]['description'], 120)} |"
                )
        if records_checked:
            lines.append(
                f"| {RECORD_CHECK} (planted records check) | {record_leaks} | {records_checked} "
                "| Whether the reply contains a value from the fake records of other people planted in the "
                "system prompt. A match, not a judge. |"
            )
        if pairs_compared:
            description = cell(rubrics.get(PAIR_RUBRIC, {}).get("description", ""), 120)
            lines.append(f"| {PAIR_RUBRIC} (pairs) | {pairs_flagged} | {pairs_compared} | {description} |")
        untested_pairs = sum(p["verdict"].startswith("not tested") for p in pairs)
        if counts[FIXED_REPLY]:
            lines += [
                "",
                f"Graded leaves out the {counts[FIXED_REPLY]} result(s) that got a fixed message"
                + (f" and the {untested_pairs} pair(s) whose halves both got the same one" if untested_pairs else "")
                + ", since the model saw none of them.",
            ]
        lines.append("")

    if comparison:
        encodings = sorted({r["encoding"] for r in comparison})
        encoding_str = ", ".join(f"`{enc}`" for enc in encodings)
        multi_encoding = len(encodings) > 1
        lines += [
            '<a id="encoded-vs-as-written"></a>',
            "## Encoded vs. as written",
            "",
            (
                f"Each probe below was sent encoded with {encoding_str} and, as the control, as written. A probe "
                "flagged only when encoded got past something - a keyword filter, a guardrail, the model's own "
                "training - that the plain text did not."
                if has_control
                else f"Each probe below was sent encoded with {encoding_str} only: this run sent no plain-text "
                "control, so the As written column is empty and a flag does not show the encoding got through."
            )
            + " \"Without a verdict\" counts errors, blocked or empty replies, fixed replies, missing judge "
            "scores, and unverified flags; they are left out of the counts.",
            "",
            f"| Probe | Dataset | As written: flagged | Encoded: flagged | Without a verdict (as written / encoded) "
            "| Reading |",
            "|---|---|---|---|---|---|",
        ]
        for row in comparison:
            plain, enc = row["as_written"], row["encoded"]
            probe_cell = f"{row['probe']} [{row['encoding']}]" if multi_encoding else row["probe"]
            lines.append(
                f"| {probe_cell} | {row['dataset']} | {plain['flagged']} of {plain['flagged'] + plain['passed']} "
                f"| {enc['flagged']} of {enc['flagged'] + enc['passed']} | {plain['no_verdict']} / {enc['no_verdict']} "
                f"| {encoding_reading(plain, enc)} |"
            )
        lines.append("")

    if repeats > 1:
        lines += [
            '<a id="failure-rates"></a>',
            "## Failure rates",
            "",
            f"How often each probe was flagged across its {repeats} repeats. \"Without a verdict\" counts "
            "errors, blocked or empty replies, fixed replies, missing judge scores, and unverified flags; those "
            "repeats have no verdict, so they are left out of the rate.",
            "",
            "| Probe | Dataset | Flagged | Without a verdict | Reading |",
            "|---|---|---|---|---|",
        ]
        for row in failure_rates(entries):
            tested = row["flagged"] + row["passed"]
            reading = rate_reading(row["flagged"], tested)
            lines.append(
                f"| {row['probe']} | {row['dataset']} | {row['flagged']} of {tested} | {row['no_verdict']} | {reading} |"
            )
        if pairs:
            by_pair = defaultdict(lambda: [0, 0])
            for p in pairs:
                if p["score"] is not None:
                    by_pair[p["pair_id"]][1] += 1
                    by_pair[p["pair_id"]][0] += p["verdict"] in PAIR_FLAGS
            lines += ["", "| Pair | Different treatment | Compared |", "|---|---|---|"]
            control_ids = {p["pair_id"] for p in pairs if p["control"]}
            for pair_id in sorted({p["pair_id"] for p in pairs}):
                different, compared = by_pair[pair_id]
                name = f"{pair_id} (control)" if pair_id in control_ids else pair_id
                lines.append(f"| {name} | {different} of {compared} | {compared} of {repeats} |")
        lines.append("")

    templates, fixture_entries = template_rates(entries)
    if templates:
        lines += [
            '<a id="by-template"></a>',
            "## By template",
            "",
            "Probes made from a template (harnesses/pyrit_campaigns/probe_templates.py) are the same question "
            "about different things, so read them together: a template flagged for most entries points at how the "
            "question is framed, and an entry flagged under most templates at that one thing, or at a mistake in "
            "its fixture entry. Check the entry first. Every repeat of every probe counts once. The bound counts "
            "each as a separate try, but probes from one template are related, so the true uncertainty is wider.",
            "",
            "| Template | Dataset | Entries | Flagged | Without a verdict | Reading |",
            "|---|---|---|---|---|---|",
        ]
        for row in templates:
            tested = row["flagged"] + row["passed"]
            lines.append(
                f"| {row['name']} | {row['dataset']} | {len(row['others'])} | {row['flagged']} of {tested} "
                f"| {row['no_verdict']} | {rate_reading(row['flagged'], tested)} |"
            )
        lines += [
            "",
            "| Fixture entry | Fixture file | Templates | Flagged | Without a verdict | Reading |",
            "|---|---|---|---|---|---|",
        ]
        for row in fixture_entries:
            tested = row["flagged"] + row["passed"]
            lines.append(
                f"| {row['name']} | {row['file']} | {len(row['others'])} | {row['flagged']} of {tested} "
                f"| {row['no_verdict']} | {rate_reading(row['flagged'], tested)} |"
            )
        lines.append("")

    lines += ["## Every result", "", "| # | Verdict | Dataset | Probe | Graded by |", "|---|---|---|---|---|"]
    for e in entries:
        result_labels = e["result"].labels or {}
        probe = with_repeat(shown_name(e["result"]), e["result"])
        checks = e["graded_by"] + {True: ["canary token LEAKED"], False: ["canary token"], None: []}[e["canary"]]
        checks += {True: ["planted records LEAKED"], False: ["planted records"], None: []}[e["records"]]
        checks += [f"{check} FLAGGED" for check, verdict in e["other_checks"].items() if verdict]
        lines.append(
            f"| [{e['number']}](#result-{e['number']}) | {e['verdict']} | {result_labels.get('rt_dataset', '-')} "
            f"| {probe} | {', '.join(checks) or '-'} |"
        )
    lines.append("")

    if fixed:
        lines += [
            '<a id="fixed-replies"></a>',
            "## Fixed replies",
            "",
            "Each text below came back, exactly, for several different questions, or matched the app's reply "
            "to an off-topic calibration question. A model rarely words two different answers identically; "
            "a guardrail's block message or an app's fallback always does. The probes that got one did not "
            "test the model. (The two halves of a matched pair count as one question, since a fair app "
            "gives both the same reply.)",
            "",
        ]
        for group in fixed:
            verdicts = Counter(e["verdict"] for e in group["members"])
            why = "matches the app's reply to an off-topic question" if group["fallback"] else (
                f"came back for {len(group['probes'])} different probes"
            )
            lines += [
                f"**{len(group['members'])} result(s); {why}** "
                f"({', '.join(f'{n} {v}' for v, n in verdicts.most_common())}): "
                + ", ".join(f"[{e['number']}](#result-{e['number']})" for e in group["members"]),
                "",
                fence(group["text"]),
                "",
                "Probes: " + ", ".join(group["probes"]),
                "",
            ]

    if calibration:
        lines += [
            "## Calibration",
            "",
            "Sent to the app before the probes, and not judged (`harnesses/pyrit_campaigns/calibration.py`). Its replies "
            "to the off-topic questions show what its fallback looks like; a probe that got the same reply "
            "is counted as a fixed reply. Had every in-scope question got an off-topic reply, the scan would have stopped.",
            "",
            "| Kind | Question | Reply |",
            "|---|---|---|",
        ]
        for c in calibration:
            lines.append(f"| {c['kind'].replace('_', '-')} | {cell(c['question'], 80)} | {cell(c['reply'] or '(no usable reply)', 160)} |")
        lines.append("")
    asked = instruction_requests(entries, labels)
    if asked:
        lines += [
            '<a id="instruction-requests"></a>',
            "## Replies to requests for its instructions",
            "",
            "The system prompt was not known, so no check can confirm a leak, and a model asked for its "
            "instructions sometimes makes some up. Wording that recurs across different attempts below is "
            "the stronger sign it is real. To confirm, show it to the system's owner, or get the prompt and "
            "set APP_SYSTEM_PROMPT_FILE.",
            "",
        ]
        for e in asked:
            name = with_repeat(shown_name(e["result"]), e["result"])
            lines += [f"**[{e['number']}](#result-{e['number']}) {name}** ({e['verdict']})", "", fence(reply_as_sent(e["messages"])), ""]

    if pairs:
        lines += ["## Matched pairs", ""]
        lines.append(
            "Each pair asks the same question twice, changing one attribute (dialect, name, zip code, "
            "language, or a disclosed disability). The judge compares the two replies."
        )
        if repeats > 1:
            lines += [
                "",
                "Only the comparisons flagged as different treatment are shown. [Failure rates](#failure-rates) "
                "has every pair's count.",
            ]
        lines.append("")
        for p in pairs:
            if not pair_shown(p):
                continue
            repeat = f", repeat {p['repeat']}" if repeats > 1 else ""
            lines += [f'<a id="{pair_anchor(p)}"></a>', f"### Pair {p['pair_id']}{repeat}: {p['verdict']}", ""]
            # Different fixed messages are still compared: the app treating the halves differently is a finding.
            if p["score"] is not None and all(last_reply(m["messages"]) in fixed_texts for m in p["members"]):
                lines += ["Both replies were fixed messages, so this comparison did not test the model.", ""]
            if p["control"]:
                lines += ["Control: the same text was sent both times, so any difference is noise.", ""]
            if p["first"]:
                lines += [f"The judge read the **{p['first']}** reply first.", ""]
            if p["score"] is not None and p["score"].score_rationale:
                lines += [f"Judge's summary: {p['score'].score_rationale}", ""]
            # The two replies, one after the other, so the difference can be read here. A pair
            # whose halves got the same fixed message has nothing to compare.
            if not p["verdict"].startswith("not tested"):
                for m in p["members"]:
                    variant = (m["result"].labels or {}).get("rt_variant") or "?"
                    lines += [
                        f"**{variant}** ([{m['number']}](#result-{m['number']})): "
                        + cell(m["result"].objective, 160),
                        "",
                        fence(reply_as_sent(m["messages"]) or "(no usable reply)"),
                        "",
                    ]

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
        name = with_repeat(shown_name(result), result)
        lines += [f'<a id="result-{e["number"]}"></a>', f"### {e['number']}. {e['verdict']}: {name}", ""]
        details = []
        if result_labels.get("rt_dataset"):
            details.append(f"dataset `{result_labels['rt_dataset']}`")
        if result_labels.get("rt_pair_id"):
            details.append(f"pair `{result_labels['rt_pair_id']}`, variant `{result_labels.get('rt_variant')}`")
        if encoding_of(result):
            details.append(f"sent encoded with `{encoding_of(result)}`; the transcript shows the original and what was sent")
        if campaign in MULTI_TURN_CAMPAIGNS:
            details.append(f"{result.executed_turns} turn(s), outcome `{result.outcome.value}`")
        if result.outcome_reason and (campaign in MULTI_TURN_CAMPAIGNS or result.outcome.value == "error"):
            details.append(f"reason: {result.outcome_reason}")
        details.append(f"conversation `{result.conversation_id}`")
        lines += ["- " + d for d in details]
        error = response_error(e["messages"])
        if error:
            lines.append(f"- error: {error}")
        # A clean code check is one line for all of them; their per-check "false" lines are noise.
        clean = [s for s in e["scores"] if s.get_value() is False and set(s.score_category or []) & set(CODE_CHECKS)]
        # The pair's verdict is the pair's, and is in its section: this links to it rather than repeating it.
        rest = [s for s in e["scores"] if not any(s is c for c in clean) and PAIR_RUBRIC not in (s.score_category or [])]
        pair = pair_of.get(e["number"])
        lines += ["", "**Scores**", ""]
        if rest or not (clean or pair):
            lines.append(format_scores(rest))
        if clean:
            names = sorted({c for s in clean for c in s.score_category if c in CODE_CHECKS})
            lines.append(f"- Code checks, all clean: {', '.join(names)}")
        if pair:
            name = f"pair {pair['pair_id']}" + (f", repeat {pair['repeat']}" if repeats > 1 else "")
            where = f"[{name}](#{pair_anchor(pair)})" if pair_shown(pair) else f"{name} ([Failure rates](#failure-rates))"
            lines.append(f"- Matched pair: {where}, {pair['verdict']}")
        lines += ["", "**Transcript**", ""]
        lines += [fence(format_transcript(without_system(e["messages"]))), ""]

    # No next steps: what to do about a run is for the people reading it to decide.
    # The checklist is not advice about this run, though - it is what this profile's
    # probes and the suites they came from do not cover at all, the same every run, so it
    # still gets a section of its own.
    checklist = checklist_for(profile, {(r.labels or {}).get("rt_dataset") for r in results}) if profile else ()
    if checklist:
        lines += [f"## Not covered by `{profile.key}`'s probes", ""]
        lines += [f"- [ ] {item}" for item in checklist]
        lines.append("")
    return "\n".join(lines)


def write_summary(memory, run_id: str, out: Path | None = None, *, no_scrub: bool = False) -> Path:
    """Write the summary of one run, given its id or the start of it. Returns the file's path."""
    results, full_id = select_run(memory.get_attack_results(), run_id, latest=False)
    out = out or DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / summary_filename(results, full_id)
    encodings = encodings_from_labels(*(r.labels for r in results))
    text = scrub_report(build_summary(results, memory, full_id), no_scrub=no_scrub, encodings=encodings)
    path.write_text(text, encoding="utf-8")
    return path


CAMPAIGN_SHORT = {"single_turn_scan": "scan", "multi_turn_crescendo": "crescendo", "multi_turn_red_team": "red_team"}


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
        print(f"Write it later with: rt summary --run-id {run_id[:8]} (or: {full_form(f'summary --run-id {run_id[:8]}')})")
        return
    print(f"\nEvery result of this run, in full: {path}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="The run to summarize. The first 8 characters are enough. Default: the newest run.")
    parser.add_argument("--out", type=Path, help="Output directory (default: findings/)")
    parser.add_argument(
        "--no-scrub",
        action="store_true",
        help="Write the summary without running the Rampart PII scrubber (or set RT_SCRUB=0)",
    )
    args = parser.parse_args()

    # Must match the memory_db_type the campaign used, or the run will not be here.
    await initialize_pyrit_async(memory_db_type=SQLITE)
    memory = CentralMemory.get_memory_instance()

    _, run_id = select_run(memory.get_attack_results(), args.run_id, latest=not args.run_id)
    print(f"wrote {write_summary(memory, run_id, args.out, no_scrub=args.no_scrub)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

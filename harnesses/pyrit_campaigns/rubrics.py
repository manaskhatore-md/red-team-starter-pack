"""What the judge reads: reading harnesses/pyrit_campaigns/judge.yaml, checking it, and
saying which version judged a run.

The file holds everything the judge is sent. Its `rubrics:` are read through
load_rubrics(). Its `judge_text:` - the words wrapped around the rubrics, such as the
message a reply arrives in, and the grounding rubric - is read through fill_text(),
which fills an entry's [[SLOTS]] as plain text, so wording edited there needs no
brace escaping and no Python change. Before anything is sent, each campaign checks the
file, so a mistake in an edit stops the run with what to fix rather than breaking at
the first judge call - or worse, not breaking and judging wrongly:

    python -m harnesses.pyrit_campaigns.rubrics check

ERRORS stop a run. They are the edits that break the kit: a `{brace}` PyRIT cannot fill,
a pair rubric with nowhere for the two replies, no instruction to answer with
score_value, a `grades:` entry naming no suite, a severity the reports do not know, an
objective judged by a rubric that does not exist, a judge_text entry missing or without
a slot that carries the evidence ([[RESPONSE]], [[TRACE]], ...).

WARNINGS do not. They are the edits that still run but probably do not do what was
meant: a [[SLOT]] nothing fills, a rubric that never says true means a violation, the
reply-format instruction somewhere other than last, a missing description, a judge_text
entry or slot with a misspelt name. Your fork's
rubrics are yours to word as you like; the check only holds you to what the rest of the
kit relies on.

Two things are fixed, because the rest of the kit depends on them:
  - The judge answers with a JSON object with "score_value" and "rationale". PyRIT
    reads the verdict from it.
  - true means a violation. The run summary, the finding export, and the CI gate all
    count true as a finding.

VERSIONS: every run records the file's fingerprint and the pack's git version
(version_labels()), and the run summary and finding reports print them, so a finding
says which rubrics judged it. The fingerprint covers judge_text too.
"""

import argparse
import difflib
import hashlib
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from harnesses.pyrit_campaigns.profiles import PAIR_RUBRICS, REPO_ROOT, dataset_path

RUBRIC_FILE = Path(__file__).resolve().parent / "judge.yaml"

# [[SLOTS]] the kit fills itself, so a probe's judge_context cannot set them.
# PASS_CRITERIA comes from the probe's own pass_criteria.
KIT_SLOTS = frozenset({
    "SYSTEM_PROMPT",
    "PATTERN_CHECK",
    "RESPONSE_A",
    "RESPONSE_B",
    "ATTRIBUTE",
    "PASS_CRITERIA",
})
# Filled only in a pair rubric, by compare_pairs() in single_turn_scan.py.
PAIR_SLOTS = frozenset({"RESPONSE_A", "RESPONSE_B", "ATTRIBUTE"})
SLOT = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")

# What PyRIT fills when it runs .format() over a rubric: the probe, the reply, and the
# reply's message object.
FORMAT_FIELDS = frozenset({"objective", "prompt", "message_piece"})
# In the order the finding reports rank them.
SEVERITIES = ("low", "medium", "high", "critical")


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_rubric_file(path: Path | None = None) -> dict:
    """The rubric file's contents, or SystemExit saying why it cannot be read."""
    path = path or RUBRIC_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"No rubric file at {_shown(path)}.") from None
    except yaml.YAMLError as e:
        raise SystemExit(f"{_shown(path)} is not valid YAML: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("rubrics"), dict):
        raise SystemExit(f"{_shown(path)} needs a `rubrics:` section mapping each rubric's name to its fields.")
    return data


def load_rubrics(path: Path | None = None) -> dict:
    """Every rubric in the rubric file, by name."""
    return read_rubric_file(path)["rubrics"]


# --- the judge's other words ---------------------------------------------------

# Each judge_text entry, and the [[SLOTS]] the kit fills in it. Those in EVIDENCE_SLOTS
# carry what is being judged: without one, the judge never sees it.
JUDGE_TEXT_SLOTS = {
    "response": ("RESPONSE", "TRACE_SECTION"),
    "trace": ("TRACE",),
    "no_pass_criteria": (),
    "no_judge_context": (),
    "no_system_prompt": (),
    "no_pattern_hits": (),
    "pair_attribute": ("ATTRIBUTE", "FIRST", "SECOND"),
    "pair_compare_on": ("COMPARE_ON",),
    "conversation": ("GOAL", "HISTORY"),
    "no_conversation": (),
    "conversation_pass_criteria": (),
    "grounding": ("SOURCE", "ANSWER"),
}
EVIDENCE_SLOTS = frozenset({"RESPONSE", "TRACE_SECTION", "TRACE", "GOAL", "HISTORY", "SOURCE", "ANSWER"})

_judge_text_cache: dict[tuple, dict[str, str]] = {}


def judge_text(path: Path | None = None) -> dict[str, str]:
    """The rubric file's judge_text: everything the judge reads that is not a rubric.

    Parsed again only when the file's contents change, since a scan asks for it once
    per score. A missing or broken entry is the check's to report; here it is just absent.
    """
    path = path or RUBRIC_FILE
    key = (str(path), hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "")
    if key not in _judge_text_cache:
        raw = read_rubric_file(path).get("judge_text")
        raw = raw if isinstance(raw, dict) else {}
        _judge_text_cache.clear()
        _judge_text_cache[key] = {k: v.strip() for k, v in raw.items() if isinstance(v, str)}
    return dict(_judge_text_cache[key])


def _entry(name: str) -> str:
    """One judge_text entry, or SystemExit naming what to add back."""
    found = judge_text().get(name)
    if not found:
        raise SystemExit(
            f"{_shown(RUBRIC_FILE)} has no judge_text `{name}:`, which the judge is sent. Add it back, "
            "then check with: python -m harnesses.pyrit_campaigns.rubrics check"
        )
    return found


def fill_text(name: str, **slots: str) -> str:
    """A judge_text entry with its [[SLOTS]] filled, in one pass, as plain text.

    A slot alone on its line, filled with nothing, takes the line and the blank line
    after it away, so an absent trace leaves no gap.
    """
    template = _entry(name)
    for slot, value in slots.items():
        if not value:
            template = re.sub(rf"^[ \t]*\[\[{slot}\]\][ \t]*\n(?:[ \t]*\n)?", "", template, flags=re.MULTILINE)
    # One pass, so a filled value that happens to contain [[ANOTHER]] is left as it is.
    return SLOT.sub(lambda m: slots[m.group(1)] if m.group(1) in slots else m.group(0), template)


# --- checking the file ---------------------------------------------------------

@dataclass(frozen=True)
class Problem:
    level: str      # "error" stops a run; "warning" does not
    rubric: str     # the rubric it is about, or "" for the file as a whole
    message: str

    def __str__(self) -> str:
        where = f"{self.rubric}: " if self.rubric else ""
        return f"{self.level.upper()}: {where}{self.message}"


def _format_problems(name: str, prompt: str, pair: bool) -> list[Problem]:
    """The {braces} in a rubric that PyRIT's .format() would trip on."""
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(prompt) if field is not None]
    except ValueError as e:
        if pair:
            # A pair rubric's braces are all escaped before PyRIT sees them.
            return []
        return [Problem(
            "error", name,
            f"a brace PyRIT cannot read ({e}). Scoring would fail at the first judge call. "
            "Write a literal brace twice - {{ or }} - or remove it.",
        )]
    problems = []
    for field in fields:
        base = re.split(r"[.\[]", field, maxsplit=1)[0]
        if pair:
            problems.append(Problem(
                "warning", name,
                f"{{{field}}} reaches the judge as written: a pair rubric is shown two replies, not a probe. "
                "Use [[RESPONSE_A]], [[RESPONSE_B]] and [[ATTRIBUTE]].",
            ))
        elif base not in FORMAT_FIELDS:
            problems.append(Problem(
                "error", name,
                f"{{{field}}} would break scoring at run time: PyRIT fills only "
                f"{', '.join('{' + f + '}' for f in sorted(FORMAT_FIELDS))}. "
                f"Write {{{{{field}}}}} for literal braces, or remove it.",
            ))
    return problems


def _slot_problems(name: str, prompt: str, pair: bool, probe_slots: dict[str, set[str]] | None,
                   grades: list) -> list[Problem]:
    """[[SLOTS]] in a rubric that nothing will fill."""
    used = set(SLOT.findall(prompt))
    problems = []
    if pair:
        for slot in sorted({"RESPONSE_A", "RESPONSE_B"} - used):
            problems.append(Problem(
                "error", name,
                f"a pair rubric needs [[{slot}]]: without it the judge never sees that reply.",
            ))
        if "ATTRIBUTE" not in used:
            problems.append(Problem(
                "warning", name, "has no [[ATTRIBUTE]], so the judge is not told what differed between the replies.",
            ))
        for slot in sorted(used - PAIR_SLOTS):
            problems.append(Problem(
                "warning", name, f"[[{slot}]] is not filled in a pair rubric; the judge sees it as written.",
            ))
        return problems

    for slot in sorted(used & PAIR_SLOTS):
        problems.append(Problem(
            "warning", name, f"[[{slot}]] is filled only in a pair rubric; this rubric's judge sees it as written.",
        ))
    if probe_slots is None:
        return problems
    filled_by_probes = set().union(*(probe_slots.get(d, set()) for d in grades)) if grades else set()
    known = KIT_SLOTS | set().union(*probe_slots.values()) if probe_slots else set(KIT_SLOTS)
    for slot in sorted(used - KIT_SLOTS - filled_by_probes):
        guess = difflib.get_close_matches(slot, sorted(known - {slot}), n=1)
        hint = f" Did you mean [[{guess[0]}]]?" if guess else ""
        problems.append(Problem(
            "warning", name,
            f"[[{slot}]] is filled by no probe this rubric grades, so its judge is always told "
            f"\"None were given for this probe.\"{hint}",
        ))
    return problems


def _reply_format_problems(name: str, prompt: str) -> list[Problem]:
    """Whether the judge is told to answer in the form PyRIT reads, and that true means a violation."""
    if "score_value" not in prompt:
        return [Problem(
            "error", name,
            'never tells the judge to answer with a JSON object with "score_value" and "rationale". '
            "PyRIT reads the verdict from that, so every score would fail to parse.",
        )]
    problems = []
    last = prompt.strip().split("\n\n")[-1]
    if not all(word in last for word in ("JSON", "score_value", "rationale")):
        problems.append(Problem(
            "warning", name,
            'the instruction to answer with JSON ("score_value", "rationale") is not the last paragraph. '
            "The judge reads the end last; put it there.",
        ))
    if not re.search(r"\btrue\b[^.\n]{0,40}violat", prompt, re.IGNORECASE):
        problems.append(Problem(
            "warning", name,
            "never says that true means a violation. Reports count true as a finding, so if this rubric "
            'means the opposite, every result is inverted. The shipped rubrics say "Set score_value to '
            'true (VIOLATION)".',
        ))
    return problems


def _judge_text_problems(entries) -> list[Problem]:
    """Whatever in the judge_text section the kit cannot use, or the judge would misread."""
    if not isinstance(entries, dict):
        return [Problem(
            "error", "judge_text",
            "the rubric file needs a `judge_text:` section: the words the kit wraps around the rubrics. "
            "Copy it from the starter pack's judge.yaml.",
        )]
    problems = []
    for name, slots in JUDGE_TEXT_SLOTS.items():
        where = f"judge_text.{name}"
        value = entries.get(name)
        if not isinstance(value, str) or not value.strip():
            problems.append(Problem("error", where, "is missing or empty; the judge is sent it."))
            continue
        used = set(SLOT.findall(value))
        for slot in slots:
            if slot in used:
                continue
            if slot in EVIDENCE_SLOTS:
                problems.append(Problem("error", where, f"needs [[{slot}]]: without it the judge never sees what fills it."))
            else:
                problems.append(Problem("warning", where, f"has no [[{slot}]], so the judge is not told it."))
        for slot in sorted(used - set(slots)):
            guess = difflib.get_close_matches(slot, slots, n=1)
            hint = f" Did you mean [[{guess[0]}]]?" if guess else ""
            problems.append(Problem("warning", where, f"[[{slot}]] is not filled here; the judge sees it as written.{hint}"))
        for field in sorted(set(re.findall(r"\{(\w+)\}", value))):
            problems.append(Problem(
                "warning", where,
                f"{{{field}}} reaches the judge as written: judge_text is plain text, filled only through its [[SLOTS]].",
            ))
    grounding = entries.get("grounding")
    if isinstance(grounding, str) and grounding.strip() and "score_value" not in grounding:
        problems.append(Problem(
            "error", "judge_text.grounding",
            'never tells the judge to answer with a JSON object with "score_value" and "rationale".',
        ))
    for name in sorted(set(entries) - set(JUDGE_TEXT_SLOTS)):
        guess = difflib.get_close_matches(name, list(JUDGE_TEXT_SLOTS), n=1)
        hint = f" Did you mean `{guess[0]}`?" if guess else ""
        problems.append(Problem("warning", f"judge_text.{name}", f"is not one the kit uses, so the judge never reads it.{hint}"))
    return problems


def check(rubrics: dict | None = None, probes=None, objectives: bool = False,
          judge_text_entries: dict | None = None) -> list[Problem]:
    """Everything wrong with the rubrics and the judge_text section, errors first.

    `probes`, if given, are every probe in every suite: the check then also says when
    a [[SLOT]] is filled by none of the probes a rubric grades. With `objectives`, it
    also checks every profile's multi-turn objectives name a rubric that can judge them.
    A campaign leaves that out: the multi-turn campaigns check their own profile's when
    they load them, and a scan uses none. `judge_text_entries` defaults to the file's.
    """
    rubrics = load_rubrics() if rubrics is None else rubrics
    if judge_text_entries is None:
        judge_text_entries = read_rubric_file().get("judge_text")
    probe_slots: dict[str, set[str]] | None = None
    if probes is not None:
        probe_slots = {}
        for probe in probes:
            probe_slots.setdefault(probe.dataset, set()).update(slot for slot, _ in probe.judge_context)

    problems: list[Problem] = []
    for name, rubric in rubrics.items():
        if not isinstance(rubric, dict):
            problems.append(Problem("error", name, "must be a set of fields (system_prompt, grades, ...), not a value."))
            continue
        prompt = rubric.get("system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append(Problem("error", name, "needs a `system_prompt:`, the text the judge is given."))
            continue
        pair = name in PAIR_RUBRICS

        grades = rubric.get("grades")
        if grades is None:
            problems.append(Problem("warning", name, "has no `grades:` list, so it grades no probes."))
            grades = []
        elif not isinstance(grades, list) or not all(isinstance(g, str) for g in grades):
            problems.append(Problem("error", name, "`grades:` must be a list of dataset names, e.g. [prompt_injection]."))
            grades = []
        for dataset in grades:
            if not dataset_path(dataset).exists():
                problems.append(Problem(
                    "error", name, f"grades {dataset!r}, which has no attack_suites/*/{dataset}.yaml.",
                ))

        severity = rubric.get("severity")
        if severity is None:
            problems.append(Problem("warning", name, "has no `severity:`; finding reports will assume high."))
        elif severity not in SEVERITIES:
            problems.append(Problem(
                "error", name, f"severity {severity!r} is not one of: {', '.join(SEVERITIES)}.",
            ))
        if not str(rubric.get("description") or "").strip():
            problems.append(Problem(
                "warning", name, "has no `description:`; the run summary and finding reports show it.",
            ))

        problems += _format_problems(name, prompt, pair)
        problems += _slot_problems(name, prompt, pair, probe_slots, grades)
        problems += _reply_format_problems(name, prompt)

    problems += _judge_text_problems(judge_text_entries)
    if objectives:
        problems += _objective_problems(rubrics)
    return sorted(problems, key=lambda p: p.level != "error")


def _objective_problems(rubrics: dict) -> list[Problem]:
    """Multi-turn objectives judged by a rubric that does not exist, or cannot judge one reply."""
    from harnesses.pyrit_campaigns.profiles import PROFILES

    problems = []
    for profile in PROFILES.values():
        for number, objective in enumerate(profile.objectives, 1):
            where = f"{profile.objectives_file or profile.key}, objective {number}"
            if objective.rubric not in rubrics:
                problems.append(Problem("error", "", f"{where} is judged by {objective.rubric!r}, which is not a rubric."))
            elif objective.rubric in PAIR_RUBRICS:
                problems.append(Problem(
                    "error", "", f"{where} is judged by {objective.rubric!r}, which compares two replies.",
                ))
    return problems


def require_valid(rubrics: dict | None = None, probes=None) -> None:
    """Print the check's warnings, and stop with its errors. Every campaign calls this before sending."""
    problems = check(rubrics, probes)
    errors = [p for p in problems if p.level == "error"]
    for problem in problems:
        if problem.level == "warning":
            print(f"!! Rubric {problem}")
    if errors:
        raise SystemExit(
            f"\nThe rubrics in {_shown(RUBRIC_FILE)} have {len(errors)} error(s). Nothing was sent.\n"
            + "\n".join(f"  - {p}" for p in errors)
            + "\nFix them, then check again with: python -m harnesses.pyrit_campaigns.rubrics check"
        )


# --- which version judged a run -------------------------------------------------

def _git(*args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def rubric_fingerprint(path: Path | None = None) -> str:
    """The first 12 characters of the rubric file's SHA-256: the same text gives the same fingerprint.

    Line endings are made \\n first, so a Windows checkout (CRLF) and a Linux CI runner
    give the same fingerprint for the same committed file.
    """
    return hashlib.sha256((path or RUBRIC_FILE).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def version_labels() -> dict[str, str]:
    """What every run records about the rubrics that judged it.

    rt_pack_version is `git describe`: the nearest release tag and commit, with
    "-dirty" when the working tree has uncommitted changes. "unknown" without git, as
    in a downloaded copy.
    """
    return {
        "rt_pack_version": _git("describe", "--tags", "--always", "--dirty") or "unknown",
        "rt_rubric_sha": rubric_fingerprint(),
    }


def describe_version(labels: dict) -> str:
    """One line for a report: the rubric fingerprint and pack version a run recorded."""
    sha = labels.get("rt_rubric_sha")
    if not sha:
        return "not recorded (the run predates rubric versions)"
    return f"`judge.yaml` fingerprint `{sha}`, pack `{labels.get('rt_pack_version') or 'unknown'}`"


# --- command line ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the judge's rubrics before a run.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check the rubric file; exits 1 if it has errors")
    args = parser.parse_args(argv)

    if args.command == "check":
        from harnesses.pyrit_campaigns.single_turn_scan import load_probes

        rubrics = load_rubrics()
        problems = check(rubrics, load_probes(), objectives=True)
        for problem in problems:
            print(problem)
        errors = sum(p.level == "error" for p in problems)
        warnings = len(problems) - errors
        print(
            f"\n{_shown(RUBRIC_FILE)}: {len(rubrics)} rubric(s), {errors} error(s), {warnings} warning(s). "
            f"Fingerprint {rubric_fingerprint()}."
        )
        if errors:
            print("Errors stop every campaign before it sends anything.")
        return 1 if errors else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

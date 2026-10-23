"""The judge's rubrics: reading the rubric file, checking it, and saying which version judged a run.

Every campaign reads its rubrics from harnesses/pyrit_campaigns/state_policy_rubric.yaml
through load_rubrics(). Before anything is sent, each campaign checks the file, so a
mistake in an edited rubric stops the run with what to fix rather than breaking at the
first judge call - or worse, not breaking and judging wrongly:

    python -m harnesses.pyrit_campaigns.rubrics check

ERRORS stop a run. They are the edits that break the kit: a `{brace}` PyRIT cannot fill,
a pair rubric with nowhere for the two replies, no instruction to answer with
score_value, a `grades:` entry naming no suite, a severity the reports do not know, an
objective judged by a rubric that does not exist.

WARNINGS do not. They are the edits that still run but probably do not do what was
meant: a [[SLOT]] nothing fills, a rubric that never says true means a violation, the
reply-format instruction somewhere other than last, a missing description. Your fork's
rubrics are yours to word as you like; the check only holds you to what the rest of the
kit relies on.

Two things are fixed, because the rest of the kit depends on them:
  - The judge answers with a JSON object with "score_value" and "rationale". PyRIT
    reads the verdict from it.
  - true means a violation. The run summary, the finding export, and the CI gate all
    count true as a finding.

VERSIONS: every run records the rubric file's fingerprint and the pack's git version
(version_labels()), and the run summary and finding reports print them, so a finding
says which rubrics judged it.
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

RUBRIC_FILE = Path(__file__).resolve().parent / "state_policy_rubric.yaml"

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


def check(rubrics: dict | None = None, probes=None, objectives: bool = False) -> list[Problem]:
    """Everything wrong with the rubrics, errors first.

    `probes`, if given, are every probe in every suite: the check then also says when
    a [[SLOT]] is filled by none of the probes a rubric grades. With `objectives`, it
    also checks every profile's multi-turn objectives name a rubric that can judge them.
    A campaign leaves that out: the multi-turn campaigns check their own profile's when
    they load them, and a scan uses none.
    """
    rubrics = load_rubrics() if rubrics is None else rubrics
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
    return f"`state_policy_rubric.yaml` fingerprint `{sha}`, pack `{labels.get('rt_pack_version') or 'unknown'}`"


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

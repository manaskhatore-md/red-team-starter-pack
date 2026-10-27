"""One command for the whole kit: `rt scan`, `rt crescendo`, `rt findings`, ...

Each rt command runs one of the kit's modules exactly as its full form would, and
prints that full form first:

    rt scan --profile public_conversational --probes 1-15
    Running: python -m harnesses.pyrit_campaigns.single_turn_scan --probes 1-15
      with RT_PROFILE=public_conversational

The long `python -m ...` commands keep working and keep meaning the same thing; rt is
a shorter way to type them.

Four options are rt's own. --profile, --provider, --judge, and --repeats set
RT_PROFILE, RT_PROVIDER, RT_JUDGE_PROVIDER, and RT_REPEATS for that one command. They
beat .env, because load_dotenv() never replaces a setting already in the environment,
and unlike PowerShell's $env: they are gone when the command ends. Every other option
goes to the module unchanged and in order, so the module's own parser, not this one,
decides what `--probes 1-15` means. tests/test_rt_cli.py fails if a module ever gets an
option with one of rt's names.

Three ways to run it, all this file:

    rt scan                  # after `pip install -e .` once, from anywhere (see pyproject.toml)
    .\\rt scan               # no install, from the repo root: rt.cmd on Windows, ./rt elsewhere
    python -m redteam scan   # no install, from the repo root

Each command runs as its own process. The module's exit code comes back unchanged
(RT_FAIL_ON_FINDING relies on it), and the campaigns `rt all` runs one after another
share no state: in one process they would share PyRIT's database connection.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# rt's own options: each sets one setting, for the one command it is given to.
SETTINGS = {
    "--profile": "RT_PROFILE",
    "--provider": "RT_PROVIDER",
    "--judge": "RT_JUDGE_PROVIDER",
    "--repeats": "RT_REPEATS",
}
SETTING_HELP = {
    "--profile": "e.g. public_conversational; see Deployment profiles in the README",
    "--provider": "the model under test: app, gemini, anthropic, openai, bedrock, vertex",
    "--judge": "the model that grades the replies (same choices as --provider)",
    "--repeats": "send each probe or objective N times",
}
# rt all's own option. Like SETTINGS, no module may have an option by this name.
WITH_ACCESS = "--with-access"

SCAN = "harnesses.pyrit_campaigns.single_turn_scan"
EXPORT = "scripts.export_finding_report"
SETUP, ATTACK, NEEDS_SETUP, RESULTS = "Setup", "Attack", "Needs setup first", "Results"


@dataclass(frozen=True)
class Command:
    module: str
    group: str
    summary: str
    # Passed before the user's own options: `rt probes` is the scan with --list-probes.
    fixed: tuple[str, ...] = ()
    # False for a module that reads no options: it would ignore them, so rt refuses them.
    takes_args: bool = True


COMMANDS = {
    "smoke": Command(
        "scripts.smoke_test", SETUP,
        "check credentials and network: two benign prompts, replies printed", takes_args=False,
    ),
    "probes": Command(
        SCAN, SETUP, "list the probes a scan would send, numbered, and what grades each; calls no model",
        ("--list-probes",),
    ),
    "rubrics": Command(
        "harnesses.pyrit_campaigns.rubrics", SETUP,
        "check judge.yaml; every campaign also does this before it sends anything", ("check",),
    ),
    "scan": Command(SCAN, ATTACK, "single-turn scan: each of the profile's probes, one request each"),
    "adaptive": Command(
        "harnesses.pyrit_campaigns.multi_turn_red_team", ATTACK,
        "multi-turn: an attacker model adapts to each reply (also: rt red-team)",
    ),
    "crescendo": Command(
        "harnesses.pyrit_campaigns.multi_turn_crescendo", ATTACK,
        "multi-turn: gradual escalation that rewinds refusals; an app must take {HISTORY}",
    ),
    "access": Command("harnesses.access_checks", ATTACK, "login gates, OAuth/API access controls, outsider accounts"),
    "all": Command(
        "", ATTACK,
        f"scan, adaptive, then crescendo, carrying on past one that fails; {WITH_ACCESS} runs access first",
    ),
    "agent": Command(
        "harnesses.pyrit_campaigns.agent_tool_exploitation", NEEDS_SETUP,
        "tool misuse: needs RT_TOOL_NAME, RT_RECORD_TYPE, and RT_PROGRAM_NAME", takes_args=False,
    ),
    "rag": Command(
        "harnesses.pyrit_campaigns.rag_grounding_eval", NEEDS_SETUP,
        "RAG grounding: needs data_fixtures/grounding_eval_set.json", takes_args=False,
    ),
    "summary": Command("scripts.run_summary", RESULTS, "write the newest run's full summary again (--run-id for another)"),
    "findings": Command(EXPORT, RESULTS, "export finding reports for the newest run (--run-id for another)"),
    "runs": Command(EXPORT, RESULTS, "list the runs and results in the database", ("--list",)),
    "rescore": Command("scripts.rescore", RESULTS, "judge a past scan's replies again with the rubrics as they are now"),
    "scrub": Command("scripts.scrub", RESULTS, "scrub personal data from .md reports in place (default: findings/)"),
}
ALIASES = {"red-team": "adaptive"}
# What rt all runs, in order. --with-access puts access first: it is quick, and a broken
# login gate is worth knowing about before the long campaigns.
ALL = ("scan", "adaptive", "crescendo")
HELP = {"-h", "--help"}


@dataclass(frozen=True)
class Step:
    """One module run: what rt starts, and what it prints first."""

    name: str
    module: str
    args: list[str]
    settings: dict[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]

    def shown(self) -> str:
        line = " ".join(["python", "-m", self.module, *map(_quoted, self.args)])
        if self.settings:
            line += "\n  with " + ", ".join(f"{key}={value}" for key, value in self.settings.items())
        return line


def _quoted(arg: str) -> str:
    return f'"{arg}"' if not arg or any(c in arg for c in " \t'&|<>;$`(){}") else arg


def split_settings(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Take rt's own options out of argv; return them as settings, and the rest in order.

    Only an exact name counts (`--profile x` or `--profile=x`), never an abbreviation, so
    rt cannot take an option meant for the module. Everything after `--` is left alone.
    """
    settings: dict[str, str] = {}
    rest: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token == "--":
            rest.append(token)
            rest.extend(tokens)
            break
        name, equals, value = token.partition("=")
        if name not in SETTINGS:
            rest.append(token)
            continue
        if not equals:
            value = next(tokens, "")
        if not value or value.startswith("-"):
            raise SystemExit(f"rt: {name} needs a value: {SETTING_HELP[name]}.")
        settings[SETTINGS[name]] = value
    return settings, rest


def findings_dir() -> str:
    """The kit's findings/ folder, written the way the module would resolve it from here."""
    return "findings" if Path.cwd().resolve() == REPO_ROOT else str(REPO_ROOT / "findings")


def _given(args: list[str]) -> set[str]:
    return {arg.partition("=")[0] for arg in args if arg.startswith("--")}


def findings_args(args: list[str]) -> list[str]:
    """`rt findings` exports what a judge flagged in the newest run, unless told otherwise.

    --rubric any covers every campaign, not only the scan: multi-turn judges label their
    scores with the rubric too, and on a Crescendo run it exported the same reports as
    --outcome success (checked against a real run, 2026-10-26).
    """
    given = _given(args)
    added = []
    if not given & {"--rubric", "--outcome", "--conversation-id", "--list"}:
        added += ["--rubric", "any"]
    if not given & {"--run-id", "--latest-run", "--conversation-id", "--list"}:
        added.append("--latest-run")
    # The module's --out default is findings/ under the current folder; rt means the kit's.
    if "--out" not in given and findings_dir() != "findings":
        added += ["--out", findings_dir()]
    return added + args


def scrub_args(args: list[str]) -> list[str]:
    return args or [findings_dir()]


DEFAULTS = {"findings": findings_args, "scrub": scrub_args}
DEFAULT_NOTES = {
    "findings": "Adds --rubric any and --latest-run unless you give a filter or a run "
    "(--rubric, --outcome, --run-id, --conversation-id, or --list).",
    "scrub": "With no path, scrubs every .md file in findings/.",
}


def resolve(name: str) -> str:
    name = ALIASES.get(name, name)
    if name in COMMANDS:
        return name
    close = difflib.get_close_matches(name, [*COMMANDS, *ALIASES], n=1)
    hint = f" Did you mean rt {close[0]}?" if close else ""
    raise SystemExit(f"rt: no command {name!r}.{hint} Run rt on its own to list them.")


def plan(argv: list[str]) -> list[Step]:
    """What `rt ARGV` runs, without running it."""
    settings, rest = split_settings(argv)
    if not rest or rest[0].startswith("-"):
        raise SystemExit("rt: name a command first, e.g. rt scan. Run rt on its own to list them.")
    name, args = resolve(rest[0]), rest[1:]
    if name == "all":
        return plan_all(settings, args)
    command = COMMANDS[name]
    if args and not command.takes_args:
        raise SystemExit(
            f"rt {name} reads no options of its own, so it would ignore {' '.join(args)}. "
            f"It takes rt's {', '.join(SETTINGS)}, and the rest of its settings from .env."
        )
    args = DEFAULTS.get(name, list)(args)
    return [Step(name, command.module, [*command.fixed, *args], settings)]


def plan_all(settings: dict[str, str], args: list[str]) -> list[Step]:
    extra = [arg for arg in args if arg != WITH_ACCESS]
    if extra:
        raise SystemExit(
            f"rt all takes only {', '.join(SETTINGS)} and {WITH_ACCESS}, since the campaigns have different "
            f"options. Give {' '.join(extra)} to one campaign at a time, e.g. rt scan {' '.join(extra)}."
        )
    names = (("access",) if WITH_ACCESS in args else ()) + ALL
    return [Step(name, COMMANDS[name].module, list(COMMANDS[name].fixed), settings) for name in names]


def full_form(rt_args: str) -> str:
    """The python -m form of `rt RT_ARGS`, for a message that gives a command both ways.

    It comes from plan(), so a message cannot name a different command than rt runs.
    """
    (step,) = plan(rt_args.split())
    if step.settings:
        raise ValueError(f"rt {rt_args}: a setting has no python -m form; name it in .env instead")
    return step.shown()


def command_for(module: str) -> str:
    """The rt command that runs MODULE as it is: crescendo for ...multi_turn_crescendo."""
    return next(
        name for name, c in COMMANDS.items() if c.module == module and not c.fixed and name not in DEFAULTS
    )


def module_env(settings: dict[str, str]) -> dict[str, str]:
    env = {**os.environ, **settings}
    # The kit is found through PYTHONPATH, so rt works from any folder. PYTHONSAFEPATH
    # keeps the folder rt is run from off the module path, so a scripts/ folder there
    # cannot stand in for the kit's (Python 3.11+; older versions ignore it).
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)
    env["PYTHONSAFEPATH"] = "1"
    return env


def run(step: Step) -> int:
    print(f"Running: {step.shown()}\n", flush=True)
    process = subprocess.Popen(step.argv(), env=module_env(step.settings))
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Ctrl+C reached the module too; let it finish stopping before rt does.
        process.wait()
        raise


def run_all(steps: list[Step]) -> int:
    finished = []
    for step in steps:
        try:
            finished.append((step.name, run(step)))
        except KeyboardInterrupt:
            print(f"\nrt all: stopped during {step.name}; the rest did not run.")
            return 130
        print()
    print("rt all:")
    for name, code in finished:
        print(f"  {name:<10} {'finished' if code == 0 else f'exit {code}: its own output above says why'}")
    return 0 if all(code == 0 for _, code in finished) else 1


def usage() -> str:
    lines = [
        "rt - one command for the Red Team Starter Pack",
        "",
        "Usage: rt COMMAND [options]",
        "  (no install: .\\rt in PowerShell or cmd, ./rt elsewhere, or python -m redteam, from the repo root)",
    ]
    for group in (SETUP, ATTACK, NEEDS_SETUP, RESULTS):
        lines += ["", group]
        lines += [f"  {name:<10}  {c.summary}" for name, c in COMMANDS.items() if c.group == group]
    lines += ["", "Settings for one command (instead of .env, or $env: in PowerShell)"]
    lines += [
        f"  {name + (' N' if name == '--repeats' else ' NAME'):<16}  {setting:<17}  {SETTING_HELP[name]}"
        for name, setting in SETTINGS.items()
    ]
    lines += [
        "",
        "Every other option goes to the command unchanged, e.g. rt scan --probes 1-15 --converters base64.",
        "rt COMMAND --help lists that command's own options. Each command prints its full python -m form first.",
    ]
    return "\n".join(lines)


def command_help(name: str) -> int:
    """`rt NAME --help`: rt's settings, then the module's own options."""
    command = COMMANDS[name]
    print(f"rt {name}: {command.summary}")
    if name == "all":
        print(f"Runs rt {', rt '.join(ALL)}, each with rt's own settings: {', '.join(SETTINGS)}.")
        return 0
    print(f"Runs python -m {command.module}{''.join(' ' + a for a in command.fixed)}.")
    if name in DEFAULT_NOTES:
        print(DEFAULT_NOTES[name])
    print(f"rt's own settings: {', '.join(SETTINGS)} (run rt on its own for details).")
    if not command.takes_args:
        return 0
    print(f"\nThe options of python -m {command.module}:\n", flush=True)
    return subprocess.call([sys.executable, "-m", command.module, *command.fixed, "--help"], env=module_env({}))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in HELP | {"help"}:
        print(usage())
        return 0
    if not (REPO_ROOT / "harnesses").is_dir():
        raise SystemExit(
            f"rt cannot find the kit next to {Path(__file__).parent}. Install it with "
            "`pip install -e .` from the repo, not `pip install .`, so rt runs the clone itself."
        )
    if HELP & set(argv) and "--" not in argv:
        _, rest = split_settings([arg for arg in argv if arg not in HELP])
        if not rest:
            print(usage())
            return 0
        return command_help(resolve(rest[0]))
    steps = plan(argv)
    try:
        return run(steps[0]) if len(steps) == 1 else run_all(steps)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

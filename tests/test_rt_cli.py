"""The rt command (redteam/cli.py): it must run each module exactly as its full form would."""

import ast
import functools
import re
import shlex
import importlib.util
import sys
from pathlib import Path

import pytest

from redteam import cli
from redteam.cli import ALL, COMMANDS, REPO_ROOT, SETTINGS, WITH_ACCESS, Step, plan


def _only(argv: str) -> Step:
    (step,) = plan(argv.split())
    return step


@functools.cache
def _module_options() -> dict[str, set[str]]:
    """Every option string any module passes to add_argument, by file."""
    found = {}
    for path in [*(REPO_ROOT / "harnesses").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found[path.relative_to(REPO_ROOT).as_posix()] = {
            arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("-")
        }
    return found


def test_probe_selection_reaches_the_scan_unchanged():
    # The scan's own parser decides what --probes means; rt only passes it on, in order.
    step = _only("scan --probes 1-15 --converters base64+rot13 --no-control")
    assert step.module == "harnesses.pyrit_campaigns.single_turn_scan"
    assert step.args == ["--probes", "1-15", "--converters", "base64+rot13", "--no-control"]
    assert _only("scan --probes=3,7,12").args == ["--probes=3,7,12"]


def test_settings_are_taken_out_and_set_for_the_one_command():
    step = _only("scan --profile public_conversational --probes core --repeats=3")
    assert step.args == ["--probes", "core"]
    assert step.settings == {"RT_PROFILE": "public_conversational", "RT_REPEATS": "3"}
    # Before the command name works too.
    assert _only("--provider app --judge gemini adaptive").settings == {
        "RT_PROVIDER": "app", "RT_JUDGE_PROVIDER": "gemini",
    }


def test_only_exact_setting_names_are_taken():
    # argparse would read --prof as --profile. rt does not, so it never takes an option
    # meant for the module, such as access_checks' --outsider-profile.
    step = _only("access --prof x --outsider-profile C:/profile --profiles y")
    assert step.args == ["--prof", "x", "--outsider-profile", "C:/profile", "--profiles", "y"]
    assert step.settings == {}


def test_everything_after_a_double_dash_is_left_alone():
    assert _only("scan -- --profile kept").args == ["--", "--profile", "kept"]


@pytest.mark.parametrize("argv", ["scan --profile", "scan --profile --probes 1", "scan --repeats="])
def test_a_setting_needs_a_value(argv):
    with pytest.raises(SystemExit, match="needs a value"):
        plan(argv.split())


def test_no_module_has_an_option_named_like_rts_own():
    # If one did, rt would take it as a setting and the module would never see it.
    options = _module_options()
    assert "--probes" in options["harnesses/pyrit_campaigns/single_turn_scan.py"], "the AST walk found nothing"
    ours = {*SETTINGS, WITH_ACCESS}
    clashes = {path: names & ours for path, names in options.items() if names & ours}
    assert not clashes


def test_every_command_runs_a_module_that_exists():
    for name, command in COMMANDS.items():
        if name != "all":
            assert importlib.util.find_spec(command.module), f"rt {name}: no module {command.module}"


def test_commands_that_refuse_options_run_modules_that_read_none():
    # A module without a parser ignores any option, so rt refuses them instead. A module
    # that gains a parser should start taking them.
    for name, command in COMMANDS.items():
        if name == "all":
            continue
        source = Path(importlib.util.find_spec(command.module).origin).read_text(encoding="utf-8")
        reads_options = "parse_args" in source or "sys.argv" in source
        assert reads_options == command.takes_args, f"rt {name}"
    with pytest.raises(SystemExit, match="reads no options"):
        plan(["smoke", "--verbose"])
    assert _only("smoke --profile x").settings == {"RT_PROFILE": "x"}


def test_fixed_options_go_before_the_users():
    assert _only("probes --probes 1-15").args == ["--list-probes", "--probes", "1-15"]
    assert _only("runs --limit 100").args == ["--list", "--limit", "100"]
    assert _only("rubrics").args == ["check"]
    assert _only("red-team --show-judge-prompt 3").name == "adaptive"


def test_findings_exports_what_a_judge_flagged_in_the_newest_run(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert _only("findings").args == ["--rubric", "any", "--latest-run"]
    assert _only("findings --run-id 1a2b3c4d").args == ["--rubric", "any", "--run-id", "1a2b3c4d"]
    assert _only("findings --outcome failure").args == ["--latest-run", "--outcome", "failure"]
    assert _only("findings --rubric=ungrounded_claim").args == ["--latest-run", "--rubric=ungrounded_claim"]
    assert _only("findings --list").args == ["--list"]
    assert _only("findings --conversation-id abc").args == ["--conversation-id", "abc"]


def test_findings_and_scrub_mean_the_kits_findings_folder_from_anywhere(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    assert _only("scrub").args == ["findings"]
    assert _only("scrub findings/one.md").args == ["findings/one.md"]
    monkeypatch.chdir(tmp_path)
    kit_findings = str(REPO_ROOT / "findings")
    assert _only("scrub").args == [kit_findings]
    assert _only("findings").args[-2:] == ["--out", kit_findings]
    assert "--out" not in _only("findings --out reviews").args[:-2]


def test_all_runs_the_campaigns_in_order_with_the_same_settings():
    steps = plan(["all", "--profile", "p"])
    assert [s.name for s in steps] == list(ALL)
    assert all(s.settings == {"RT_PROFILE": "p"} and s.args == [] for s in steps)
    assert [s.name for s in plan(["all", WITH_ACCESS])] == ["access", *ALL]
    with pytest.raises(SystemExit, match="one campaign at a time"):
        plan(["all", "--probes", "1-5"])


def test_all_carries_on_past_a_campaign_that_fails(monkeypatch, capsys):
    codes = iter([1, 0, 0])
    ran = []
    monkeypatch.setattr(cli, "run", lambda step: ran.append(step.name) or next(codes))
    assert cli.run_all(plan(["all"])) == 1
    assert ran == list(ALL)
    out = capsys.readouterr().out
    assert "scan       exit 1" in out and "crescendo  finished" in out


def test_ctrl_c_stops_all(monkeypatch, capsys):
    def run(step):
        if step.name == "adaptive":
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(cli, "run", run)
    assert cli.run_all(plan(["all"])) == 130
    assert "stopped during adaptive" in capsys.readouterr().out


def test_unknown_command_suggests_the_closest():
    with pytest.raises(SystemExit, match="Did you mean rt scan"):
        plan(["scna"])


def test_shown_command_is_the_full_form():
    step = _only("scan --profile p --probes 1-15")
    assert step.shown() == "python -m harnesses.pyrit_campaigns.single_turn_scan --probes 1-15\n  with RT_PROFILE=p"
    assert step.argv()[:3] == [sys.executable, "-m", "harnesses.pyrit_campaigns.single_turn_scan"]


def test_a_module_runs_with_the_settings_and_its_exit_code_comes_back(monkeypatch, tmp_path, capfd):
    # A real child process, run from a folder holding a decoy scripts/ package: the kit's
    # scripts/ must win, which PYTHONSAFEPATH (cwd off the module path) makes sure of.
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "rt_fake_module.py").write_text(
        "import os, sys, scripts\n"
        "print('profile', os.environ['RT_PROFILE'], 'scripts', scripts.__file__)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "scripts").mkdir(parents=True)
    (elsewhere / "scripts" / "__init__.py").write_text("raise SystemExit('decoy imported')\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("PYTHONPATH", str(modules))
    assert cli.run(Step("fake", "rt_fake_module", [], {"RT_PROFILE": "from-flag"})) == 3
    out = capfd.readouterr().out
    assert "Running: python -m rt_fake_module" in out
    assert f"profile from-flag scripts {REPO_ROOT / 'scripts' / '__init__.py'}" in out


def _options_of(module: str) -> set[str]:
    """The options a module's parser takes, including a parser it imports from another module."""
    path = Path(importlib.util.find_spec(module).origin)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options = _module_options()[path.relative_to(REPO_ROOT).as_posix()]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "parse_args" for a in node.names):
            options |= _options_of(node.module)
    return options


def _readme_rt_commands() -> list[str]:
    """Every rt command line in the README: in code blocks, and in `inline code`."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    lines = [line.split("  #")[0].strip() for block in re.findall(r"```(?:bash)?\n(.*?)```", readme, re.DOTALL)
             for line in block.splitlines()]
    lines += re.findall(r"`([^`\n]*\brt [^`\n]*)`", readme)
    commands = []
    for line in lines:
        line = re.sub(r"^(?:> )?(?:[A-Z_]+=\S+ )*", "", line)
        if line.startswith("rt ") and "COMMAND" not in line and "<" not in line and "..." not in line:
            commands.append(line)
    return commands


def test_the_readmes_rt_commands_run_and_every_option_exists():
    commands = _readme_rt_commands()
    assert len(commands) > 40, "the README scan found too few rt commands"
    for line in commands:
        for step in plan(shlex.split(line)[1:]):
            unknown = {a.partition("=")[0] for a in step.args if a.startswith("--")} - _options_of(step.module)
            assert not unknown, f"{line!r}: {step.module} has no {sorted(unknown)}"


def test_full_form_is_what_rt_runs():
    assert cli.full_form("findings --run-id 1a2b3c4d").startswith(
        "python -m scripts.export_finding_report --rubric any --run-id 1a2b3c4d"
    )
    assert cli.full_form("rubrics") == "python -m harnesses.pyrit_campaigns.rubrics check"
    assert cli.command_for("harnesses.pyrit_campaigns.multi_turn_crescendo") == "crescendo"
    assert cli.command_for("harnesses.pyrit_campaigns.multi_turn_red_team") == "adaptive"
    with pytest.raises(ValueError, match="no python -m form"):
        cli.full_form("scan --repeats 5")


def test_no_message_names_a_command_only_in_its_long_form():
    # Every command the kit prints, in the console or a report, gives the rt form too.
    # The long form alone is how these messages read before rt.
    long_only = []
    for path in [*(REPO_ROOT / "harnesses").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
        }
        # An f-string's literal pieces are Constants too; it is read whole, as the JoinedStr.
        pieces = {id(v) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for v in node.values}
        for node in ast.walk(tree):
            if isinstance(node, (ast.JoinedStr, ast.Constant)) and id(node) not in docstrings | pieces:
                text = ast.unparse(node) if isinstance(node, ast.JoinedStr) else str(node.value)
                if "python -m " in text and "rt " not in text and "full_form" not in text:
                    long_only.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    assert not long_only

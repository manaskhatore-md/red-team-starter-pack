"""The built-in default profile, which a run uses when RT_PROFILE is not set.

How RT_PROFILE is read, that nothing the default sends goes out as a {{ token }}, that
it fits the call limit, that --probes replaces its `core` selection, and that a run
against an app warns about authorization whatever the profile.
"""

import asyncio
import re

import pytest

from harnesses.pyrit_campaigns import multi_turn_crescendo, single_turn_scan, system_prompt as sp
from harnesses.pyrit_campaigns.planted_records import records_for, records_section
from harnesses.pyrit_campaigns.profiles import DEFAULT_PROFILE, PROFILES, get_profile, placeholders
from helpers import FakeModel, nothing

# From the warning every run against an app prints: "Runs against an app need a ...".
AUTHORIZATION_BANNER = "non-production instance, synthetic data only"
# Every probe in every suite.
ALL = single_turn_scan.load_probes()
TOTAL = len(ALL)


class Stop(Exception):
    """Raised where the scan would first call a model, so a test sees everything printed before it."""


def scan_until_sending(monkeypatch, probes_spec=None) -> None:
    """Run the scan's main() up to its first model call, so capsys holds what it printed before it."""
    async def stop(**targets):
        raise Stop

    for name in ("initialize_pyrit_async", "close_target"):
        monkeypatch.setattr(single_turn_scan, name, nothing)
    monkeypatch.setattr(single_turn_scan, "check_models", stop)
    monkeypatch.setattr(single_turn_scan, "build_target", lambda: FakeModel())
    monkeypatch.setattr(single_turn_scan, "build_scoring_target", lambda: FakeModel())
    with pytest.raises(Stop):
        asyncio.run(single_turn_scan.main(probes_spec))


@pytest.mark.parametrize("setting", [None, "", "  "])
def test_an_unset_or_blank_rt_profile_is_the_default(setting, clean_env):
    # .env.example ships RT_PROFILE= blank.
    if setting is not None:
        clean_env.setenv("RT_PROFILE", setting.strip())
    assert get_profile().key == DEFAULT_PROFILE


def test_rt_profile_is_read_in_any_case(clean_env):
    # An existing .env must keep working.
    clean_env.setenv("RT_PROFILE", "Public_Conversational")
    assert get_profile().key == "public_conversational"


def test_nothing_the_default_sends_has_an_unfilled_token():
    profile = get_profile()
    assert profile.key == DEFAULT_PROFILE
    # Every probe, not just core: --probes all sends the rest.
    probes = single_turn_scan.profile_probes(profile)
    records = records_section(records_for(single_turn_scan.rubrics_for(probes)))
    prompt = sp.load_system_prompt(profile.key, placeholders(), "RTC-0000-0000", records)
    goals = [o.goal for o in multi_turn_crescendo.load_objectives(profile)]
    texts = [p.prompt for p in probes] + [p.pass_criteria for p in probes] + [prompt.text] + goals
    assert not [t for t in texts if single_turn_scan.UNRENDERED.search(t)]
    assert goals and len(goals) == len(profile.objectives)


@pytest.mark.parametrize("probes_spec", [None, "all"])
def test_the_default_fits_the_call_limit_at_ten_repeats(probes_spec, monkeypatch, clean_env, memory, capsys):
    clean_env.setenv("RT_REPEATS", "10")
    scan_until_sending(monkeypatch, probes_spec)
    planned = re.search(r"about (\d+) target calls and (\d+) judge calls", capsys.readouterr().out)
    assert sum(map(int, planned.groups())) <= single_turn_scan.DEFAULT_MAX_CALLS


@pytest.mark.parametrize("profile_key", sorted(PROFILES))
def test_every_shipped_profile_fits_the_call_limit_at_ten_repeats(profile_key, monkeypatch, clean_env, memory, capsys):
    # The README promises it, and the scan stops unsent when it is not so.
    clean_env.setenv("RT_PROFILE", profile_key)
    clean_env.setenv("RT_REPEATS", "10")
    scan_until_sending(monkeypatch)
    planned = re.search(r"about (\d+) target calls and (\d+) judge calls", capsys.readouterr().out)
    assert sum(map(int, planned.groups())) <= single_turn_scan.DEFAULT_MAX_CALLS


def count(spec: str) -> int:
    return len(single_turn_scan.select_probes(ALL, spec))


def test_the_default_runs_core_unless_probes_says_otherwise(monkeypatch, clean_env, capsys):
    profile = get_profile()
    single_turn_scan.list_probes(profile)
    out = capsys.readouterr().out
    core = count("core")
    assert core < TOTAL
    assert f"Probe filter (core): {core} of {TOTAL} probe(s) selected." in out
    assert f"{core} of {core} probe(s) would be sent" in out

    single_turn_scan.list_probes(profile, "all")
    assert f"{TOTAL} of {TOTAL} probe(s) would be sent" in capsys.readouterr().out

    templated = count("bias-templated")
    single_turn_scan.list_probes(profile, "bias-templated")
    assert f"Probe filter (bias-templated): {templated} of {TOTAL} probe(s) selected." in capsys.readouterr().out

    hallucination = count("hallucination-templated")
    single_turn_scan.list_probes(profile, "hallucination-templated")
    assert f"Probe filter (hallucination-templated): {hallucination} of {TOTAL} probe(s) selected." in capsys.readouterr().out

    # Bias is the last suite, so the first probes are the general ones.
    general = count("general")
    assert single_turn_scan.select_probes(ALL, f"1-{general}") == single_turn_scan.select_probes(ALL, "general")
    clean_env.setenv("RT_PROBES", f"1-{general}")
    single_turn_scan.list_probes(profile)
    assert f"Probe filter (1-{general}): {general} of {TOTAL} probe(s) selected." in capsys.readouterr().out


def test_a_named_profile_runs_its_own_selection(capsys):
    single_turn_scan.list_probes(get_profile("internal_productivity"))
    out = capsys.readouterr().out
    own = count("prompt_injection,sensitive_data_leakage,excessive_agency")
    assert own < TOTAL
    assert f"Probe filter (prompt_injection,sensitive_data_leakage,excessive_agency): {own} of {TOTAL}" in out
    assert f"{own} of {own} probe(s) would be sent" in out
    single_turn_scan.list_probes(get_profile("constituent_decision"))
    assert f"{TOTAL} of {TOTAL} probe(s) would be sent" in capsys.readouterr().out


def test_narrowing_the_probes_keeps_the_profiles_planted_records(monkeypatch, clean_env, memory, capsys):
    # The records belong to the deployment the prompt describes, not to this run's selection.
    scan_until_sending(monkeypatch, "prompt_injection")
    assert "It also holds 2 fake records of other people" in capsys.readouterr().out


def test_a_run_against_an_app_warns_about_authorization_with_no_profile_set(monkeypatch, clean_env, memory, capsys):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("RT_JUDGE_PROVIDER", "gemini")
    scan_until_sending(monkeypatch)
    out = capsys.readouterr().out
    assert "PROFILE: default" in out and AUTHORIZATION_BANNER in out
    assert "Runs against an app need a non-production instance" in out


def test_no_profile_warns_about_authorization_against_a_model(monkeypatch, clean_env, memory, capsys):
    # The warning is about the app; a raw model is not anyone's system.
    clean_env.setenv("RT_PROFILE", "constituent_decision")
    scan_until_sending(monkeypatch)
    assert AUTHORIZATION_BANNER not in capsys.readouterr().out


def test_a_run_against_a_model_with_no_profile_set_does_not(monkeypatch, clean_env, memory, capsys):
    scan_until_sending(monkeypatch)
    assert AUTHORIZATION_BANNER not in capsys.readouterr().out

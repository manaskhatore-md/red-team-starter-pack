"""Modular probe selection: categories, 1-based numbers/ranges, names, and pair integrity."""

import re

import pytest
from pyrit.memory import CentralMemory

from harnesses.pyrit_campaigns import single_turn_scan as scan
from harnesses.pyrit_campaigns.profiles import PROFILES, SUITES, get_profile
from scripts import run_summary
from helpers import FakeModel, scan_main
from test_bias_pairs import PairJudge

# Every probe in every suite, numbered as --list-probes numbers them.
ALL = scan.load_probes()
TOTAL = len(ALL)


def number(name: str) -> int:
    """A probe's 1-based number in --list-probes."""
    return next(i for i, p in enumerate(ALL, 1) if p.name == name)


def test_category_keywords_partition_probes_cleanly():
    probes = scan.profile_probes(get_profile("constituent_decision"))
    assert probes == ALL

    general = scan.select_probes(probes, "general")
    core = scan.select_probes(probes, "core")
    bias = scan.select_probes(probes, "bias")
    bias_core = scan.select_probes(probes, "bias-core")
    bias_templated = scan.select_probes(probes, "bias-templated")

    # general = every suite but bias, in suite order; with bias, every probe.
    assert general == [p for suite in SUITES if suite != "algorithmic_bias" for p in scan.select_probes(probes, suite)]
    assert set(general) | set(bias) == set(probes) and not set(general) & set(bias)

    # bias = its inline pairs (bias-core), then the ones its fixtures make (bias-templated).
    assert bias_core and all(p.dataset == "algorithmic_bias" and not p.template for p in bias_core)
    assert bias_templated and all(p.dataset == "algorithmic_bias" and p.template for p in bias_templated)
    assert bias == bias_core + bias_templated

    # core = general + bias-core: everything but the templated bias probes, in any profile.
    assert set(core) == set(general) | set(bias_core)
    pub_probes = scan.profile_probes(get_profile("public_conversational"))
    assert set(scan.select_probes(pub_probes, "core")) == set(pub_probes) - set(bias_templated)

    # Underscore aliases and 'all' / empty spec also work.
    assert scan.select_probes(probes, "bias_core") == bias_core
    assert scan.select_probes(probes, "bias_templated") == bias_templated
    assert scan.select_probes(probes, "all") == probes
    assert scan.select_probes(probes, "") == probes


def test_selecting_by_name_dataset_pair_id_and_range():
    probes = scan.profile_probes(get_profile("constituent_decision"))

    # Single probe by name
    by_name = scan.select_probes(probes, "override_simple")
    assert [p.name for p in by_name] == ["override_simple"]

    # Dataset name
    by_dataset = scan.select_probes(probes, "prompt_injection")
    assert by_dataset and len(by_dataset) == sum(p.dataset == "prompt_injection" for p in probes)
    assert all(p.dataset == "prompt_injection" for p in by_dataset)

    # Pair id selects both halves directly
    by_pair = scan.select_probes(probes, "dialect_01")
    assert [p.name for p in by_pair] == ["dialect_aave", "dialect_sae"]

    # 1-based single index and m-n inclusive range, combined with a probe name
    by_range = scan.select_probes(probes, "1, 3-5, cross_session_bleed")
    assert [p.name for p in by_range] == [
        probes[0].name,
        probes[2].name,
        probes[3].name,
        probes[4].name,
        "cross_session_bleed",
    ]


def test_selecting_half_of_an_inline_pair_auto_includes_partner_in_file_order(capsys):
    probes = scan.profile_probes(get_profile("constituent_decision"))
    # dialect_sae is the second half; dialect_aave, the probe before it, is the first.
    selected = scan.select_probes(probes, "dialect_sae")
    out = capsys.readouterr().out

    assert [p.name for p in selected] == ["dialect_aave", "dialect_sae"]
    assert "auto-included 1 partner probe(s)" in out
    assert "dialect_aave (pair: dialect_01)" in out


def test_selecting_half_of_a_templated_pair_auto_includes_non_adjacent_partner_in_file_order(capsys):
    probes = scan.profile_probes(get_profile("constituent_decision"))

    def made_by(template):
        return [p for p in probes if p.template == template]

    # In algorithmic_bias.yaml, the scenario_dialect_standard[..] probes, one per fixture
    # entry, come just before the scenario_dialect_aave[..] probes.
    standard, aave = made_by("scenario_dialect_standard"), made_by("scenario_dialect_aave")
    assert len(standard) == len(aave) > 1
    assert probes.index(aave[0]) == probes.index(standard[-1]) + 1
    selected_one = scan.select_probes(probes, aave[0].name)
    assert selected_one == [standard[0], aave[0]]

    # Selecting one half's template name pulls in every partner from the other template.
    selected_template = scan.select_probes(probes, "scenario_spanish")
    assert selected_template == made_by("scenario_spanish_english") + made_by("scenario_spanish")

    # A range that crosses pair boundaries (the last standard probe to the first aave one)
    # pulls in each one's partner, in original file order.
    selected_range = scan.select_probes(probes, f"{number(standard[-1].name)}-{number(aave[0].name)}")
    assert selected_range == [standard[0], standard[-1], aave[0], aave[-1]]


OUT_OF_RANGE = re.escape(f"out of range: there are {TOTAL} probe(s) (1-{TOTAL})")


@pytest.mark.parametrize(
    "profile_key, spec, match",
    [
        ("constituent_decision", "no_such_probe", "Unknown probe selector 'no_such_probe'"),
        ("constituent_decision", "0", OUT_OF_RANGE),
        ("constituent_decision", str(TOTAL + 1), OUT_OF_RANGE),
        ("constituent_decision", f"1-{TOTAL + 1}", OUT_OF_RANGE),
        ("constituent_decision", "10-5", "has start > end: write 5-10 instead"),
        ("constituent_decision", " , ", "Empty probe selector"),
        # A subset with no bias probes, as a named profile's own spec gives.
        ("internal_productivity", "bias", "matched 0 probes"),
        ("internal_productivity", "bias-templated", "matched 0 probes"),
    ],
)
def test_invalid_or_empty_probe_selectors_fail_fast(profile_key, spec, match):
    probes = scan.profile_probes(get_profile(profile_key))
    with pytest.raises(SystemExit, match=match):
        scan.select_probes(probes, spec)


def test_list_probes_shows_original_numbers_and_supports_filtering(capsys):
    profile = get_profile("public_conversational")
    scan.list_probes(profile, "dialect_sae")
    out = capsys.readouterr().out

    assert f"Probe filter (dialect_sae): 2 of {TOTAL} probe(s) selected." in out
    # Numbered as in the full list, not 1 and 2.
    aave, sae = number("dialect_aave"), number("dialect_sae")
    assert sae == aave + 1 > 2
    assert f"[{aave}] algorithmic_bias / dialect_aave  [unauthorized_determination]  (pair: dialect_01)" in out
    assert f"[{sae}] algorithmic_bias / dialect_sae  [unauthorized_determination]  (pair: dialect_01)" in out
    assert "2 of 2 probe(s) would be sent" in out


def test_a_probe_has_the_same_number_under_every_profile(capsys):
    numbered = {}
    for key in PROFILES:
        scan.list_probes(get_profile(key))
        for number, dataset, name in re.findall(r"^\[(\d+)\] (\S+) / (\S+)", capsys.readouterr().out, re.MULTILINE):
            assert numbered.setdefault((dataset, name), number) == number, (key, dataset, name)
    assert len(numbered) == TOTAL


def test_probes_replaces_a_named_profiles_selection(capsys):
    # internal_productivity picks no bias probes; --probes picks from every suite.
    scan.list_probes(get_profile("internal_productivity"), "bias-core")
    bias_core = len(scan.select_probes(ALL, "bias-core"))
    assert f"Probe filter (bias-core): {bias_core} of {TOTAL} probe(s) selected." in capsys.readouterr().out


def test_scan_main_respects_rt_probes_and_records_filter_in_summary(memory, monkeypatch, capsys):
    # Select a single half of a pair via RT_PROBES; verify the scan auto-includes its partner,
    # sends only those 2 probes, compares the pair, and records the filter in the summary.
    monkeypatch.setenv("RT_PROBES", "dialect_sae")
    target = FakeModel()
    judge = PairJudge()
    exit_code, run_id = scan_main(monkeypatch, target, judge, profile="public_conversational")
    out = capsys.readouterr().out

    assert exit_code == 0
    assert f"Probe filter (dialect_sae): 2 of {TOTAL} probe(s) selected." in out
    assert "Sending 2 probes" in out and "1 matched pairs compared" in out
    assert len(target.seen) == 2

    results = [
        r for r in CentralMemory.get_memory_instance().get_attack_results()
        if (r.labels or {}).get("rt_run_id") == run_id
    ]
    assert {r.labels["rt_probe"] for r in results} == {"dialect_aave", "dialect_sae"}
    assert all(r.labels["rt_probes"] == "dialect_sae" for r in results)

    text = run_summary.build_summary(results, CentralMemory.get_memory_instance(), run_id)
    assert "| Probe filter | `dialect_sae` |" in text

"""The profiles, datasets, and rubrics agree with each other.

These catch the edits most likely to break a scan without any error message: a
profile naming a dataset that was renamed, a probe that no rubric grades, a new
{{ token }} with no default value, or a matched pair missing its other half.
"""

import re
from collections import Counter

import pytest

from pyrit_campaigns import profiles, single_turn_scan
from pyrit_campaigns.profiles import PAIR_RUBRICS, PROFILES, get_profile

RUBRICS = single_turn_scan.load_rubrics()
TOKEN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@pytest.fixture(params=sorted(PROFILES))
def profile(request):
    return get_profile(request.param)


def test_every_profile_rubric_exists(profile):
    assert set(profile.rubrics) <= set(RUBRICS)


JSON_INSTRUCTION = "Respond with only a JSON object with exactly these two keys"


def test_every_rubric_asks_for_json_once_at_the_end():
    # The judge's reply is parsed as JSON, and nothing in PyRIT asks for it. Two
    # differently worded instructions in one prompt leave the judge to pick one.
    for name, rubric in RUBRICS.items():
        prompt = rubric["system_prompt"]
        assert prompt.count("JSON") == 1, name
        assert JSON_INSTRUCTION in prompt.strip().splitlines()[-3], name


def test_every_rubric_labels_its_verdicts_with_score_value():
    for name, rubric in RUBRICS.items():
        prompt = rubric["system_prompt"]
        assert "Return true" not in prompt and "Return false" not in prompt, name
        assert "Set score_value to true (VIOLATION)" in prompt, name
        assert "Set score_value to false (ACCEPTABLE)" in prompt, name


def test_every_rubric_grades_only_real_datasets():
    for name, rubric in RUBRICS.items():
        for dataset in rubric.get("grades", []):
            assert (single_turn_scan.DATASETS_DIR / f"{dataset}.yaml").exists(), f"{name} grades {dataset}"


def test_every_probe_in_a_profile_gets_graded(profile):
    # A probe no rubric grades is not sent, so a gap here is a silent hole in coverage.
    probes = single_turn_scan.load_probes(profile)
    compare = bool(PAIR_RUBRICS & set(profile.rubrics))
    groups = single_turn_scan.group_by_rubrics(probes, profile.rubrics, compare=compare)
    sent = {p.name for group in groups.values() for p in group}
    assert {p.name for p in probes} == sent


def test_every_dataset_token_has_a_default():
    used = set()
    for path in single_turn_scan.DATASETS_DIR.glob("*.yaml"):
        used |= set(TOKEN.findall(path.read_text(encoding="utf-8")))
    assert used <= set(profiles._GENERIC_PLACEHOLDERS)


def test_matched_pairs_have_exactly_two_halves(profile):
    pairs = Counter(p.pair_id for p in single_turn_scan.load_probes(profile) if p.pair_id)
    assert all(count == 2 for count in pairs.values()), pairs


def test_placeholder_values_come_from_env(clean_env):
    clean_env.setenv("RT_PROGRAM_NAME", "Energy Assistance")
    assert get_profile("internal_productivity").placeholders["program_name"] == "Energy Assistance"


def test_unknown_profile_lists_the_options():
    with pytest.raises(SystemExit, match="Unknown RT_PROFILE"):
        get_profile("nope")


def test_placeholder_warning_names_the_env_var(capsys):
    profile = get_profile("public_conversational")
    single_turn_scan.check_placeholders(profile, single_turn_scan.load_probes(profile))
    assert "RT_PROGRAM_NAME=" in capsys.readouterr().out

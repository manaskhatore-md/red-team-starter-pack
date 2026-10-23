"""Excessive agency: false action claims, commitments, and exceptions, in every profile.

The suite is attack_suites/05_excessive_agency/; the excessive_agency rubric judges it
in the single-turn scan, and each profile's multi-turn objectives include one goal it
judges, which multi_turn_crescendo and multi_turn_red_team both run.
"""

import pytest

from harnesses.pyrit_campaigns import multi_turn_crescendo, multi_turn_red_team, probe_templates, single_turn_scan
from harnesses.pyrit_campaigns.profiles import PROFILES, dataset_path, get_profile
from helpers import house_style

# The whole suite, as every profile should send it.
SUITE = single_turn_scan.load_probes(["excessive_agency"])
# The framings are the suite's template seeds, the actions the entries in its fixture.
FRAMINGS = tuple(dict.fromkeys(p.template for p in SUITE if p.template))
ACTIONS = tuple(e["id"] for e in probe_templates.load_fixtures(dataset_path("excessive_agency").parent / "fixtures" / "actions.yaml")[1])
RUBRIC = single_turn_scan.load_rubrics()["excessive_agency"]


@pytest.fixture(params=sorted(PROFILES))
def profile(request):
    return get_profile(request.param)


def suite(profile):
    return [p for p in single_turn_scan.profile_probes(profile) if p.dataset == "excessive_agency"]


def test_every_action_is_asked_each_way():
    probes = suite(get_profile("public_conversational"))
    templated = [p for p in probes if p.template]
    assert FRAMINGS and ACTIONS
    assert [p.name for p in templated] == [f"{t}[{a}]" for t in FRAMINGS for a in ACTIONS]
    for probe in templated:
        # Each probe's criteria name its own action, and nothing is left unfilled.
        assert "{{" not in probe.prompt and "{{" not in probe.pass_criteria, probe.name
        assert "Does not say or imply that it " in probe.pass_criteria, probe.name
    by_name = {p.name: p for p in templated}
    assert "changed the mailing address" in by_name["action_claim_pressure[address_change]"].pass_criteria
    assert by_name["action_claim_plain[address_change]"].prompt.startswith("Can you change the mailing address")
    # The request runs into the sentence around it: no line break before the "?".
    assert by_name["action_claim_plain[address_change]"].prompt.rstrip().endswith("MD 21218?")


def test_every_profile_runs_the_suite_and_grades_it_with_the_rubric(profile):
    assert "excessive_agency" in single_turn_scan.profile_rubrics(profile)
    probes = suite(profile)
    assert [p.name for p in probes] == [p.name for p in SUITE]
    groups = single_turn_scan.group_by_rubrics(probes, single_turn_scan.profile_rubrics(profile), compare=False)
    # The suite is graded solely by excessive_agency (not double-graded by unauthorized_determination).
    assert set(groups) == {("excessive_agency",)}


def test_each_framing_and_each_action_can_be_run_on_its_own():
    probes = single_turn_scan.profile_probes(get_profile("constituent_decision"))
    assert single_turn_scan.select_probes(probes, "excessive_agency") == SUITE
    pressure = single_turn_scan.select_probes(probes, "action_claim_pressure")
    assert [p.name for p in pressure] == [f"action_claim_pressure[{a}]" for a in ACTIONS]
    one = single_turn_scan.select_probes(probes, "action_claim_authority[flag_urgent]")
    assert [p.name for p in one] == ["action_claim_authority[flag_urgent]"]
    assert [p.name for p in single_turn_scan.select_probes(probes, "override_fee_waiver")] == ["override_fee_waiver"]


def test_the_rubric_uses_the_trace_when_there_is_one():
    # The scan shows the judge the app's tool calls in <response_trace>.
    assert "<response_trace>" in RUBRIC["system_prompt"]


@house_style
def test_the_rubric_says_how_to_read_the_trace():
    prompt = RUBRIC["system_prompt"]
    # A "done" backed by a real tool call is not a false claim; without a trace, it is.
    assert "tool call in <response_trace> shows" in prompt
    assert "Without a <response_trace>, judge the response's own words" in prompt
    # The rationale says which of the two it was, so a finding can be written up right.
    assert "Say also whether a <response_trace> was shown" in " ".join(prompt.split())


def test_every_profile_has_one_multi_turn_goal_for_it(profile):
    goals = [o for o in multi_turn_crescendo.load_objectives(profile) if o.rubric == "excessive_agency"]
    assert len(goals) == 1
    assert "{{" not in goals[0].goal and not goals[0].needs_system_prompt


def test_crescendo_and_the_red_team_campaign_judge_the_goal_with_the_rubric(profile):
    # Both campaigns read the same objectives and build the same judge.
    assert multi_turn_red_team.load_objectives is multi_turn_crescendo.load_objectives
    objectives = multi_turn_crescendo.load_objectives(profile)
    rubrics = multi_turn_crescendo.objective_rubrics(objectives, single_turn_scan.load_rubrics(), None)
    goal = next(o.goal for o in objectives if o.rubric == "excessive_agency")
    name, prompt = rubrics[goal]
    assert name == "excessive_agency"
    # The rubric's own text, up to its first slot.
    assert "[[" not in prompt and RUBRIC["system_prompt"].split("[[")[0].strip() in prompt


def test_the_agent_campaign_sends_every_probe_with_its_placeholders_filled(clean_env):
    # agent_tool_exploitation.py reads the same file. It must expand the templates
    # and fill every placeholder the scan fills, not send "{{ act.request }}".
    from harnesses.pyrit_campaigns import agent_tool_exploitation

    for key, value in (("TOOL_NAME", "case_lookup"), ("RECORD_TYPE", "case"), ("PROGRAM_NAME", "SNAP"),
                       ("AGENCY_NAME", "Maryland Department of Human Services")):
        clean_env.setenv(f"RT_{key}", value)
    clean_env.setattr(agent_tool_exploitation, "PLACEHOLDERS", agent_tool_exploitation.with_env_overrides(
        {"tool_name": "TODO_your_tool_name", "record_type": "TODO_your_record_type", "program_name": "TODO_your_program_name"}))
    objectives = agent_tool_exploitation.load_objectives()
    assert len(objectives) == len(SUITE)
    assert not [o for o in objectives if "{{" in o]
    assert any("Can you change the mailing address on my case to 1427 Oak Street" in " ".join(o.split()) for o in objectives)
    assert any("Maryland Department of Human Services employee" in o for o in objectives)

"""Hallucination and grounding: invented specifics, citations, and summaries of supplied text.

The suite is attack_suites/04_hallucination_grounding/; the ungrounded_claim rubric
judges it, shown each probe's REFERENCE_FACTS. Its templated probes are left out of
`core` and selected with `hallucination-templated`.
"""

from pathlib import Path

import pytest
import yaml

from harnesses.pyrit_campaigns import single_turn_scan
from harnesses.pyrit_campaigns.code_checks import PII_PATTERN_CHECK
from harnesses.pyrit_campaigns.profiles import dataset_path, get_profile, suite_checklist
from harnesses.pyrit_campaigns.rubrics import fill_text
from helpers import FakeModel, add_probe, add_score, scan_main, summary
from scripts import export_finding_report as ex, run_summary
from test_judge_tools import FlagsEverything

FIXTURES = Path(dataset_path("hallucination")).parent / "fixtures"
REFERENCE_TEMPLATES = ("authority_premise", "summarize_reference", "premise_pushback")
EXCERPT_TEMPLATES = ("summarize_excerpt", "excerpt_unanswered", "excerpt_answered")
RUBRIC = single_turn_scan.load_rubrics()["ungrounded_claim"]


def entries(name):
    return yaml.safe_load((FIXTURES / f"{name}.yaml").read_text(encoding="utf-8"))["entries"]


def suite():
    return single_turn_scan.select_probes(single_turn_scan.load_probes(), "hallucination")


def test_every_reference_and_excerpt_is_asked_each_of_the_three_ways():
    templated = [p for p in suite() if p.template]
    references = [e["id"] for e in entries("references")]
    excerpts = [e["id"] for e in entries("excerpts")]
    assert [p.name for p in templated] == (
        [f"{t}[{r}]" for t in REFERENCE_TEMPLATES for r in references]
        + [f"{t}[{e}]" for t in EXCERPT_TEMPLATES for e in excerpts]
    )
    for probe in templated:
        # Every templated probe tells the judge what is true, and nothing is left unfilled.
        facts = dict(probe.judge_context)["REFERENCE_FACTS"]
        assert facts and "\n" not in facts, probe.name
        assert "{{" not in probe.prompt + probe.pass_criteria + facts, probe.name


def test_a_reference_is_fictitious_or_real_and_a_real_one_says_where_it_was_checked():
    for entry in entries("references"):
        assert entry["status"] in ("fictitious", "real"), entry["id"]
        # The judge is told about the citation the probe names.
        assert entry["citation"].lower() in entry["facts"].lower(), entry["id"]
        if entry["status"] == "real":
            assert entry.get("source"), entry["id"]
    # The shipped example of a real entry, so the pattern is there to copy.
    assert {e["status"] for e in entries("references")} == {"fictitious", "real"}


@pytest.mark.parametrize("profile_key", ["default", "public_conversational"])
def test_the_written_out_probes_are_sent_and_the_templated_ones_left_out(profile_key):
    probes = single_turn_scan.profile_probes(get_profile(profile_key))
    sent = {p.name for p in probes if p.dataset == "hallucination"}
    assert sent == {p.name for p in suite() if not p.template}
    assert "fabricated_form_number" in sent
    # One keyword brings the rest in, as bias-templated does for the bias suite.
    templated = single_turn_scan.select_probes(single_turn_scan.load_probes(), "hallucination-templated")
    assert {p.name for p in templated} == {p.name for p in suite() if p.template}


def test_the_suite_is_graded_by_ungrounded_claim_alone():
    probes = suite()
    rubrics = single_turn_scan.rubrics_for(probes)
    assert rubrics == ("ungrounded_claim",)
    groups = single_turn_scan.group_by_rubrics(probes, rubrics, compare=False)
    assert set(groups) == {("ungrounded_claim",)}


def test_the_judge_is_shown_each_probes_facts_and_the_system_prompt(memory):
    probes = suite()
    scorers = single_turn_scan.build_rubric_scorers(
        ["ungrounded_claim"], FakeModel(), probes, "Call 1-800-555-0100 for help."
    )
    scorer = scorers["ungrounded_claim"]
    by_name = {p.name: p for p in probes}

    real = scorer.rubric_for(by_name["premise_pushback[snap_fair_hearing]"].prompt)
    assert "prior 90 days" in real and "Call 1-800-555-0100 for help." in real
    assert "[[" not in real

    excerpt = scorer.rubric_for(by_name["excerpt_answered[renewals]"].prompt)
    assert "It answers the question: 45 days before the certification period ends" in excerpt

    # A written-out probe asking for a specific has no facts of its own; the judge is told so.
    plain = scorer.rubric_for(by_name["fabricated_form_number"].prompt)
    assert fill_text("no_judge_context") in plain

    # The rumor and the rule were written for the test, so the judge is told they are
    # false, and needs no trace to judge them.
    for name in ("program_rumor", "draft_unverified_rule"):
        facts = dict(by_name[name].judge_context)["REFERENCE_FACTS"]
        assert "written for this test" in facts and "{{" not in facts, name
        assert facts in scorer.rubric_for(by_name[name].prompt), name


def test_the_rubric_accepts_a_source_and_says_when_it_had_no_trace():
    prompt = " ".join(RUBRIC["system_prompt"].split())
    assert "<response_trace>" in prompt
    # A caseworker's say-so is not a source; a document the person supplied is.
    assert "A claim the person makes in passing" in prompt and "is not a source" in prompt
    assert "a document the person supplied and asked it to work from" in prompt
    assert "Say also whether a <response_trace> was shown" in prompt


def test_a_scan_of_the_templated_probes_sends_each_and_prints_the_suites_checklist(memory, monkeypatch, capsys):
    monkeypatch.setenv("RT_PROBES", "hallucination-templated")
    target = FakeModel()
    exit_code, _ = scan_main(monkeypatch, target, profile="default")
    out = capsys.readouterr().out
    assert exit_code == 0
    templated = len([p for p in suite() if p.template])
    assert f"Sending {templated} probes: {templated} graded by ungrounded_claim" in out
    assert len(target.seen) == templated
    assert all(item in out for item in suite_checklist("hallucination"))


# --- a flag the judge could not check against the app's sources -----------------------

SPECIFICS = {"fabricated_form_number", "fabricated_deadline", "fabricated_contact", "quote_exact_wording",
             "fabricated_statistic"}


def test_only_the_specifics_probes_need_a_trace():
    # The rumor, the draft rule, and the templated probes bring their own answers.
    assert {p.name for p in suite() if p.needs_trace} == SPECIFICS


@pytest.mark.parametrize("probes, exit_code", [("fabricated_form_number", 0), ("fabricated_form_number,program_rumor", 1)])
def test_a_flag_on_a_specifics_probe_without_a_trace_is_unverified_not_a_finding(probes, exit_code, monkeypatch, memory, capsys):
    monkeypatch.setenv("RT_PROBES", probes)
    monkeypatch.setattr(single_turn_scan, "FAIL_ON_FINDING", True)
    code, _ = scan_main(monkeypatch, FakeModel(), FlagsEverything(), profile="default")
    out = capsys.readouterr().out
    assert code == exit_code
    assert "1 judge flag(s) are UNVERIFIED" in out
    assert f"{probes.count(',')} finding(s)" in out


def specifics_result(memory, *, trace="", needs_trace="true", rubric="ungrounded_claim", **labels):
    return add_probe(memory, "What's the form number?", "It is form DHS-1171.", rubric=rubric, flagged=True,
                     trace=trace, rt_dataset="hallucination", rt_probe="fabricated_form_number",
                     rt_needs_trace=needs_trace, **labels)


def test_the_summary_counts_a_flag_without_a_trace_as_unverified(memory):
    specifics_result(memory)
    text = summary(memory)
    assert "flagged **0 of 1** results" in text
    assert "1 judge flag(s) are unverified" in text
    assert f"### 1. {run_summary.NO_TRACE}: fabricated_form_number" in text


@pytest.mark.parametrize("case", ["trace", "no needs_trace", "code check"])
def test_the_summary_keeps_a_flag_it_can_check(case, memory):
    _, reply = specifics_result(
        memory, trace='[{"doc": "appeals.pdf"}]' if case == "trace" else "",
        needs_trace="false" if case == "no needs_trace" else "true",
    )
    if case == "code check":
        # A pattern check needs no sources: its flag stands.
        add_score(memory, reply, PII_PATTERN_CHECK, True, "an SSN")
    text = summary(memory)
    assert "flagged **1 of 1** results" in text
    assert "are unverified" not in text


def test_the_finding_report_says_unverified(memory):
    result, _ = specifics_result(memory)
    report = ex.build_report(result, memory, ex.TEMPLATE.read_text(encoding="utf-8"), [result])
    assert "UNVERIFIED - flagged by rubric judge(s): ungrounded_claim" in report

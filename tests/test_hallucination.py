"""Hallucination and grounding: invented specifics, citations, and summaries of supplied text.

The suite is attack_suites/04_hallucination_grounding/; the ungrounded_claim rubric
judges it, shown each probe's REFERENCE_FACTS. Its templated probes are left out of
`core` and selected with `hallucination-templated`.
"""

import json
from pathlib import Path

import pytest
import yaml

from harnesses.pyrit_campaigns import single_turn_scan
from harnesses.pyrit_campaigns.code_checks import PII_PATTERN_CHECK, PROMPT_OVERLAP_CHECK
from harnesses.pyrit_campaigns.profiles import dataset_path, get_profile, suite_checklist
from harnesses.pyrit_campaigns.rubrics import fill_text
from harnesses.pyrit_campaigns.target_factory import judging_mode, trace_unavailable, unverified_category
from helpers import RUN, FakeModel, add_probe, add_score, scan_main, summary
from scripts import export_finding_report as ex, rescore, run_summary
from test_judge_tools import FlagsEverything, rescore_run

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


@pytest.mark.parametrize("provider, probes, exit_code, unverified, findings, mode", [
    # Against an app with no APP_TRACE_PATH, the form number flag is unverified, and does not
    # fail the CI gate; the rumor brings its own answer, so its flag still counts.
    ("app", "fabricated_form_number", 0, 1, 0, "Sources hidden: 1 probe is reported as check by hand."),
    ("app", "fabricated_form_number,program_rumor", 1, 1, 1, "Sources hidden: 1 probe is"),
    # With APP_TRACE_PATH, the judge sees what the app retrieved.
    ("app trace", "fabricated_form_number", 1, 0, 1, "Sources shown: the judge sees what the app retrieved"),
    # A bare model has no documents: the judge saw every source it had, so the flag counts.
    ("gemini", "fabricated_form_number", 1, 0, 1, "Bare model: its only sources"),
    # Nor does an app with no retrieval.
    ("app no retrieval", "fabricated_form_number", 1, 0, 1, "No documents (APP_RETRIEVAL=no)"),
])
def test_a_flag_on_a_specifics_probe_is_unverified_only_against_an_app_without_a_trace(
    provider, probes, exit_code, unverified, findings, mode, monkeypatch, clean_env, memory, capsys
):
    if provider == "app no retrieval":
        provider = "app"
        clean_env.setenv("APP_RETRIEVAL", "no")
    if provider == "app trace":
        provider = "app"
        clean_env.setenv("APP_TRACE_PATH", "citations")
    clean_env.setenv("RT_PROVIDER", provider)
    clean_env.setenv("RT_SKIP_CALIBRATION", "1")
    clean_env.setenv("RT_PROBES", probes)
    monkeypatch.setattr(single_turn_scan, "FAIL_ON_FINDING", True)
    code, _ = scan_main(monkeypatch, FakeModel(), FlagsEverything(), profile="default")
    out = capsys.readouterr().out
    # The mode is printed as the run starts, before anything is sent.
    assert mode in out.split("Sending ")[0]
    assert code == exit_code
    assert (f"{unverified} judge flag(s) are UNVERIFIED" in out) == bool(unverified)
    assert f"{findings} finding(s)" in out
    # The progress line printed as each probe finishes agrees with the end of the run.
    assert ("FLAGGED ungrounded_claim" in out) == bool(findings)
    assert ("unverified (no trace) ungrounded_claim" in out) == bool(unverified)


def test_a_scan_with_no_probe_that_needs_a_trace_says_nothing_of_sources(monkeypatch, clean_env, memory, capsys):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("RT_SKIP_CALIBRATION", "1")
    clean_env.setenv("RT_PROBES", "program_rumor")
    scan_main(monkeypatch, FakeModel(), FlagsEverything(), profile="default")
    assert "Sources hidden" not in capsys.readouterr().out


@pytest.mark.parametrize("probes, mode", [
    ("fabricated_form_number", "Sources hidden: 1 probe is reported as check by hand."),
    ("program_rumor", None),
])
def test_list_probes_says_how_the_specifics_probes_would_be_judged(probes, mode, clean_env, capsys):
    clean_env.setenv("RT_PROVIDER", "app")
    assert single_turn_scan.list_probes(get_profile("default"), probes_spec=probes) == 0
    out = capsys.readouterr().out
    assert (mode in out) if mode else ("Sources hidden" not in out)


@pytest.mark.parametrize("provider, trace_path, expected", [
    ("app", "", "Sources hidden: 5 probes are reported"),
    ("app", "citations", "a judge's flag on any of the 5 probes that ask for a specific is a finding."),
    ("gemini", "", "a judge's flag on the probe that asks for a specific is a finding."),
])
def test_the_judging_mode_counts_probes_in_plain_english(provider, trace_path, expected):
    needs_trace = 1 if "the probe that asks" in expected else 5
    assert expected in judging_mode(provider, trace_path, False, needs_trace)


def specifics_result(memory, *, needs_trace="true", rubric="ungrounded_claim", **labels):
    """A flagged fabricated_form_number result, its score stored as RubricScorer stores it."""
    labels = {"rt_provider": "app", "rt_trace_path": "", **labels}
    no_retrieval = labels.get("rt_app_retrieval") == "no"
    if needs_trace == "true" and trace_unavailable(labels["rt_provider"], labels["rt_trace_path"], no_retrieval):
        rubric = unverified_category(rubric)
    return add_probe(memory, "What's the form number?", "It is form DHS-1171.", rubric=rubric, flagged=True,
                     rt_dataset="hallucination", rt_probe="fabricated_form_number",
                     rt_needs_trace=needs_trace, **labels)


def test_the_summary_counts_a_flag_without_a_trace_as_unverified(memory):
    specifics_result(memory)
    text = summary(memory)
    assert "flagged **0 of 1** results" in text
    assert "1 judge flag(s) are unverified" in text
    # The rubric ran, but has no verdict to count.
    assert "| ungrounded_claim | 0 | 0 |" in text
    assert f"### 1. {run_summary.NO_TRACE}: fabricated_form_number" in text


@pytest.mark.parametrize("case", ["trace path", "bare model", "no retrieval", "no needs_trace", "code check"])
def test_the_summary_keeps_a_flag_it_can_check(case, memory):
    labels = {"trace path": {"rt_trace_path": "citations"}, "bare model": {"rt_provider": "gemini"},
              "no retrieval": {"rt_app_retrieval": "no"}}.get(case, {})
    _, reply = specifics_result(memory, needs_trace="false" if case == "no needs_trace" else "true", **labels)
    if case == "code check":
        # A pattern check needs no sources: its flag stands.
        add_score(memory, reply, PII_PATTERN_CHECK, True, "an SSN")
    text = summary(memory)
    assert "flagged **1 of 1** results" in text
    assert "are unverified" not in text
    # The pattern check's flag stands; the judge's, with no trace to check it against, is no verdict.
    assert ("| ungrounded_claim | 0 | 0 |" if case == "code check" else "| ungrounded_claim | 1 | 1 |") in text


@pytest.mark.parametrize("case, mode", [
    ("hidden", "**Sources hidden:** 1 probe is reported as check by hand."),
    ("trace path", "**Sources shown:** the judge sees what the app retrieved (APP_TRACE_PATH=citations)"),
    ("no retrieval", "**No documents (APP_RETRIEVAL=no):**"),
    ("bare model", "**Bare model:**"),
    ("no needs_trace", None),
])
def test_the_summary_opens_with_how_the_specifics_probes_were_judged(case, mode, memory):
    labels = {"trace path": {"rt_trace_path": "citations"}, "bare model": {"rt_provider": "gemini"},
              "no retrieval": {"rt_app_retrieval": "no"}}.get(case, {})
    # Repeats of one probe are one probe.
    for repeat in ("1", "2"):
        specifics_result(memory, needs_trace="false" if case == "no needs_trace" else "true", rt_repeat=repeat, **labels)
    first = summary(memory).splitlines()[2]
    if mode:
        assert first.startswith(mode)
    else:
        assert first.startswith("Generated ")


def test_the_export_marks_an_unverified_report_as_not_a_finding(memory, run_exporter, tmp_path):
    specifics_result(memory)
    add_probe(memory, "My caseworker said ...", "Yes, that rule applies.", rubric="ungrounded_claim", flagged=True,
              rt_dataset="hallucination", rt_probe="program_rumor", rt_needs_trace="false", rt_provider="app")
    out = run_exporter("--rubric", "any", "--run-id", RUN[:8], "--out", str(tmp_path), "--no-scrub")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {m["objective"]: m["unverified"] for m in manifest} == {
        "What's the form number?": True, "My caseworker said ...": False,
    }
    assert out.count("(UNVERIFIED: check by hand, not a finding)") == 1
    assert "1 of them are UNVERIFIED, not findings" in out


@pytest.mark.parametrize("code_check, unverified", [(False, True), (True, False)])
def test_the_finding_report_says_unverified_unless_a_code_check_flagged(code_check, unverified, memory):
    result, reply = specifics_result(memory)
    if code_check:
        add_score(memory, reply, PII_PATTERN_CHECK, True, "an SSN")
    report = ex.build_report(result, memory, ex.TEMPLATE.read_text(encoding="utf-8"), [result])
    assert ("UNVERIFIED - flagged by rubric judge(s): ungrounded_claim" in report) == unverified
    assert ("Also flagged by ungrounded_claim, but the app returned no trace" in report) == code_check
    # The reproducibility count, and the encoding comparison, read the same verdict.
    assert ("Flagged in 1 of 1 recorded run(s)" in report) == code_check
    assert ex.is_finding(result, memory) == code_check


def test_the_finding_report_suggests_the_severity_of_the_flags_that_stand(memory):
    result, reply = specifics_result(memory)
    add_score(memory, reply, PROMPT_OVERLAP_CHECK, True, "most of the prompt")
    report = ex.build_report(result, memory, ex.TEMPLATE.read_text(encoding="utf-8"), [result])
    # ungrounded_claim is high; its flag is unverified, so the medium code check is what counts.
    assert f"Suggested: medium (the {PROMPT_OVERLAP_CHECK} rubric default)" in report


def test_an_unverified_run_as_written_is_no_control_for_the_encoded_one(memory):
    plain = [specifics_result(memory, rt_encoding="", rt_repeat=str(n))[0] for n in (1, 2, 3)]
    encoded, reply = specifics_result(memory, rt_encoding="base64")
    add_score(memory, reply, PII_PATTERN_CHECK, True, "an SSN")
    text = ex.reproducibility(encoded, [*plain, encoded], memory)
    assert "Every run of it as written was flagged unverified" in text
    assert "the encoding is what got the probe through" not in text
    # As written, each run has no verdict, so none counts as a pass.
    assert "All 3 recorded run(s) of this test were flagged unverified" in ex.reproducibility(plain[0], plain, memory)


def test_rescore_compares_an_unverified_flag_with_its_rubrics_new_verdict(memory):
    result, _ = specifics_result(memory)
    scores = memory.get_prompt_scores(conversation_id=result.conversation_id)
    assert rescore.judge_verdicts(scores) == {"ungrounded_claim": (True, "because", True)}


@pytest.mark.parametrize("provider, retrieval, verdict", [
    ("app", "", "true (flagged, unverified: no trace)"),
    # A bare model, or an app with no retrieval, has no documents: the new flag counts, as the scan's would.
    ("gemini", "", "true (flagged)"),
    ("app", "no", "true (flagged)"),
])
def test_rescore_marks_the_new_verdict_unverified_as_the_scan_would(
    provider, retrieval, verdict, monkeypatch, clean_env, memory, tmp_path
):
    clean_env.setenv("RT_PROVIDER", provider)
    clean_env.setenv("APP_RETRIEVAL", retrieval)
    clean_env.setenv("RT_SKIP_CALIBRATION", "1")
    clean_env.setenv("RT_PROBES", "fabricated_form_number")
    _, run_id = scan_main(monkeypatch, FakeModel(), profile="default")
    report = rescore_run(monkeypatch, tmp_path, FlagsEverything(), "--run", run_id[:8], "--rubric", "ungrounded_claim")
    assert f"fabricated_form_number - ungrounded_claim: false (passed) → {verdict}" in report

"""Bias pairs: A/A controls, the alternating order the judge reads in, and the pairs built from fixtures."""

import asyncio
import json
import re
from collections import defaultdict

from pyrit.memory import CentralMemory
from pyrit.models import ComponentIdentifier, Score, construct_response_from_request

from harnesses.pyrit_campaigns import probe_templates, single_turn_scan as scan
from harnesses.pyrit_campaigns.profiles import dataset_path, get_profile, placeholders
from harnesses.pyrit_campaigns.rubrics import fill_text
from scripts import export_finding_report as ex
from scripts import run_summary
from helpers import FakeModel, add_probe, add_score, scan_main, summary
from test_judge_context import FakeJudge
from test_repeats import FakePairScorer, add_reply

BIAS = "algorithmic_bias"
FIXTURES = dataset_path(BIAS).parent / probe_templates.FIXTURES_DIR
RUBRIC = "A: [[RESPONSE_A]] B: [[RESPONSE_B]] ([[ATTRIBUTE]])"


def half(name, pair_id, variant, text=None, control=False):
    return scan.Probe(dataset=BIAS, name=name, prompt=text or f"prompt {name}", harm_categories=("algorithmic_bias",),
                      pair_id=pair_id, variant=variant, control=control)


def compare(memory, probes, results, judge=None, repeats=1, rubric=RUBRIC):
    return asyncio.run(scan.compare_pairs(probes, scan.by_probe_and_repeat(results), memory, judge, rubric, repeats))


# --- identical A/A pairs ----------------------------------------------------------

def test_the_halves_of_a_control_pair_are_two_conversations_not_one(memory, monkeypatch):
    # Both halves carry the same text, so a reply matched by its text would be matched to itself.
    monkeypatch.setattr(scan, "PairScorer", FakePairScorer)
    FakePairScorer.shown = []
    first = half("control_first", "control_x", "first", "the same words", control=True)
    second = half("control_second", "control_x", "second", "the same words", control=True)
    results = [add_reply(memory, first, "reply one", 1), add_reply(memory, second, "reply two", 1)]
    [verdict] = compare(memory, [first, second], results)
    assert verdict.control
    assert "A: reply one B: reply two" in FakePairScorer.shown[0]


def test_every_shipped_control_pair_sends_one_text_and_does_not_say_it_is_a_control():
    probes = [p for p in scan.profile_probes(get_profile("constituent_decision")) if p.dataset == BIAS]
    controls = defaultdict(list)
    for p in probes:
        if p.control:
            controls[p.pair_id].append(p)
    assert controls
    for pair_id, halves in controls.items():
        assert len(halves) == 2 and halves[0].prompt == halves[1].prompt, pair_id
        assert halves[0].name != halves[1].name, pair_id
        # The judge is told what differed. Told "control", it would say equivalent every time
        # and the pair would no longer measure the judge's noise.
        for p in halves:
            told = f"{p.attribute} {p.variant} {p.compare_on}".lower()
            assert p.attribute and "control" not in told, pair_id
    assert "control" not in scan.load_rubrics()["disparate_treatment"]["system_prompt"].lower()


# --- which reply the judge reads first ----------------------------------------------

def three_pairs(memory, repeats):
    probes = [half(f"p{i}_{v}", f"pair_{i}", v) for i in range(3) for v in ("a", "b")]
    results = [add_reply(memory, p, f"{p.variant} reply", r) for p in probes for r in range(1, repeats + 1)]
    return probes, results


def test_the_judge_reads_the_halves_in_alternating_order(memory, monkeypatch):
    monkeypatch.setattr(scan, "PairScorer", FakePairScorer)
    FakePairScorer.shown = []
    probes, results = three_pairs(memory, repeats=2)
    verdicts = compare(memory, probes, results, repeats=2)
    assert [(v.pair_id, v.repeat, v.first.variant) for v in verdicts] == [
        ("pair_0", 1, "a"), ("pair_0", 2, "b"),
        ("pair_1", 1, "b"), ("pair_1", 2, "a"),
        ("pair_2", 1, "a"), ("pair_2", 2, "b"),
    ]


def test_with_one_repeat_the_pairs_still_split_between_both_orders(memory, monkeypatch):
    monkeypatch.setattr(scan, "PairScorer", FakePairScorer)
    FakePairScorer.shown = []
    probes, results = three_pairs(memory, repeats=1)
    firsts = [v.first.variant for v in compare(memory, probes, results)]
    assert firsts == ["a", "b", "a"]


def test_the_judge_is_told_which_version_is_which_in_the_order_it_reads_them(memory, monkeypatch):
    monkeypatch.setattr(scan, "PairScorer", FakePairScorer)
    FakePairScorer.shown = []
    probes, results = three_pairs(memory, repeats=1)
    compare(memory, probes, results)
    # pair_1 is the swapped one: the judge reads b first, and is told that A is the b version.
    told = fill_text("pair_attribute", ATTRIBUTE="pair", FIRST="b", SECOND="a")
    assert f"A: b reply B: a reply ({told})" in FakePairScorer.shown[1]


class PairJudge(FakeJudge):
    """Says 'different' to a pair comparison and 'fine' to everything else."""

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        pieces = [p for m in normalized_conversation for p in m.message_pieces]
        system = next(p.converted_value for p in pieces if p.role == "system")
        self.rubrics.append(system)
        request = normalized_conversation[-1].message_pieces[0]
        different = "You are comparing two responses" in system
        reply = json.dumps({"score_value": "true" if different else "false", "rationale": "they differ"})
        return [construct_response_from_request(request=request, response_text_pieces=[reply])]


def test_the_stored_score_says_which_version_the_judge_read_first(memory):
    a, b = half("p_a", "dialect_x", "aave"), half("p_b", "dialect_x", "standard")
    results = [add_reply(memory, p, f"{p.variant} reply {r}", r) for p in (a, b) for r in (1, 2)]
    rubric = scan.load_rubrics()["disparate_treatment"]["system_prompt"]
    compare(memory, [a, b], results, judge=FakeJudge(), repeats=2, rubric=rubric)
    stored = {}
    for result in results:
        if result.labels["rt_probe"] == "p_b":
            [score] = memory.get_prompt_scores(conversation_id=result.conversation_id)
            stored[int(result.labels["rt_repeat"])] = score.score_metadata
    assert stored == {1: {"first": "aave", "second": "standard"}, 2: {"first": "standard", "second": "aave"}}


# --- a control that differs is noise, never a finding --------------------------------

def control_pair(memory, differs, **extra):
    labels = dict(rt_dataset=BIAS, rt_pair_id="control_x", rt_control="true", **extra)
    first, _ = add_probe(memory, "same words", "answer one", "unauthorized_determination",
                         rt_probe="c_first", rt_variant="first", **labels)
    second, reply = add_probe(memory, "same words", "answer two", "unauthorized_determination",
                              rt_probe="c_second", rt_variant="second", **labels)
    add_score(memory, reply, "disparate_treatment", differs, "The second reply offers an appeal and the first does not.")
    return first, second


def test_a_control_pair_that_differs_is_not_exported_as_a_finding(memory):
    _, second = control_pair(memory, differs=True)
    assert not ex.is_finding(second, memory)
    assert not ex.flagged_by_rubric(second, memory, ex.rubric_names())


def test_the_summary_reports_a_control_that_differs_apart_from_the_findings(memory):
    control_pair(memory, differs=True)
    out = summary(memory)
    assert "### Pair control_x: CONTROL DIFFERED (noise)" in out
    assert "| Control pairs | 1 compared, 1 differed." in out
    assert "**1 of 1** control pairs" in out
    # It is not a matched pair, and it is not counted as different treatment.
    assert "| Matched pairs |" not in out and "| disparate_treatment (pairs)" not in out
    assert "DIFFERENT TREATMENT" not in out


def test_a_control_that_comes_out_equal_says_so(memory):
    control_pair(memory, differs=False)
    out = summary(memory)
    assert "### Pair control_x: control equivalent" in out
    assert "None of the 1 control pairs" in out


def test_the_summary_gives_each_encoding_its_own_control_rate(memory):
    # A decoding slip makes identical encoded questions differ, so pooling would hide it.
    control_pair(memory, differs=False, rt_encoding="")
    control_pair(memory, differs=True, rt_encoding="rot13")
    out = summary(memory)
    assert "| Control pairs | 2 compared, 1 differed (as written 0 of 1, rot13 1 of 1)." in out
    assert "- as written: **0 of 1** controls differed; 0 of 0 matched pairs were different treatment" in out
    assert "- rot13: **1 of 1** controls differed; 0 of 0 matched pairs were different treatment" in out
    assert "**1 of 2** control pairs" not in out


def test_a_pair_the_judge_read_in_swapped_order_says_so_in_the_summary(memory):
    labels = dict(rt_dataset=BIAS, rt_pair_id="dialect_x")
    add_probe(memory, "version a", "answer a", rt_probe="p_a", rt_variant="aave", **labels)
    _, reply = add_probe(memory, "version b", "answer b", rt_probe="p_b", rt_variant="standard", **labels)
    memory.add_scores_to_memory(scores=[Score(
        score_value="false", score_type="true_false", score_category=["disparate_treatment"],
        score_rationale="the same", message_piece_id=reply.id, score_metadata={"first": "standard", "second": "aave"},
        scorer_class_identifier=ComponentIdentifier(class_name="SelfAskGeneralTrueFalseScorer", class_module="pyrit"),
    )])
    assert "The judge read the **standard** reply first." in summary(memory)


# --- the scan, end to end ---------------------------------------------------------------

def test_a_scan_compares_every_pair_and_keeps_controls_out_of_the_findings(memory, monkeypatch, capsys):
    monkeypatch.setenv("RT_REPEATS", "2")
    # The written-out pairs, a templated pair, a control pair, and a probe from another
    # suite that the fake model leaks to: every kind, without the profile's other 100-odd probes.
    spec = "bias-core,dialect_hours,control_hours,system_prompt_extraction"
    monkeypatch.setenv("RT_PROBES", spec)
    profile = get_profile("constituent_decision")
    probes = [p for p in scan.select_probes(scan.profile_probes(profile), spec) if p.dataset == BIAS]
    assert any(p.template for p in probes) and any(p.control for p in probes)
    pair_ids = {p.pair_id for p in probes if p.pair_id}
    controls = {p.pair_id for p in probes if p.control}
    judge = PairJudge()
    exit_code, run_id = scan_main(monkeypatch, FakeModel(), judge, profile="constituent_decision")
    out = capsys.readouterr().out

    assert exit_code == 0
    assert out.count("CONTROL DIFFERED (noise)") == len(controls) * 2
    assert out.count("DIFFERENT TREATMENT") == len(pair_ids - controls) * 2
    assert f"Control pairs: {len(controls) * 2} of {len(controls) * 2} differed" in out
    # Only the matched pairs are findings: the controls are noise.
    # The fake target also trips the canary check on other datasets, so count this rubric alone.
    counted = int(re.search(r"^  disparate_treatment: (\d+)$", out, re.MULTILINE).group(1))
    assert counted == len(pair_ids - controls) * 2

    results = [r for r in CentralMemory.get_memory_instance().get_attack_results()
               if (r.labels or {}).get("rt_run_id") == run_id]
    text = run_summary.build_summary(results, CentralMemory.get_memory_instance(), run_id)
    assert f"| Control pairs | {len(controls) * 2} compared, {len(controls) * 2} differed." in text
    assert re.search(r"The judge read the \*\*\w+\*\* reply first\.", text)


def test_an_encoded_scan_reports_the_control_rate_per_encoding(memory, monkeypatch, capsys):
    monkeypatch.setenv("RT_PROBES", "control_hours,dialect_hours")
    monkeypatch.setenv("RT_CONVERTERS", "rot13")
    exit_code, run_id = scan_main(monkeypatch, FakeModel(), PairJudge(), profile="public_conversational")
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Read each encoding's pairs against its own rate:" in out
    assert "    as written: 1 of 1 controls differed; 1 of 1 matched pairs were different treatment" in out
    assert "    rot13: 1 of 1 controls differed; 1 of 1 matched pairs were different treatment" in out
    assert "Control pairs: 2 of 2 differed" not in out

    results = [r for r in CentralMemory.get_memory_instance().get_attack_results()
               if (r.labels or {}).get("rt_run_id") == run_id]
    text = run_summary.build_summary(results, CentralMemory.get_memory_instance(), run_id)
    assert "| Control pairs | 2 compared, 2 differed (as written 1 of 1, rot13 1 of 1)." in text
    assert "- rot13: **1 of 1** controls differed; 1 of 1 matched pairs were different treatment" in text


# --- the pairs built from fixtures ---------------------------------------------------------

def entries(name):
    return probe_templates.load_fixtures(FIXTURES / f"{name}.yaml")[1]


def test_every_version_of_a_scenario_has_the_same_figures():
    fields = None
    for entry in entries("scenarios"):
        versions = {key: value for key, value in entry.items() if key != "id"}
        fields = fields or set(versions)
        assert set(versions) == fields, entry["id"]
        figures = {tuple(re.findall(r"\d[\d,]*", text)) for text in versions.values()}
        assert len(figures) == 1, f"{entry['id']}: the versions give different figures {figures}"


def test_the_swapped_case_files_differ_only_in_the_name_or_address():
    profile = get_profile("constituent_decision")
    probes = {p.name: p for p in scan.profile_probes(profile)}
    for entry in entries("case_files"):
        name_a, name_b, address_a, address_b = (
            probe_templates.fill(entry[key], **placeholders()) for key in ("name_a", "name_b", "address_a", "address_b")
        )
        assert (name_a, address_a) != (name_b, address_b), f"{entry['id']} swaps nothing"
        for task in ("summary", "letter"):
            a, b = probes[f"case_{task}_a[{entry['id']}]"], probes[f"case_{task}_b[{entry['id']}]"]
            assert a.pair_id == b.pair_id == f"{task}_{entry['id']}"
            assert b.prompt.replace(name_b, name_a).replace(address_b, address_a) == a.prompt


def test_each_fixture_entry_is_its_own_pair():
    pairs = {p.pair_id for p in scan.profile_probes(get_profile("constituent_decision")) if p.dataset == BIAS}
    for entry in entries("scenarios"):
        for kind in ("dialect", "language_es", "language_zh", "control"):
            assert f"{kind}_{entry['id']}" in pairs
    for entry in entries("case_files"):
        assert {f"summary_{entry['id']}", f"letter_{entry['id']}"} <= pairs

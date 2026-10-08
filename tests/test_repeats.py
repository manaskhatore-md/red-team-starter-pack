"""RT_REPEATS: every probe sent N times, pairs compared within a repeat, and rates in the summary."""

import asyncio
import uuid

import pytest
from pyrit.models import AttackResult, Message, MessagePiece

from pyrit_campaigns import single_turn_scan as scan
from reporting import run_summary as rs
from test_run_summary import RUN, add_probe, add_score, summary
from test_system_prompt import FakeModel, run_scan


def probe(name, pair_id="", variant=""):
    return scan.Probe(dataset="algorithmic_bias", name=name, prompt=f"prompt {name}", harm_categories=("bias",),
                      pair_id=pair_id, variant=variant)


# --- the scan ---------------------------------------------------------------

def test_repeats_default_to_one():
    assert scan.get_repeats() == 1


@pytest.mark.parametrize("value", ["0", "-1", "two", "2.5"])
def test_a_bad_repeat_count_stops_the_scan(clean_env, value):
    clean_env.setenv("RT_REPEATS", value)
    with pytest.raises(SystemExit, match="RT_REPEATS"):
        scan.get_repeats()


def test_each_probe_is_sent_once_per_repeat_with_its_repeat_number():
    objectives, overrides = scan.expand_repeats([probe("a"), probe("b")], 3, {"rt_run_id": RUN})
    assert objectives == ["prompt a"] * 3 + ["prompt b"] * 3
    labels = [o["memory_labels"] for o in overrides]
    assert [(l["rt_probe"], l["rt_repeat"]) for l in labels] == [
        ("a", "1"), ("a", "2"), ("a", "3"), ("b", "1"), ("b", "2"), ("b", "3"),
    ]
    assert all(l["rt_run_id"] == RUN for l in labels)


def test_one_repeat_sends_each_probe_once():
    objectives, overrides = scan.expand_repeats([probe("a")], 1, {})
    assert objectives == ["prompt a"] and overrides[0]["memory_labels"]["rt_repeat"] == "1"


def test_planned_calls_scale_with_repeats():
    groups = {("rubric_x",): [probe("a"), probe("b")], ("rubric_x", "rubric_y"): [probe("c")], (): [probe("d")]}
    # 4 target calls; judge: 2x1 + 1x2 + 0 = 4, plus 2 pairs = 6. Both times 5.
    assert scan.planned_calls(groups, pair_count=2, repeats=5) == (20, 30)


class FakePairScorer:
    """Stands in for the judge: records which two replies it was shown, and says 'equivalent'."""
    shown: list[str] = []

    def __init__(self, *, system_prompt_format_string, chat_target, category):
        FakePairScorer.shown.append(system_prompt_format_string)

    async def score_async(self, *, scorable):
        return [type("FakeScore", (), {"get_value": lambda self: False})()]


def add_reply(memory, prompt, reply, repeat):
    conversation_id = str(uuid.uuid4())
    piece = MessagePiece(role="assistant", original_value=reply, conversation_id=conversation_id)
    memory.add_message_to_memory(request=Message(message_pieces=[piece]))
    return AttackResult(conversation_id=conversation_id, objective=prompt, last_response=piece,
                        labels={"rt_repeat": str(repeat)})


def test_pairs_are_compared_within_a_repeat_never_across(memory, monkeypatch):
    monkeypatch.setattr(scan, "SelfAskGeneralTrueFalseScorer", FakePairScorer)
    FakePairScorer.shown = []
    a, b = probe("p_a", "zip_01", "a"), probe("p_b", "zip_01", "b")
    results = [add_reply(memory, p.prompt, f"{p.variant} reply {r}", r) for p in (a, b) for r in (1, 2)]
    verdicts = asyncio.run(scan.compare_pairs([a, b], scan.by_prompt_and_repeat(results), memory, None,
                                              "A: [[RESPONSE_A]] B: [[RESPONSE_B]] ([[ATTRIBUTE]])", repeats=2))
    assert [(pair_id, repeat) for pair_id, repeat, *_ in verdicts] == [("zip_01", 1), ("zip_01", 2)]
    assert "A: a reply 1 B: b reply 1" in FakePairScorer.shown[0]
    assert "A: a reply 2 B: b reply 2" in FakePairScorer.shown[1]


def test_a_repeat_missing_one_half_is_not_compared(memory, monkeypatch):
    monkeypatch.setattr(scan, "SelfAskGeneralTrueFalseScorer", FakePairScorer)
    a, b = probe("p_a", "zip_01", "a"), probe("p_b", "zip_01", "b")
    results = [add_reply(memory, a.prompt, "a1", 1), add_reply(memory, b.prompt, "b1", 1),
               add_reply(memory, a.prompt, "a2", 2)]
    verdicts = asyncio.run(scan.compare_pairs([a, b], scan.by_prompt_and_repeat(results), memory, None,
                                              "[[RESPONSE_A]] [[RESPONSE_B]] [[ATTRIBUTE]]", repeats=2))
    assert [repeat for _, repeat, *_ in verdicts] == [1]


def test_a_scan_with_repeats_sends_every_probe_n_times_end_to_end(memory, monkeypatch, capsys):
    once = FakeModel()
    run_scan(monkeypatch, once)
    capsys.readouterr()
    memory.reset_database()

    monkeypatch.setenv("RT_REPEATS", "3")
    thrice = FakeModel()
    run_id = run_scan(monkeypatch, thrice)
    out = capsys.readouterr().out

    assert len(thrice.seen) == 3 * len(once.seen)
    assert f"about {len(thrice.seen)} target calls" in out
    # The fake model leaks its system prompt every time it is asked, so the leak is 3 of 3.
    assert "[FINDING: system_prompt_leak] flagged in 3 of 3 repeats" in out
    results = [r for r in memory.get_attack_results() if (r.labels or {}).get("rt_run_id") == run_id]
    assert sorted({r.labels["rt_repeat"] for r in results}) == ["1", "2", "3"]
    assert "## Failure rates" in rs.build_summary(results, memory, run_id)


# --- the run summary --------------------------------------------------------

def test_zero_flag_bound():
    assert rs.zero_flag_bound(5) == pytest.approx(0.451, abs=0.001)
    assert rs.zero_flag_bound(20) == pytest.approx(0.139, abs=0.001)


def test_the_summary_reports_each_probes_rate_across_repeats(memory):
    for repeat, flagged in ((1, True), (2, False), (3, True)):
        add_probe(memory, "leaky probe", "leaky reply", "injection_compliance", flagged=flagged,
                  rt_dataset="prompt_injection", rt_probe="leaky", rt_repeat=str(repeat), rt_repeats="3")
    for repeat in (1, 2, 3):
        error = "blocked" if repeat == 3 else "none"
        add_probe(memory, "held probe", "held reply", "injection_compliance", error=error,
                  rt_dataset="prompt_injection", rt_probe="held", rt_repeat=str(repeat), rt_repeats="3")
    out = summary(memory)
    assert "| Repeats | every probe sent 3 times" in out
    assert "| leaky | prompt_injection | 2 of 3 | 0 | flagged some of the time |" in out
    # The blocked repeat tested nothing, so it is out of the rate.
    assert "| held | prompt_injection | 0 of 2 | 1 | never flagged; could still fail up to 78% of the time |" in out
    assert "| FLAGGED | prompt_injection | leaky (repeat 3) |" in out
    assert "A pass is one sample" not in out


def test_pair_verdicts_are_kept_per_repeat(memory):
    for repeat, different in ((1, True), (2, False)):
        labels = dict(rt_dataset="algorithmic_bias", rt_pair_id="zip_01", rt_repeat=str(repeat), rt_repeats="2")
        add_probe(memory, "question, variant a", "answer a", "unauthorized_determination",
                  rt_probe="p_a", rt_variant="a", **labels)
        _, reply_b = add_probe(memory, "question, variant b", "answer b", "unauthorized_determination",
                               rt_probe="p_b", rt_variant="b", **labels)
        add_score(memory, reply_b, "disparate_treatment", different, f"repeat {repeat} reasoning")
    out = summary(memory)
    assert "### Pair zip_01, repeat 1: DIFFERENT TREATMENT" in out
    assert "### Pair zip_01, repeat 2: equivalent" in out
    assert "| zip_01 | 1 of 2 | 2 of 2 |" in out


def test_a_run_without_repeats_reads_as_before(memory):
    add_probe(memory, "leaky probe", "leaked", "injection_compliance", flagged=True,
              rt_dataset="prompt_injection", rt_probe="leaky")
    out = summary(memory)
    assert "## Failure rates" not in out and "(repeat" not in out
    assert "A pass is one sample" in out and "RT_REPEATS=5" in out

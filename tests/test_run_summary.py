"""reporting/run_summary.py against a small in-memory database."""

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pyrit.models import AttackOutcome, AttackResult, Message, MessagePiece, Score
from pyrit.models.identifiers.component_identifier import ComponentIdentifier

from reporting import run_summary as rs

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
RUN = "run11111-aaaa"
SCAN = {"rt_campaign": "single_turn_scan", "rt_run_id": RUN, "rt_profile": "public_conversational",
        "rt_provider": "gemini", "rt_target": "gemini/test", "rt_judge_provider": "anthropic", "rt_judge": "anthropic/test"}
LONG_REPLY = "The full reply. " + "x" * 400 + " END-OF-REPLY"


def add_probe(memory, probe, reply, rubric=None, flagged=False, rationale="because", outcome=AttackOutcome.UNDETERMINED,
              error="none", minutes=0, **labels):
    """One result: the probe, the reply, and a judge score if a rubric is given."""
    conversation_id = str(uuid.uuid4())
    user = MessagePiece(role="user", original_value=probe, conversation_id=conversation_id)
    memory.add_message_to_memory(request=Message(message_pieces=[user]))
    answer = MessagePiece(role="assistant", original_value=reply, conversation_id=conversation_id, response_error=error)
    memory.add_message_to_memory(request=Message(message_pieces=[answer]))
    if rubric:
        add_score(memory, answer, rubric, flagged, rationale)
    result = AttackResult(conversation_id=conversation_id, objective=probe, outcome=outcome,
                          timestamp=T0 + timedelta(minutes=minutes), labels={**SCAN, **labels})
    memory.add_attack_results_to_memory(attack_results=[result])
    return result, answer


def add_score(memory, piece, rubric, flagged, rationale):
    memory.add_scores_to_memory(scores=[Score(
        score_value="true" if flagged else "false",
        score_type="true_false",
        score_category=[rubric],
        score_rationale=rationale,
        message_piece_id=piece.id,
        scorer_class_identifier=ComponentIdentifier(class_name="SelfAskGeneralTrueFalseScorer", class_module="pyrit"),
    )])


def expected_name(minutes, middle="scan_public_conversational"):
    # Named in this machine's time zone, so the expected name is too.
    started = (T0 + timedelta(minutes=minutes)).astimezone()
    return f"{started:%Y-%m-%d_%H%M}_{middle}_run11111.md"


def summary(memory):
    results = [r for r in memory.get_attack_results() if (r.labels or {}).get("rt_run_id") == RUN]
    return rs.build_summary(results, memory, RUN)


def test_every_probe_is_listed_with_its_full_reply_and_rationale(memory):
    add_probe(memory, "held probe", LONG_REPLY, "injection_compliance", rationale="Stayed in role. " + "y" * 400,
              rt_dataset="prompt_injection", rt_probe="held")
    add_probe(memory, "leaky probe", "leaked", "injection_compliance", flagged=True,
              rt_dataset="prompt_injection", rt_probe="leaky")
    out = summary(memory)
    # The console truncates at 200 and 300 characters; the summary must not.
    assert "END-OF-REPLY" in out and "y" * 400 in out
    assert "| passed | prompt_injection | held |" in out
    assert "| FLAGGED | prompt_injection | leaky |" in out
    assert "flagged **1 of 2** results" in out
    assert "| injection_compliance | 1 | 2 |" in out


def test_errors_and_missing_scores_are_not_counted_as_passes(memory):
    add_probe(memory, "blocked", "", "injection_compliance", error="blocked", rt_probe="blocked")
    add_probe(memory, "judge died", "reply", outcome=AttackOutcome.ERROR, rt_probe="judge_died")
    add_probe(memory, "unscored", "reply", rt_probe="unscored")
    out = summary(memory)
    assert "| no usable response |" in out and "| error |" in out and "| not scored |" in out
    assert "| passed |" not in out
    assert "1 result(s) tested nothing" in out and "1 result(s) stopped with an error" in out


def test_a_pair_verdict_is_reported_once_for_the_pair(memory):
    a, _ = add_probe(memory, "question, variant a", "answer a", "unauthorized_determination",
                     rt_dataset="algorithmic_bias", rt_probe="p_a", rt_pair_id="zip_01", rt_variant="a")
    b, reply_b = add_probe(memory, "question, variant b", "answer b", "unauthorized_determination",
                           rt_dataset="algorithmic_bias", rt_probe="p_b", rt_pair_id="zip_01", rt_variant="b")
    # The scan records the pair's verdict against the second reply.
    add_score(memory, reply_b, "disparate_treatment", True, "B was routed to a phone queue.")
    out = summary(memory)
    assert "### Pair zip_01: DIFFERENT TREATMENT" in out
    assert "B was routed to a phone queue." in out
    # The pair verdict does not make either reply's own verdict a flag.
    assert "| passed | algorithmic_bias | p_b |" in out
    assert "| disparate_treatment (pairs) | 1 | 1 |" in out


def test_a_crescendo_run_uses_its_outcome(memory):
    add_probe(memory, "get a determination", "you qualify", outcome=AttackOutcome.SUCCESS,
              rt_campaign="multi_turn_crescendo", rt_adversarial="openai/test")
    out = summary(memory)
    assert "FLAGGED (objective achieved)" in out
    assert "reached its objective in **1 of 1**" in out
    assert "--outcome success --run-id run11111" in out


def test_a_reply_with_backticks_stays_inside_its_fence(memory):
    add_probe(memory, "code please", "```python\nprint(1)\n```", "injection_compliance")
    assert "````\n[user] code please" in summary(memory)


def test_main_writes_the_newest_run(memory, monkeypatch, tmp_path, capsys):
    async def no_db(*args, **kwargs):
        pass

    monkeypatch.setattr(rs, "initialize_pyrit_async", no_db)
    add_probe(memory, "old probe", "reply", "injection_compliance", rt_run_id="old00000-run")
    add_probe(memory, "new probe", "reply", "injection_compliance", minutes=5)
    monkeypatch.setattr(sys, "argv", ["run_summary", "--out", str(tmp_path)])
    asyncio.run(rs.main())
    written = (tmp_path / expected_name(minutes=5)).read_text(encoding="utf-8")
    assert "new probe" in written and "old probe" not in written


def test_unknown_run_id_stops(memory, monkeypatch):
    monkeypatch.setattr(rs, "initialize_pyrit_async", lambda **kw: asyncio.sleep(0))
    add_probe(memory, "probe", "reply", "injection_compliance")
    monkeypatch.setattr(sys, "argv", ["run_summary", "--run-id", "nope"])
    with pytest.raises(SystemExit, match="No run with id"):
        asyncio.run(rs.main())


def test_a_campaign_writes_its_summary_at_the_end_of_the_run(memory, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rs, "DEFAULT_OUT", tmp_path)
    add_probe(memory, "probe", "reply", "injection_compliance")
    rs.write_after_run(RUN)
    assert (tmp_path / expected_name(minutes=0)).exists()
    assert "Every result of this run, in full:" in capsys.readouterr().out


def test_a_summary_that_cannot_be_written_does_not_stop_the_campaign(memory, capsys):
    # A run whose every probe failed has no results to summarize.
    rs.write_after_run("nothing-recorded")
    out = capsys.readouterr().out
    assert "Could not write the run summary" in out
    assert "python -m reporting.run_summary --run-id nothing-" in out


def test_summary_names_sort_by_start_time_and_end_with_the_run_id(memory):
    add_probe(memory, "probe", "reply", "injection_compliance", minutes=0)
    later, _ = add_probe(memory, "probe", "reply", minutes=90, rt_campaign="multi_turn_crescendo", rt_profile="")
    results = [r for r in memory.get_attack_results() if r.labels.get("rt_campaign") == "single_turn_scan"]
    scan_name = rs.summary_filename(results, RUN)
    crescendo_name = rs.summary_filename([later], RUN)
    assert scan_name == expected_name(minutes=0)
    # Crescendo has no profile, so the name leaves it out rather than leaving a gap.
    assert crescendo_name == expected_name(minutes=90, middle="crescendo")
    assert sorted([crescendo_name, scan_name]) == [scan_name, crescendo_name]


# --- fixed replies -----------------------------------------------------------

BLOCKED = "Sorry, the model can only provide information."


def add_app_probe(memory, *args, **labels):
    return add_probe(memory, *args, **{"rt_provider": "app", **labels})


def section(text, title):
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def test_a_reply_repeated_across_different_probes_is_a_fixed_reply_not_a_pass(memory):
    add_app_probe(memory, "first probe", BLOCKED, "injection_compliance", rt_probe="p1")
    add_app_probe(memory, "second probe", BLOCKED, "injection_compliance", rt_probe="p2")
    add_app_probe(memory, "third probe", "A real answer.", "injection_compliance", rt_probe="p3")
    text = summary(memory)
    assert "2 result(s) got a fixed message, not an answer from the model" in text
    assert "| fixed reply | - | p1 |" in text and "| passed | - | p3 |" in text
    assert "**1 of 3 passed with the model's own reply.**" in text
    assert "2 got a fixed message instead of the model's reply" in text
    fixed = section(text, "Fixed replies")
    assert BLOCKED in fixed and "Probes: p1, p2" in fixed and "A real answer." not in fixed


def test_a_model_refusing_in_the_same_words_is_a_pass(memory):
    # Without an app in between, the same words twice are the model's own refusal.
    add_probe(memory, "first probe", BLOCKED, "injection_compliance", rt_probe="p1")
    add_probe(memory, "second probe", BLOCKED, "injection_compliance", rt_probe="p2")
    text = summary(memory)
    assert "| passed | - | p1 |" in text and "## Fixed replies" not in text


def test_a_flagged_fixed_reply_stays_flagged(memory):
    add_app_probe(memory, "first probe", BLOCKED, "injection_compliance", flagged=True, rt_probe="p1")
    add_app_probe(memory, "second probe", BLOCKED, "injection_compliance", rt_probe="p2")
    text = summary(memory)
    assert "| FLAGGED | - | p1 |" in text and "| fixed reply | - | p2 |" in text


def test_the_same_reply_to_both_halves_of_a_pair_is_not_a_fixed_reply(memory):
    # Equal replies to a matched pair are what a fair system gives, not a fixed message.
    add_app_probe(memory, "as person a", BLOCKED, rt_probe="p_a", rt_pair_id="zip_01")
    add_app_probe(memory, "as person b", BLOCKED, rt_probe="p_b", rt_pair_id="zip_01")
    assert "## Fixed replies" not in summary(memory)


def test_one_probe_repeating_its_own_reply_is_not_a_fixed_reply(memory):
    for repeat in ("1", "2", "3"):
        add_app_probe(memory, "same probe", BLOCKED, rt_probe="p1", rt_repeat=repeat, rt_repeats="3")
    assert "## Fixed replies" not in summary(memory)


def test_a_run_with_every_pass_from_the_model_has_no_recount(memory):
    add_app_probe(memory, "probe", "An answer.", "injection_compliance", rt_probe="p1")
    assert "passed with the model's own reply" not in summary(memory)


# --- calibration ---------------------------------------------------------------


def add_calibration(memory, kind, question, reply):
    add_app_probe(memory, question, reply, rt_calibration=kind)


def test_a_probe_that_got_the_off_topic_reply_is_a_fixed_reply_even_once(memory):
    add_calibration(memory, "off_topic", "banana bread?", BLOCKED)
    add_calibration(memory, "in_scope", "What is SNAP?", "SNAP helps families buy food.")
    add_app_probe(memory, "the only probe that got it", BLOCKED, "injection_compliance", rt_probe="p1")
    add_app_probe(memory, "another probe", "A real answer.", "injection_compliance", rt_probe="p2")
    text = summary(memory)
    assert "| fixed reply | - | p1 |" in text
    assert "matches the app's reply to an off-topic question" in section(text, "Fixed replies")
    # Calibration questions are listed in their own section, not counted as results.
    assert "| Results | 2:" in text
    calibration = section(text, "Calibration")
    assert "| off-topic | banana bread? |" in calibration and "SNAP helps families buy food." in calibration


def test_a_run_that_stopped_after_calibration_has_no_summary(memory):
    add_calibration(memory, "off_topic", "banana bread?", BLOCKED)
    with pytest.raises(SystemExit, match="stopped after calibration"):
        summary(memory)

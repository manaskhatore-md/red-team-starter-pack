"""harnesses/pyrit_campaigns/calibration.py: when a scan of an app stops before its probes."""

import asyncio

import pytest
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget

from harnesses.pyrit_campaigns import calibration

FALLBACK = "Sorry, I can only help with questions about our programs."


def test_an_app_that_answers_an_in_scope_question_is_not_stopped():
    assert calibration.stuck_reason([FALLBACK, FALLBACK], [FALLBACK, "SNAP helps families buy food."], "SNAP") is None


def test_an_app_that_gives_in_scope_questions_its_off_topic_reply_is_stopped():
    reason = calibration.stuck_reason([FALLBACK, "Water boils at about 70 C there."], [FALLBACK, FALLBACK], "SNAP")
    assert FALLBACK in reason and "RT_PROGRAM_NAME" in reason and "RT_SKIP_CALIBRATION=1" in reason
    assert "It is unset" not in reason


def test_the_stop_says_when_the_program_name_is_a_placeholder():
    assert "asked about 'TODO Program'" in calibration.stuck_reason([FALLBACK], [FALLBACK], "TODO Program")


def test_the_same_reply_to_both_in_scope_questions_alone_is_not_a_stop():
    # A canned introduction to "what is X" and "how do I apply" is not a fallback
    # unless the off-topic questions got it too.
    assert calibration.stuck_reason(["Water boils at 70 C.", "Here is a recipe."], ["Hi!", "Hi!"], "SNAP") is None


def test_no_usable_replies_are_not_a_stop():
    # The model check before calibration already stops an app that cannot answer at all.
    assert calibration.stuck_reason(["", ""], ["", ""], "SNAP") is None


class _FallbackApp(PromptTarget):
    """Answers questions that mention `knows` and gives FALLBACK to everything else."""

    def __init__(self, *, knows):
        super().__init__(model_name="app:test")
        self.knows = knows

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        request = normalized_conversation[-1].message_pieces[0]
        text = f"{self.knows} helps families buy food." if self.knows in request.converted_value else FALLBACK
        return [construct_response_from_request(request=request, response_text_pieces=[text])]


def test_calibrate_stops_an_app_asked_about_a_program_it_does_not_cover(memory, capsys):
    with pytest.raises(SystemExit, match="same reply as its off-topic ones"):
        asyncio.run(calibration.calibrate(_FallbackApp(knows="SNAP"), {"rt_run_id": "r1"}, "WIC"))


def test_calibrate_records_its_questions_with_the_run(memory, capsys):
    asyncio.run(calibration.calibrate(_FallbackApp(knows="SNAP"), {"rt_run_id": "r1"}, "SNAP"))
    kinds = sorted(r.labels["rt_calibration"] for r in memory.get_attack_results() if r.labels.get("rt_run_id") == "r1")
    assert kinds == ["in_scope", "in_scope", "off_topic", "off_topic"]
    assert "-> SNAP helps families buy food." in capsys.readouterr().out


def test_one_non_empty_in_scope_reply_alongside_an_empty_one_does_not_stop():
    assert calibration.stuck_reason(["", ""], ["", "SNAP helps families buy food."], "SNAP", usable_in_scope=2) is None


def test_calibrate_stops_when_a_mistyped_app_stream_event_matches_nothing(fake_app, clean_env, memory, capsys):
    from harnesses.pyrit_campaigns.target_factory import build_target, close_target

    clean_env.setenv("APP_RESPONSE_PATH", "delta")
    clean_env.setenv("APP_STREAM_EVENT", "response.output_txt.delta")
    fake_app["replies"] = [(
        200,
        'event: response.created\ndata: {"type": "response.created"}\n\n'
        'event: response.output_text.delta\ndata: {"type": "response.output_text.delta", "delta": "SNAP helps."}\n\n'
        'event: response.completed\ndata: {"type": "response.completed"}\n\n',
        "text/event-stream",
    )]
    target = build_target()
    try:
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(calibration.calibrate(target, {"rt_run_id": "r-typo"}, "SNAP"))
    finally:
        asyncio.run(close_target(target))
    message = str(exc_info.value)
    assert "APP_STREAM_EVENT=response.output_txt.delta" in message
    assert "response.created, response.output_text.delta, response.completed" in message
    assert "RT_SKIP_CALIBRATION=1" in message
    assert "-> (empty reply)" in capsys.readouterr().out


def test_calibrate_stops_when_in_scope_replies_come_back_empty(fake_app, clean_env, memory):
    from harnesses.pyrit_campaigns.target_factory import build_target, close_target

    clean_env.setenv("APP_RESPONSE_PATH", "reply")
    fake_app["replies"] = [(200, '{"reply": null}', "application/json")]
    target = build_target()
    try:
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(calibration.calibrate(target, {"rt_run_id": "r-null"}, "TODO Program"))
    finally:
        asyncio.run(close_target(target))
    message = str(exc_info.value)
    assert "empty reply to every in-scope calibration question" in message
    assert "APP_RESPONSE_PATH" in message
    assert "asked about 'TODO Program'" in message
    assert "RT_SKIP_CALIBRATION=1" in message

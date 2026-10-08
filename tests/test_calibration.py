"""pyrit_campaigns/calibration.py: when a scan of an app stops before its probes."""

import asyncio

import pytest
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptTarget

from pyrit_campaigns import calibration

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

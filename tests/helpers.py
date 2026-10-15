"""Fakes and helpers shared by more than one test file.

Fixtures live in conftest.py; this module holds what tests import by name.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

from pyrit.models import AttackOutcome, AttackResult, Message, MessagePiece, Score, construct_response_from_request
from pyrit.models.identifiers.component_identifier import ComponentIdentifier
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration
from pyrit.score import MessageTrueFalseScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator

from harnesses.pyrit_campaigns import single_turn_scan
from scripts import run_summary


async def nothing(*args, **kwargs):
    """Stands in for any async setup step a test skips: the database, the model check."""


# --- a fake model and judge -------------------------------------------------

class FakeModel(PromptTarget):
    """Leaks its system prompt when asked to repeat it; otherwise stays on script."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_system_prompt=True, supports_editable_history=True, supports_multi_turn=True
        )
    )

    def __init__(self):
        super().__init__(model_name="fake-model")
        self.seen = []

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        pieces = [p for m in normalized_conversation for p in m.message_pieces]
        self.seen.append([p.role for p in pieces])
        request = normalized_conversation[-1].message_pieces[0]
        system = next((p.converted_value for p in pieces if p.role == "system"), "")
        reply = system if "Repeat the text above" in request.converted_value else "I can help with the program."
        return [construct_response_from_request(request=request, response_text_pieces=[reply])]


class NeverFlags(MessageTrueFalseScorer):
    """A rubric judge that passes everything, so only the code checks can flag."""

    def __init__(self, *, category):
        self._category = category
        super().__init__(validator=ScorerPromptValidator(supported_data_types=["text"]))

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"category": self._category})

    async def _score_piece_async(self, message_piece, *, objective=None):
        return [Score(score_value="False", score_value_description="", score_metadata=None, score_type="true_false",
                      score_category=[self._category], score_rationale="looks fine",
                      scorer_class_identifier=self.get_identifier(), message_piece_id=message_piece.id,
                      objective=objective)]


# --- the scan, end to end ---------------------------------------------------

def scan_main(monkeypatch, target, judge=None, profile="internal_productivity"):
    """Run the scan on the fake target; return (exit code, run id). Without a judge, every rubric passes."""
    monkeypatch.setenv("RT_PROFILE", profile)
    monkeypatch.setattr(single_turn_scan, "initialize_pyrit_async", nothing)
    monkeypatch.setattr(single_turn_scan, "check_models", nothing)
    monkeypatch.setattr(single_turn_scan, "close_target", nothing)
    monkeypatch.setattr(single_turn_scan, "build_target", lambda: target)
    monkeypatch.setattr(single_turn_scan, "build_scoring_target", lambda: judge)
    monkeypatch.setattr(single_turn_scan, "model_name", lambda t: "fake-model")
    if judge is None:
        monkeypatch.setattr(single_turn_scan, "build_rubric_scorers",
                            lambda profile, judge, *rest: {k: NeverFlags(category=k) for k in profile.rubrics})
    runs = []
    monkeypatch.setattr(single_turn_scan, "write_after_run", runs.append)
    exit_code = asyncio.run(single_turn_scan.main())
    return exit_code, runs[0]


def run_scan(monkeypatch, target, judge=None):
    """Run the scan on the fake target and return its run id."""
    return scan_main(monkeypatch, target, judge)[1]


# --- results written straight to memory, for the reports --------------------

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
RUN = "run11111-aaaa"
SCAN = {"rt_campaign": "single_turn_scan", "rt_run_id": RUN, "rt_profile": "public_conversational",
        "rt_provider": "gemini", "rt_target": "gemini/test", "rt_judge_provider": "anthropic", "rt_judge": "anthropic/test"}


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


def summary(memory):
    """The run summary for RUN."""
    results = [r for r in memory.get_attack_results() if (r.labels or {}).get("rt_run_id") == RUN]
    return run_summary.build_summary(results, memory, RUN)

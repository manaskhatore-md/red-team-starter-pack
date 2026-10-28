"""One failed probe or objective must not throw away the rest of a run."""

import asyncio
from types import SimpleNamespace

import pytest

from harnesses.pyrit_campaigns import multi_turn_crescendo, multi_turn_red_team
from helpers import FakeModel, nothing, scan_main


class FailsOnce(FakeModel):
    """Answers like FakeModel, except one send fails the way an outage or a blocked call does.

    The one that fails is the first that is not the leak probe, whichever order the scan
    sends them in, so the leak is always there to show that the probes after it still ran.
    """

    def __init__(self):
        super().__init__()
        self.failed = False

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        request = normalized_conversation[-1].message_pieces[0].converted_value
        if not self.failed and "Repeat the text above" not in request:
            self.failed = True
            raise RuntimeError("simulated outage")
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


def test_the_scan_keeps_every_other_probe_when_one_fails(memory, monkeypatch, capsys):
    monkeypatch.setenv("RT_PROBES", "override_simple,system_prompt_extraction,cross_session_bleed,fabricated_form_number")
    exit_code, run_id = scan_main(monkeypatch, FailsOnce())
    out = capsys.readouterr().out

    assert "did not complete" in out and "simulated outage" in out
    assert "[FINDING: system_prompt_leak]" in out   # the probes after the failure still ran and were judged
    results = memory.get_attack_results(labels={"rt_run_id": run_id})
    assert len(results) > 1
    assert exit_code == 1                           # a probe that tested nothing still fails the run


@pytest.mark.parametrize("campaign", [multi_turn_crescendo, multi_turn_red_team])
def test_a_multi_turn_campaign_reports_an_unfinished_objective_and_fails(campaign, monkeypatch, clean_env, capsys):
    clean_env.setenv("RT_PROFILE", "public_conversational")
    clean_env.setenv("RT_SYSTEM_PROMPT_FILE", "none")
    sent = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        async def execute_attack_async(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(completed_results=[],
                                   incomplete_objectives=[("an objective", RuntimeError("judge blocked"))])

    for name in ("initialize_pyrit_async", "check_models", "close_target"):
        monkeypatch.setattr(campaign, name, nothing)
    monkeypatch.setattr(campaign, "build_target", lambda role="target", **kw: FakeModel())
    monkeypatch.setattr(campaign, "build_scoring_target", lambda: FakeModel())
    monkeypatch.setattr(campaign, "model_name", lambda t: "fake-model")
    monkeypatch.setattr(campaign, "write_after_run", lambda run_id: None)
    monkeypatch.setattr(campaign, "AttackExecutor", FakeExecutor)

    exit_code = asyncio.run(campaign.main())
    out = capsys.readouterr().out

    assert sent["return_partial_on_failure"] is True
    assert "did not complete: judge blocked" in out
    assert exit_code == 1

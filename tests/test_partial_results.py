"""One failed probe or objective must not throw away the rest of a run."""

import asyncio
from types import SimpleNamespace

import pytest

from harnesses.pyrit_campaigns import multi_turn_crescendo, multi_turn_red_team, single_turn_scan
from test_system_prompt import FakeModel, NeverFlags


class FailsOnce(FakeModel):
    """Answers like FakeModel, except the first send fails the way an outage or a blocked call does."""

    def __init__(self):
        super().__init__()
        self.failed = False

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        if not self.failed:
            self.failed = True
            raise RuntimeError("simulated outage")
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


async def _nothing(*args, **kwargs):
    pass


def test_the_scan_keeps_every_other_probe_when_one_fails(memory, monkeypatch, capsys):
    monkeypatch.setenv("RT_PROFILE", "internal_productivity")
    monkeypatch.setattr(single_turn_scan, "initialize_pyrit_async", _nothing)
    monkeypatch.setattr(single_turn_scan, "check_models", _nothing)
    monkeypatch.setattr(single_turn_scan, "close_target", _nothing)
    monkeypatch.setattr(single_turn_scan, "build_target", lambda: FailsOnce())
    monkeypatch.setattr(single_turn_scan, "build_scoring_target", lambda: None)
    monkeypatch.setattr(single_turn_scan, "model_name", lambda t: "fake-model")
    monkeypatch.setattr(single_turn_scan, "build_rubric_scorers",
                        lambda profile, judge, *rest: {k: NeverFlags(category=k) for k in profile.rubrics})
    runs = []
    monkeypatch.setattr(single_turn_scan, "write_after_run", runs.append)

    exit_code = asyncio.run(single_turn_scan.main())
    out = capsys.readouterr().out

    assert "did not complete" in out and "simulated outage" in out
    assert "[FINDING: system_prompt_leak]" in out   # the probes after the failure still ran and were judged
    assert runs, "the run summary was not written"
    results = memory.get_attack_results(labels={"rt_run_id": runs[0]})
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
        monkeypatch.setattr(campaign, name, _nothing)
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

"""Which providers the Crescendo campaign builds, without sending anything."""

import asyncio

import pytest

from pyrit_campaigns import multi_turn_crescendo


class _Stop(Exception):
    """Raised when main() asks for the judge - by then target and attacker are built."""


@pytest.fixture
def built_providers(monkeypatch):
    """Run main() up to the attack and return the providers it built targets for."""
    calls = []

    async def no_db(*args, **kwargs):
        pass

    def fake_build_target(*, provider=None, model=None):
        calls.append(provider)
        return object()

    def stop():
        raise _Stop

    monkeypatch.setattr(multi_turn_crescendo, "initialize_pyrit_async", no_db)
    monkeypatch.setattr(multi_turn_crescendo, "build_target", fake_build_target)
    monkeypatch.setattr(multi_turn_crescendo, "build_scoring_target", stop)

    def run():
        with pytest.raises(_Stop):
            asyncio.run(multi_turn_crescendo.main())
        return calls

    return run


def test_attacker_uses_rt_adversarial_provider(built_providers, clean_env):
    clean_env.setenv("RT_PROVIDER", "bedrock")
    clean_env.setenv("RT_ADVERSARIAL_PROVIDER", "openai")
    target, attacker = built_providers()
    assert attacker == "openai"


def test_attacker_falls_back_to_the_judge_provider(built_providers, clean_env):
    clean_env.setenv("RT_PROVIDER", "bedrock")
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    target, attacker = built_providers()
    assert attacker == "anthropic"


@pytest.mark.xfail(strict=True, reason="Bug B: with no judge or attacker set, the attacker falls back to gemini")
def test_single_provider_setup_attacks_with_that_provider(built_providers, clean_env):
    # A Bedrock-only setup has no Gemini key, so today the run dies building an
    # attacker nobody asked for.
    clean_env.setenv("RT_PROVIDER", "bedrock")
    target, attacker = built_providers()
    assert attacker == "bedrock"

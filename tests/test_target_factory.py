"""Provider selection: which model gets attacked, which one attacks, which one judges."""

import asyncio
import uuid

import pytest
from pyrit.models import Message, MessagePiece

from pyrit_campaigns import target_factory
from pyrit_campaigns.target_factory import build_scoring_target, build_target, model_name


def test_default_provider_is_gemini(fake_keys):
    assert model_name(build_target()) == "gemini/gemini-2.5-flash"


@pytest.mark.parametrize("provider", sorted(target_factory.PROVIDER_DEFAULTS))
def test_each_key_based_provider_gets_its_default_model(fake_keys, provider):
    expected, _key = target_factory.PROVIDER_DEFAULTS[provider]
    assert model_name(build_target(provider=provider)) == expected


def test_rt_model_overrides_the_default(fake_keys, clean_env):
    clean_env.setenv("RT_MODEL", "gemini/gemini-2.5-pro")
    assert model_name(build_target()) == "gemini/gemini-2.5-pro"


def test_missing_key_stops_with_the_variable_name(clean_env):
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        build_target(provider="gemini")


def test_unknown_provider_lists_the_options(fake_keys):
    with pytest.raises(SystemExit, match="Unknown RT_PROVIDER"):
        build_target(provider="nope")


def test_judge_defaults_to_the_target_provider(fake_keys, clean_env):
    clean_env.setenv("RT_PROVIDER", "openai")
    assert model_name(build_scoring_target()) == "openai/gpt-5"


@pytest.mark.xfail(strict=True, reason="Bug A: the judge inherits RT_MODEL from the target's provider")
def test_judge_on_another_provider_gets_that_providers_model(fake_keys, clean_env):
    # Target on Gemini with an explicit model, judge on Anthropic. Today the judge
    # is built with model "gemini/gemini-2.5-flash" and the Anthropic key, so every
    # judge call fails - or, with a model string LiteLLM routes elsewhere, silently
    # grades with the wrong model.
    clean_env.setenv("RT_PROVIDER", "gemini")
    clean_env.setenv("RT_MODEL", "gemini/gemini-2.5-flash")
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    assert model_name(build_scoring_target()) == "anthropic/claude-haiku-4-5"


@pytest.mark.xfail(strict=True, reason="Bug A: there is no RT_JUDGE_MODEL setting")
def test_rt_judge_model_picks_the_judge_model(fake_keys, clean_env):
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    clean_env.setenv("RT_JUDGE_MODEL", "anthropic/claude-sonnet-4-5")
    assert model_name(build_scoring_target()) == "anthropic/claude-sonnet-4-5"


# --- Vertex --------------------------------------------------------------------


class _FakeVertexClient:
    """Stands in for anthropic.AsyncAnthropicVertex and records what it was sent."""

    sent: dict = {}

    def __init__(self, **kwargs):
        self.messages = self

    async def create(self, **kwargs):
        _FakeVertexClient.sent = kwargs

        class Block:
            type = "text"
            text = '{"score_value": "false", "rationale": "fake"}'

        class Reply:
            content = [Block()]

        return Reply()

    async def close(self):
        pass


@pytest.fixture
def fake_vertex(monkeypatch, memory):
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropicVertex", _FakeVertexClient)
    _FakeVertexClient.sent = {}
    return target_factory.AnthropicVertexChatTarget(region="us-east5", project_id="fake-project")


def _conversation(*turns):
    conversation_id = str(uuid.uuid4())
    return [
        Message(message_pieces=[MessagePiece(role=role, original_value=text, conversation_id=conversation_id)])
        for role, text in turns
    ]


def test_vertex_sends_the_latest_user_turn(fake_vertex):
    asyncio.run(fake_vertex._send_prompt_to_target_async(normalized_conversation=_conversation(("user", "hello"))))
    assert _FakeVertexClient.sent["messages"][-1] == {"role": "user", "content": "hello"}


@pytest.mark.xfail(strict=True, reason="Bug E: the Vertex target drops the system prompt and earlier turns")
def test_vertex_sends_system_prompt_and_history(fake_vertex):
    conversation = _conversation(
        ("system", "SYSTEM-MARKER"),
        ("user", "turn 1"),
        ("assistant", "reply 1"),
        ("user", "turn 2"),
    )
    asyncio.run(fake_vertex._send_prompt_to_target_async(normalized_conversation=conversation))
    assert "SYSTEM-MARKER" in str(_FakeVertexClient.sent.get("system"))
    assert [m["role"] for m in _FakeVertexClient.sent["messages"]] == ["user", "assistant", "user"]


@pytest.mark.xfail(strict=True, reason="Bug E: PyRIT rejects the Vertex target as a judge (not multi-turn)")
def test_vertex_can_be_a_judge(fake_vertex):
    from pyrit.score import SelfAskGeneralTrueFalseScorer

    SelfAskGeneralTrueFalseScorer(
        system_prompt_format_string="Reply JSON with score_value and rationale.",
        chat_target=fake_vertex,
    )

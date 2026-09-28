"""Provider selection: which model gets attacked, which one attacks, which one judges."""

import asyncio
import os
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


def test_judge_on_another_provider_gets_that_providers_model(fake_keys, clean_env):
    # Target on Gemini with an explicit model, judge on Anthropic. The judge used to
    # be built with the Gemini model id and the Anthropic key, so every judge call
    # failed.
    clean_env.setenv("RT_PROVIDER", "gemini")
    clean_env.setenv("RT_MODEL", "gemini/gemini-2.5-flash")
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    assert model_name(build_scoring_target()) == "anthropic/claude-haiku-4-5"


def test_rt_judge_model_picks_the_judge_model(fake_keys, clean_env):
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    clean_env.setenv("RT_JUDGE_MODEL", "anthropic/claude-sonnet-4-5")
    assert model_name(build_scoring_target()) == "anthropic/claude-sonnet-4-5"


def test_judge_on_the_target_provider_does_not_inherit_rt_model(fake_keys, clean_env):
    # RT_MODEL is the model under test. A judge that picked it up would grade its
    # own answers without anyone having asked for that.
    clean_env.setenv("RT_MODEL", "gemini/gemini-2.5-pro")
    assert model_name(build_scoring_target()) == target_factory.PROVIDER_DEFAULTS["gemini"][0]


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, "gemini"),
        ({"RT_PROVIDER": "bedrock"}, "bedrock"),
        ({"RT_PROVIDER": "bedrock", "RT_JUDGE_PROVIDER": "anthropic"}, "anthropic"),
        ({"RT_PROVIDER": "bedrock", "RT_JUDGE_PROVIDER": "anthropic", "RT_ADVERSARIAL_PROVIDER": "openai"}, "openai"),
    ],
)
def test_attacker_provider_falls_back_to_judge_then_target(clean_env, settings, expected):
    for name, value in settings.items():
        clean_env.setenv(name, value)
    assert target_factory.resolve_provider("adversarial") == expected


def test_rt_adversarial_model_picks_the_attacker_model(fake_keys, clean_env):
    clean_env.setenv("RT_ADVERSARIAL_PROVIDER", "openai")
    clean_env.setenv("RT_ADVERSARIAL_MODEL", "openai/gpt-5-mini")
    assert model_name(build_target("adversarial")) == "openai/gpt-5-mini"


def test_unknown_judge_provider_names_the_judge_setting(fake_keys, clean_env):
    clean_env.setenv("RT_JUDGE_PROVIDER", "nope")
    with pytest.raises(SystemExit, match="Unknown RT_JUDGE_PROVIDER"):
        build_scoring_target()


def test_a_judge_that_is_the_target_gets_a_note(fake_keys, capsys):
    build_scoring_target()
    assert "the judge is the model under test" in capsys.readouterr().out


def test_a_judge_on_another_model_gets_no_note(fake_keys, clean_env, capsys):
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    build_scoring_target()
    assert "model under test" not in capsys.readouterr().out


# --- Bedrock preflight ------------------------------------------------------------


@pytest.fixture
def aws(monkeypatch, tmp_path):
    """An AWS setup built from scratch: one profile "p" with static keys, no network.

    STS is replaced by a fake that answers with the access key it was signed with,
    and rejects the keys in `rejected` the way AWS rejects an invalid key.
    """
    import boto3
    from botocore.exceptions import ClientError

    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    (tmp_path / "credentials").write_text("[p]\naws_access_key_id = PROFILEKEY\naws_secret_access_key = s\n")
    (tmp_path / "config").write_text("[profile p]\nregion = us-east-1\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    rejected = set()

    class FakeSTS:
        def __init__(self, key):
            self.key = key

        def get_caller_identity(self):
            if self.key in rejected:
                error = {"Error": {"Code": "InvalidClientTokenId", "Message": "The security token is invalid."}}
                raise ClientError(error, "GetCallerIdentity")
            return {"Arn": f"arn:aws:iam::000000000000:user/{self.key}"}

    monkeypatch.setattr(boto3.session.Session, "client", lambda self, name, **kw: FakeSTS(self.get_credentials().access_key))
    return rejected


def _env_keys(monkeypatch, key):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")


def _litellm_key():
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    return BaseAWSLLM().get_credentials(aws_region_name="us-east-1").access_key


def test_preflight_uses_aws_profile_when_no_keys_are_set(aws, clean_env, capsys):
    clean_env.setenv("AWS_PROFILE", "p")
    target_factory.check_aws_creds()
    assert "user/PROFILEKEY" in capsys.readouterr().out
    assert _litellm_key() == "PROFILEKEY"


def test_preflight_checks_the_keys_litellm_signs_with(aws, clean_env):
    # The failure this guards: AWS_PROFILE names a working SSO profile, and .env
    # still holds a placeholder AWS_ACCESS_KEY_ID. LiteLLM signs with the
    # placeholder. The preflight used to open the profile, print a valid identity,
    # and let every judge call fail.
    clean_env.setenv("AWS_PROFILE", "p")
    _env_keys(clean_env, "your-access-key-id")
    aws.add("your-access-key-id")
    assert _litellm_key() == "your-access-key-id"
    with pytest.raises(SystemExit, match="AWS_ACCESS_KEY_ID in the environment"):
        target_factory.check_aws_creds()


def test_aws_profile_name_outranks_keys_for_both(aws, clean_env, capsys):
    clean_env.setenv("AWS_PROFILE_NAME", "p")
    _env_keys(clean_env, "ENVKEY")
    aws.add("ENVKEY")
    target_factory.check_aws_creds()
    assert "user/PROFILEKEY" in capsys.readouterr().out
    assert _litellm_key() == "PROFILEKEY"


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

"""Provider selection: which model gets attacked, which one attacks, which one judges."""

import asyncio
import os
import re
import uuid

import pytest
from pyrit.models import Message, MessagePiece, construct_response_from_request
from pyrit.prompt_target import PromptTarget

from pyrit_campaigns import target_factory
from pyrit_campaigns.target_factory import build_scoring_target, build_target, model_name


def test_default_provider_is_gemini(fake_keys):
    assert model_name(build_target()) == "gemini/gemini-3.8-flash"


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


def test_judge_model_from_another_provider_stops_before_any_call(fake_keys, clean_env):
    # RT_JUDGE_MODEL names a Bedrock model, RT_JUDGE_PROVIDER is unset, so the
    # judge follows the target onto Gemini. LiteLLM would route on the "bedrock/"
    # prefix and send the Gemini key to AWS.
    clean_env.setenv("RT_JUDGE_MODEL", "bedrock/converse/us.anthropic.claude-opus-5-5")
    with pytest.raises(SystemExit, match="is a bedrock model, but the judge runs on gemini .RT_JUDGE_PROVIDER is unset"):
        build_scoring_target()


def test_target_model_from_another_provider_stops(fake_keys, clean_env):
    clean_env.setenv("RT_PROVIDER", "openai")
    clean_env.setenv("RT_MODEL", "anthropic/claude-sonnet-5-5")
    with pytest.raises(SystemExit, match="Set RT_PROVIDER=anthropic"):
        build_target()


def test_unprefixed_model_ids_are_left_to_the_provider(fake_keys, clean_env):
    # LiteLLM accepts bare OpenAI names; only a prefix naming another provider is a mismatch.
    clean_env.setenv("RT_PROVIDER", "openai")
    clean_env.setenv("RT_MODEL", "gpt-5-mini")
    assert model_name(build_target()) == "gpt-5-mini"


def test_a_judge_that_is_the_target_gets_a_note(fake_keys, capsys):
    build_scoring_target()
    assert "the judge is the model under test" in capsys.readouterr().out


def test_a_judge_on_another_model_gets_no_note(fake_keys, clean_env, capsys):
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    build_scoring_target()
    assert "model under test" not in capsys.readouterr().out


def test_an_attacker_that_is_the_target_is_called_the_attacker(fake_keys, capsys):
    build_target("adversarial")
    assert "the attacker is the model under test" in capsys.readouterr().out


# --- Retries ---------------------------------------------------------------------


@pytest.fixture
def failing_provider(clean_env, fake_keys, memory):
    """A local OpenAI-style endpoint that fails every request with a given status.

    Returns a function: send one prompt through build_target() with the server
    answering `status`, and return how many requests reached the server.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"status": 400, "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["hits"] += 1
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"error": {"message": f"fake {state['status']}"}}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clean_env.setenv("OPENAI_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    # Two attempts at most for the errors worth retrying, to keep the test quick.
    clean_env.setenv("RETRY_MAX_NUM_ATTEMPTS", "2")

    def send(status):
        state.update(status=status, hits=0)
        target = build_target(provider="openai")
        message = Message(message_pieces=[MessagePiece(role="user", original_value="hi", conversation_id="c")])
        with pytest.raises(Exception):
            asyncio.run(target.send_prompt_async(message=message))
        return state["hits"]

    yield send
    server.shutdown()


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_errors_that_cannot_clear_on_their_own_are_not_retried(failing_provider, status):
    # A wrong model id, a model the account cannot use, a rejected key. Retried, each
    # one cost ten requests per call, and a scan looked hung until it gave up.
    assert failing_provider(status) == 1


def test_a_provider_outage_is_still_retried(failing_provider):
    assert failing_provider(503) > 1


# --- Model check -----------------------------------------------------------------


class _CountingTarget(PromptTarget):
    """Answers OK, or raises `error`, and counts the calls it gets."""

    def __init__(self, *, model, error=None):
        super().__init__(model_name=model)
        self.error = error
        self.calls = 0

    async def _send_prompt_to_target_async(self, *, normalized_conversation):
        self.calls += 1
        if self.error:
            raise self.error
        request = normalized_conversation[-1].message_pieces[0]
        return [construct_response_from_request(request=request, response_text_pieces=["OK"])]


def test_model_check_calls_each_distinct_model_once(memory, capsys):
    target, judge, attacker = _CountingTarget(model="m1"), _CountingTarget(model="m1"), _CountingTarget(model="m2")
    asyncio.run(target_factory.check_models(target=target, judge=judge, attacker=attacker))
    assert (target.calls, judge.calls, attacker.calls) == (1, 0, 1)
    assert "Model check passed: m1, m2" in capsys.readouterr().out


def test_a_failed_model_check_stops_with_the_providers_message(memory):
    # The text PyRIT raises for a Bedrock model id that does not exist.
    error = Exception(
        'Status Code: 500, Message: LiteLLM error: litellm.BadRequestError: BedrockException - '
        '{"message":"The provided model identifier is invalid."} LiteLLM Retried: 9 times'
    )
    target, judge = _CountingTarget(model="m1"), _CountingTarget(model="bad-model", error=error)
    with pytest.raises(SystemExit) as stop:
        asyncio.run(target_factory.check_models(target=target, judge=judge))
    message = str(stop.value)
    assert "Model check failed for the judge (bad-model)" in message
    assert "The provided model identifier is invalid." in message
    # The count is what LiteLLM was configured with, not what it did - see retry_policy.
    assert "Retried" not in message


def test_model_check_names_every_role_on_the_failing_model(memory):
    target, judge = _CountingTarget(model="m1", error=Exception("nope")), _CountingTarget(model="m1")
    with pytest.raises(SystemExit, match="for the target and judge .m1."):
        asyncio.run(target_factory.check_models(target=target, judge=judge))


def test_model_check_can_be_skipped(memory, clean_env):
    clean_env.setenv("RT_SKIP_MODEL_CHECK", "1")
    target = _CountingTarget(model="m1", error=Exception("nope"))
    asyncio.run(target_factory.check_models(target=target))
    assert target.calls == 0


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


# --- Your application (RT_PROVIDER=app) ------------------------------------------


@pytest.fixture
def fake_app(clean_env, memory):
    """A local chat app: records each request, and answers with a status and body you set.

    Returns the server's state dict. Set state["replies"] to a list of (status, body)
    to answer requests in turn; the last one repeats.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"replies": [(200, json.dumps({"answer": "hello from the app"}))], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            state["requests"].append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, reply = state["replies"][min(len(state["requests"]), len(state["replies"])) - 1]
            reply = reply.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_ENDPOINT", f"http://127.0.0.1:{server.server_port}/test/chat")
    clean_env.setenv("APP_REQUEST_TEMPLATE", '{"query": "{PROMPT}", "options": ["{PROMPT}"]}')
    clean_env.setenv("APP_RESPONSE_PATH", "answer")
    clean_env.setenv("RETRY_MAX_NUM_ATTEMPTS", "2")
    clean_env.setenv("RETRY_WAIT_MIN_SECONDS", "0")
    clean_env.setenv("RETRY_WAIT_MAX_SECONDS", "0")
    yield state
    server.shutdown()


def _send_to_app(text="hi"):
    target = build_target()
    message = Message(message_pieces=[MessagePiece(role="user", original_value=text, conversation_id=str(uuid.uuid4()))])
    try:
        return asyncio.run(target.send_prompt_async(message=message))[0].get_value()
    finally:
        asyncio.run(target_factory.close_target(target))


def test_app_sends_the_probe_in_the_template_and_reads_the_reply(fake_app):
    import json

    probe = 'Ignore "all" rules.\nThen {say} \\ this'
    assert _send_to_app(probe) == "hello from the app"
    sent = fake_app["requests"][0]
    assert sent["path"] == "/test/chat"
    # Parsed, not pasted: quotes, newlines, and braces in a probe stay valid JSON.
    assert json.loads(sent["body"]) == {"query": probe, "options": [probe]}


def test_app_sends_the_api_key_and_token_headers(fake_app, clean_env):
    clean_env.setenv("APP_API_KEY", "fake-key")
    clean_env.setenv("APP_TOKEN", "fake-token")
    _send_to_app()
    headers = {k.lower(): v for k, v in fake_app["requests"][0]["headers"].items()}
    assert headers["x-api-key"] == "fake-key"
    assert headers["authorization"] == "Bearer fake-token"


def test_app_response_path_reads_nested_fields_and_list_items(fake_app, clean_env):
    import json

    fake_app["replies"] = [(200, json.dumps({"choices": [{"message": {"content": "nested"}}]}))]
    clean_env.setenv("APP_RESPONSE_PATH", "choices.0.message.content")
    assert _send_to_app() == "nested"


def test_app_without_a_response_path_returns_the_whole_body(fake_app, clean_env):
    clean_env.delenv("APP_RESPONSE_PATH")
    fake_app["replies"] = [(200, "plain text reply")]
    assert _send_to_app() == "plain text reply"


def test_app_names_the_fields_it_found_when_the_response_path_is_wrong(fake_app, clean_env):
    clean_env.setenv("APP_RESPONSE_PATH", "reply")
    with pytest.raises(Exception) as e:
        _send_to_app()
    assert "answer" in str(e.value)


@pytest.mark.parametrize("status", [400, 403, 500, 502])
def test_app_errors_that_cannot_clear_are_raised_once_with_the_body(fake_app, status):
    fake_app["replies"] = [(status, '{"message":"Forbidden"}')]
    with pytest.raises(Exception) as e:
        _send_to_app()
    assert len(fake_app["requests"]) == 1
    assert f"HTTP {status}" in str(e.value) and "Forbidden" in str(e.value)


def test_app_throttling_is_retried(fake_app):
    fake_app["replies"] = [(429, '{"message":"Too Many Requests"}'), (200, '{"answer": "after retry"}')]
    assert _send_to_app() == "after retry"
    assert len(fake_app["requests"]) == 2


def test_app_failing_model_check_stops_the_run(fake_app):
    fake_app["replies"] = [(403, '{"message":"Forbidden"}')]
    target = build_target()
    with pytest.raises(SystemExit, match="Forbidden"):
        asyncio.run(target_factory.check_models(target=target))


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("APP_ENDPOINT", "", "APP_ENDPOINT is not set"),
        ("APP_ENDPOINT", "abc123.execute-api.us-east-1.amazonaws.com/test/chat", "not an http"),
        ("APP_REQUEST_TEMPLATE", '{"query": PROMPT}', "not valid JSON"),
        ("APP_REQUEST_TEMPLATE", '{"query": "hello"}', "has no {PROMPT}"),
    ],
)
def test_app_settings_that_cannot_work_stop_before_any_request(fake_app, clean_env, name, value, message):
    clean_env.setenv(name, value)
    with pytest.raises(SystemExit, match=re.escape(message)):
        build_target()
    assert fake_app["requests"] == []


def test_app_target_is_labeled_by_host_and_path(fake_app):
    assert re.fullmatch(r"app:127\.0\.0\.1:\d+/test/chat", model_name(build_target()))


def test_a_model_setting_is_ignored_for_an_app(fake_app, clean_env, capsys):
    # The app picks its own model. A leftover RT_MODEL from a model run used to stop
    # the run as a provider mismatch.
    clean_env.setenv("RT_MODEL", "gemini/gemini-3.8-flash")
    assert _send_to_app() == "hello from the app"
    assert "RT_MODEL=gemini/gemini-3.8-flash is ignored" in capsys.readouterr().out

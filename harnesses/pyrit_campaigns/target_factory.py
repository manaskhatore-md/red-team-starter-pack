"""Unified provider factory - one place to decide which model gets red-teamed.

Everything else in this repo (scripts/smoke_test.py, the harnesses/pyrit_campaigns/* scripts, the
judges) asks this module for a target instead of constructing one itself. That is
what keeps the starter pack model-agnostic: an agency running OpenAI and an agency
running Claude change one env var, not the campaign code.

Three roles, each with its own provider and model setting:

    role         provider                  model                  provider if unset
    target       RT_PROVIDER               RT_MODEL               gemini
    judge        RT_JUDGE_PROVIDER         RT_JUDGE_MODEL         the target's
    adversarial  RT_ADVERSARIAL_PROVIDER   RT_ADVERSARIAL_MODEL   the judge's

Providers: gemini | openai | anthropic | bedrock | vertex | app. An unset provider
falls back to the role above it; an unset model never does - it takes the
provider's default (below). A model id belongs to one provider and one role, so
RT_MODEL only ever configures the model under test.

Most providers route through LiteLLM, so the model string is a LiteLLM model id
(https://docs.litellm.ai/docs/providers). "app" is the important one for real
assessments - it points at the agency's own deployed application rather than at a
raw model endpoint.

Providers come in two flavors, which is why they are not all one dict:
  - Key-based (gemini, openai, anthropic): one API key in an env var.
  - Credential-based (bedrock, vertex): ambient cloud credentials resolved by the
    cloud SDK, with no key to pass. These get a preflight check that prints the
    resolved identity, because "authenticated but not authorized" is the usual
    failure and it is invisible without knowing which principal you are.
"""

import json
import os
import re
import uuid

from pyrit.exceptions import RateLimitException, pyrit_target_retry
from pyrit.models import Message, MessagePiece, construct_response_from_request
from pyrit.prompt_target import LiteLLMChatTarget, PromptTarget, limit_requests_per_minute

# Key-based providers: (LiteLLM model id, env var holding the API key).
# TODO: pin these to the models your agency actually has contracts for, and add a
# row for Azure OpenAI ("azure/<deployment>", AZURE_API_KEY) if that is how you buy
# inference. Bedrock and Vertex are credential-based and handled separately below.
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "gemini": ("gemini/gemini-3.8-flash", "GEMINI_API_KEY"),
    "openai": ("openai/gpt-5", "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}

# Claude Sonnet 4.5 on Bedrock via the Converse API. Prefer "bedrock/converse/" over
# plain "bedrock/" for Anthropic models - it is the current API and handles system
# prompts and multi-turn correctly.
# TODO: change this to a model your account has actually been granted. Bedrock
# requires per-model access approval in the console, and the "us." prefix selects a
# cross-region inference profile (drop it for a single-region model id). The default
# is deliberately not the newest Claude: accounts get access to a new model weeks
# after release, and a default nobody can call fails every run.
DEFAULT_BEDROCK_MODEL = "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Generous on purpose. On reasoning models (Gemini 2.5 and later, GPT-5, Claude with extended
# thinking) internal reasoning tokens count against this same budget, so a tight limit
# truncates the visible answer mid-sentence. That silently corrupts scoring: a
# cut-off answer reads as a refusal to a judge, and a cut-off refusal reads as
# compliance.
# TODO: raise this further if you see responses ending mid-word.
DEFAULT_MAX_TOKENS = 8192

DEFAULT_VERTEX_MODEL = "claude-haiku-4-5"


class RetryingLiteLLMChatTarget(LiteLLMChatTarget):
    """LiteLLMChatTarget that retries only the errors that can clear on their own.

    PyRIT hands LiteLLM a flat retry count (RETRY_MAX_NUM_ATTEMPTS - 1, so 9), and
    LiteLLM applies it to every error. A wrong model id, a model the account has no
    access to (Bedrock's AccessDeniedException), or a rejected key then fails ten
    times per call - for every probe and every judge call - which looks like a hang.
    LiteLLM's retry_policy cannot fix that: it has no setting for 403 or 404, and
    falls back to the flat count for any error it has no setting for.

    So LiteLLM does not retry, and this does. PyRIT already raises RateLimitException
    for rate limits, timeouts, connection errors, and 5xx, and a plain PyritException
    for the rest; pyrit_target_retry retries only the first.
    """

    def _construct_request_body(self, **kwargs):
        return {**super()._construct_request_body(**kwargs), "num_retries": 0}

    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)

# role -> (provider setting, model setting, role whose provider it falls back to)
ROLES: dict[str, tuple[str, str, str | None]] = {
    "target": ("RT_PROVIDER", "RT_MODEL", None),
    "judge": ("RT_JUDGE_PROVIDER", "RT_JUDGE_MODEL", "target"),
    "adversarial": ("RT_ADVERSARIAL_PROVIDER", "RT_ADVERSARIAL_MODEL", "judge"),
}


def resolve_provider(role: str = "target") -> str:
    """The provider a role runs on: its own setting, else the fallback role's."""
    provider_var, _model_var, fallback = ROLES[role]
    provider = os.getenv(provider_var)
    if provider:
        return provider.lower()
    return resolve_provider(fallback) if fallback else "gemini"


def default_model(provider: str) -> str | None:
    """The model a provider uses when no model is set. None for "app"."""
    if provider in PROVIDER_DEFAULTS:
        return PROVIDER_DEFAULTS[provider][0]
    return {"bedrock": DEFAULT_BEDROCK_MODEL, "vertex": DEFAULT_VERTEX_MODEL}.get(provider)


def resolve_model(role: str = "target", provider: str | None = None) -> str | None:
    """The model a role calls: its own model setting, else the provider's default.

    The model setting applies only on the role's own provider. Asked about another
    provider, this returns that provider's default instead.
    """
    role_provider = resolve_provider(role)
    provider = (provider or role_provider).lower()
    model = os.getenv(ROLES[role][1])
    if model and provider == role_provider:
        return model
    return default_model(provider)


def build_target(role: str = "target", *, provider: str | None = None, model: str | None = None) -> PromptTarget:
    """Return the PyRIT target for one role.

    Args:
        role: "target" (the model under test), "judge", or "adversarial" (the
            attacker in multi-turn campaigns). Picks which settings are read - see
            ROLES.
        provider: Overrides the role's provider setting. One of PROVIDER_DEFAULTS,
            "bedrock", "vertex", or "app".
        model: Overrides the role's model setting.
    """
    provider = (provider or resolve_provider(role)).lower()
    model = model or resolve_model(role, provider)
    if provider == "app" and model:
        # The app picks its own model, so a model setting does nothing - say so rather
        # than let the run look like it tests that model.
        print(f"Note: {ROLES[role][1]}={model} is ignored - with provider app, the app picks its own model.")
        model = None
    _check_model_matches_provider(role, provider, model)
    label = {"target": "Target", "judge": "Judge", "adversarial": "Attacker"}[role]
    if role != "target" and model and model == resolve_model("target"):
        _warn_same_as_target(role, model)

    if provider in PROVIDER_DEFAULTS:
        _default, key_var = PROVIDER_DEFAULTS[provider]
        api_key = os.getenv(key_var)
        if not api_key:
            raise SystemExit(f"{key_var} is not set - see .env.example for the {provider} setup.")

        _require_litellm()
        model_name = model
        print(f"{label}: {model_name} ({provider})")
        return RetryingLiteLLMChatTarget(
            model_name=model_name,
            api_key=api_key,
            max_tokens=DEFAULT_MAX_TOKENS,
            # TODO: set max_requests_per_minute to stay inside your provider's
            # rate limit - multi-turn campaigns issue far more calls than this
            # smoke test does.
        )

    if provider == "bedrock":
        # Claude (or Llama, Mistral, Nova) on AWS Bedrock. Authenticates with SigV4
        # via the standard boto3 credential chain - env vars, a named profile, an
        # assumed role, or an ambient EC2/ECS/Lambda role. There is no API key to
        # pass, which is why api_key is absent below rather than empty.
        #
        # GOTCHA: LiteLLMChatTarget falls back to the LITELLM_API_KEY env var when no
        # api_key is given, so a leftover LITELLM_API_KEY from another provider gets
        # sent on Bedrock calls and breaks them. Unset it if Bedrock auth fails oddly.
        check_aws_creds()
        _require_litellm()

        model_name = model
        print(f"{label}: {model_name} (bedrock)")
        return RetryingLiteLLMChatTarget(
            model_name=model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            # TODO: Bedrock quotas are per-model and per-region, and lower than most
            # people expect. Set max_requests_per_minute before running a multi-turn
            # campaign or you will spend the run getting throttled.
        )

    if provider == "vertex":
        # Claude on Google Vertex. Auth is GCP ADC, not an API key, and the project
        # needs billing plus Anthropic models enabled in Vertex Model Garden.
        check_adc()
        return AnthropicVertexChatTarget(model_name=model)

    if provider == "app":
        # THIS IS THE ONE MOST AGENCIES ACTUALLY NEED. Red-teaming a raw model tells
        # you about the model; red-teaming your deployed app tells you about your
        # system prompt, your RAG index, and your tool permissions. Configured by
        # the APP_* settings - see AppChatTarget and .env.example.
        #
        # Use a non-production instance with synthetic data, and get written
        # authorization before you point this at anything.
        target = AppChatTarget.from_env()
        print(f"{label}: {target._endpoint} (app)")
        return target

    raise SystemExit(
        f"Unknown {ROLES[role][0]} {provider!r}. Options: {', '.join(PROVIDER_DEFAULTS)}, bedrock, vertex, app"
    )


# The model-id prefix LiteLLM routes on, for each provider that has one.
MODEL_PREFIXES = {"gemini": "gemini/", "openai": "openai/", "anthropic": "anthropic/", "bedrock": "bedrock/"}


def _check_model_matches_provider(role: str, provider: str, model: str | None) -> None:
    """Stop when a role's model id belongs to a different provider than the role runs on.

    LiteLLM picks the vendor from the model id's prefix, not from our provider
    setting, while the API key comes from the provider. A mismatch sends one
    vendor's key to another - which rejects it, once per retry, for every call.
    """
    owner = next((p for p, prefix in MODEL_PREFIXES.items() if model and model.startswith(prefix)), None)
    if owner is None or owner == provider:
        return
    provider_var, model_var, fallback = ROLES[role]
    if os.getenv(provider_var):
        why = f"{provider_var}={provider}"
    elif fallback:
        why = f"{provider_var} is unset, so it follows {ROLES[fallback][0]}"
    else:
        why = f"{provider_var} is unset, so it defaults to {provider}"
    raise SystemExit(
        f"{model_var}={model} is a {owner} model, but the {role} runs on {provider} ({why}). "
        f"Set {provider_var}={owner}, or change {model_var}."
    )


def _warn_same_as_target(role: str, model: str) -> None:
    """Say so when a judge or attacker is the model under test. Allowed, not advised."""
    provider_var, model_var, _fallback = ROLES[role]
    why = {
        "judge": "A model grading its own answers tends to go easy on them.",
        "adversarial": "An attacker with the target's guardrails tends to refuse to escalate.",
    }[role]
    name = {"judge": "judge", "adversarial": "attacker"}[role]
    print(f"Note: the {name} is the model under test ({model}). {why} Set {provider_var} and {model_var}.")


def _require_litellm() -> None:
    """Fail fast with an install hint instead of deep inside PyRIT."""
    try:
        import litellm  # noqa: F401  (LiteLLMChatTarget imports it lazily)
    except ImportError as e:
        raise SystemExit("LiteLLM providers need: pip install litellm") from e


def build_scoring_target() -> PromptTarget:
    """Return the target used as an LLM judge.

    Kept separate from the attacked model on purpose: a model should not grade its
    own jailbreaks, and the judge usually wants to be a stronger model than the
    system under test.

    TODO: point RT_JUDGE_PROVIDER at a different provider than RT_PROVIDER, and
    pin RT_JUDGE_MODEL - a judge that changes between runs changes the finding rate.

    GOTCHA: PyRIT's self-ask scorers require a judge that supports JSON response
    format, and LiteLLMChatTarget derives that capability from LiteLLM's model
    metadata. If you set RT_JUDGE_MODEL to a model LiteLLM does not know about (a brand-new
    release, or a custom deployment name), the capability resolves to False and
    scoring fails with "This target LiteLLMChatTarget does not support JSON response
    format" - which reads like a bug in the judge rather than a metadata gap. Check
    it up front with target.is_json_response_supported().
    """
    return build_target("judge")


class AnthropicVertexChatTarget(PromptTarget):
    """Claude via Google Vertex AI.

    PyRIT ships no Anthropic-on-Vertex target, so this implements the one abstract
    hook. Use it as the reference when you need to wrap any other SDK as a target.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        project_id: str | None = None,
        model_name: str = DEFAULT_VERTEX_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        from anthropic import AsyncAnthropicVertex

        self._region = region or os.getenv("GOOGLE_VERTEX_REGION", "us-east5")
        self._project_id = project_id or os.getenv("PROJECT_ID")
        self._max_tokens = max_tokens

        # Uses Google Application Default Credentials (ADC) automatically
        self._client = AsyncAnthropicVertex(region=self._region, project_id=self._project_id)

    @limit_requests_per_minute
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]

        message = await self._client.messages.create(
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": request.converted_value}],
            model=self._model_name,
        )
        response_text = "".join(block.text for block in message.content if block.type == "text")

        return [construct_response_from_request(request=request, response_text_pieces=[response_text])]

    async def cleanup_target_async(self) -> None:
        await self._client.close()


# Statuses that can clear on their own: throttling, and a gateway or backend that
# is briefly unavailable or slow. Anything else - a bad request, a rejected key, a
# Lambda that crashed - fails the same way every time, so it is not retried.
APP_RETRY_STATUSES = {429, 503, 504}

PROMPT_SLOT = "{PROMPT}"


class AppChatTarget(PromptTarget):
    """Your own deployed chat application, over HTTP.

    Sends each probe as one JSON request and reads the reply from one field of the
    JSON response. Settings (all from .env):

        APP_ENDPOINT          the chat URL, e.g. https://abc123.execute-api.us-east-1.amazonaws.com/test/chat
        APP_REQUEST_TEMPLATE  the request body, with {PROMPT} where the probe goes.
                              Default: {"message": "{PROMPT}"}
        APP_RESPONSE_PATH     where the reply is in the response, dotted, e.g. answer
                              or choices.0.message.content. Unset: the whole body.
        APP_API_KEY           sent as the x-api-key header (API Gateway API keys)
        APP_TOKEN             sent as Authorization: Bearer <token>
        APP_TIMEOUT           seconds to wait for a reply. Default 60.
        APP_MAX_RPM           requests per minute, to stay inside the app's throttling

    PyRIT's HTTPXAPITarget does not fit: it sends json_data as given, without
    putting the prompt into it. The template is parsed as JSON before the probe is
    put in, so a probe full of quotes and newlines cannot break the request.

    The app keeps its own system prompt and history, so this target declares
    neither: it is single-turn, and the scan sends it no stand-in system prompt.
    A multi-turn campaign against it stops with PyRIT's capability error.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        request_template: str = '{"message": "{PROMPT}"}',
        response_path: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_requests_per_minute: int | None = None,
    ):
        from urllib.parse import urlparse

        url = urlparse(endpoint)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise SystemExit(f"APP_ENDPOINT={endpoint!r} is not an http(s) URL.")
        super().__init__(
            endpoint=endpoint,
            model_name=f"app:{url.netloc}{url.path}",
            max_requests_per_minute=max_requests_per_minute,
        )
        try:
            self._template = json.loads(request_template)
        except ValueError as e:
            raise SystemExit(f"APP_REQUEST_TEMPLATE is not valid JSON ({e}): {request_template}") from e
        if PROMPT_SLOT not in request_template:
            raise SystemExit(f"APP_REQUEST_TEMPLATE has no {PROMPT_SLOT} for the probe to go in: {request_template}")
        self._response_path = response_path.strip()
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout
        self._client = None

    @classmethod
    def from_env(cls) -> "AppChatTarget":
        endpoint = os.getenv("APP_ENDPOINT", "").strip()
        if not endpoint:
            raise SystemExit("APP_ENDPOINT is not set - see the YOUR APPLICATION section of .env.example.")
        headers = {}
        if os.getenv("APP_API_KEY"):
            headers["x-api-key"] = os.environ["APP_API_KEY"].strip()
        if os.getenv("APP_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['APP_TOKEN'].strip()}"
        rpm = os.getenv("APP_MAX_RPM", "").strip()
        return cls(
            endpoint=endpoint,
            request_template=os.getenv("APP_REQUEST_TEMPLATE") or '{"message": "{PROMPT}"}',
            response_path=os.getenv("APP_RESPONSE_PATH", ""),
            headers=headers,
            timeout=float(os.getenv("APP_TIMEOUT") or 60),
            max_requests_per_minute=int(rpm) if rpm else None,
        )

    def request_body(self, prompt: str) -> str:
        """The JSON request for one probe: the template with the probe in every {PROMPT}."""

        def fill(value):
            if isinstance(value, str):
                return value.replace(PROMPT_SLOT, prompt)
            if isinstance(value, list):
                return [fill(v) for v in value]
            if isinstance(value, dict):
                return {k: fill(v) for k, v in value.items()}
            return value

        return json.dumps(fill(self._template))

    def reply_text(self, body: str) -> str:
        """The reply, read from APP_RESPONSE_PATH in the response body."""
        if not self._response_path:
            return body
        try:
            value = json.loads(body)
        except ValueError as e:
            raise ValueError(f"The app's response is not JSON, so APP_RESPONSE_PATH cannot be read: {body[:300]}") from e
        for step in self._response_path.split("."):
            if isinstance(value, list) and step.lstrip("-").isdigit() and -len(value) <= int(step) < len(value):
                value = value[int(step)]
            elif isinstance(value, dict) and step in value:
                value = value[step]
            else:
                found = sorted(value) if isinstance(value, dict) else type(value).__name__
                raise ValueError(
                    f"APP_RESPONSE_PATH={self._response_path}: no {step!r} in the response (found: {found}). "
                    f"Response: {body[:300]}"
                )
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value)

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        import httpx

        request = normalized_conversation[-1].message_pieces[0]
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        response = await self._client.post(
            self._endpoint, content=self.request_body(request.converted_value), headers=self._headers
        )
        if response.status_code in APP_RETRY_STATUSES:
            raise RateLimitException(status_code=response.status_code, message=response.text[:300])
        if not response.is_success:
            # The status alone is ambiguous on API Gateway: 403 "Missing Authentication
            # Token" is a wrong URL, 403 "Forbidden" a missing or wrong API key, and 502
            # a Lambda that crashed or returned the wrong shape. The body says which.
            raise RuntimeError(f"The app returned HTTP {response.status_code}: {response.text[:300]}")

        text = self.reply_text(response.text)
        return [construct_response_from_request(request=request, response_text_pieces=[text])]

    async def cleanup_target_async(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def check_aws_creds() -> None:
    """Fail fast with an actionable message if AWS credentials or region are missing.

    Prints the resolved IAM principal, which is the fastest way to diagnose the most
    common Bedrock failure: credentials that work fine but belong to a principal
    without bedrock:InvokeModel, or without access granted to the specific model.
    That returns AccessDeniedException, not an auth error - so the fix is an IAM
    policy or a Model Access request, not a new credential.
    """
    try:
        import boto3
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            PartialCredentialsError,
            ProfileNotFound,
            SSOError,
            TokenRetrievalError,
        )
    except ImportError as e:
        raise SystemExit("Bedrock needs the AWS SDK: pip install boto3") from e

    # Resolve credentials exactly as LiteLLM will sign with them, or this preflight
    # checks one identity while the calls use another - the one failure mode a
    # preflight must not have. LiteLLM opens a profile explicitly only for
    # AWS_PROFILE_NAME. Otherwise it takes boto3's default chain, where
    # AWS_ACCESS_KEY_ID in the environment outranks AWS_PROFILE - and "the
    # environment" includes .env, which every script loads. So AWS_PROFILE is left
    # for boto3 to read; passing it here would skip those keys.
    profile = os.getenv("AWS_PROFILE_NAME")
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    except ProfileNotFound as e:
        profile = profile or os.getenv("AWS_PROFILE")
        raise SystemExit(f"AWS profile {profile!r} not found ({e}). List them with: aws configure list-profiles") from e

    # LiteLLM reads AWS_REGION_NAME; boto3 reads AWS_REGION / AWS_DEFAULT_REGION or
    # the profile's configured region. Accept any of them so a normal AWS setup works
    # without extra configuration.
    region = (
        os.getenv("AWS_REGION_NAME")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or session.region_name
    )
    if not region:
        raise SystemExit(
            "No AWS region configured. Set AWS_REGION_NAME to a region where your "
            "Bedrock model is available (e.g. us-east-1), or run: aws configure"
        )

    # Bedrock API keys (bearer token) bypass the SigV4 credential chain entirely.
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        print(f"Authenticating to Bedrock ({region}) with AWS_BEARER_TOKEN_BEDROCK")
        return

    try:
        credentials = session.get_credentials()
    except PartialCredentialsError as e:
        raise SystemExit(
            f"Incomplete AWS keys in the environment (which includes .env): {e}.\n"
            "AWS_ACCESS_KEY_ID needs AWS_SECRET_ACCESS_KEY with it, and temporary keys (ASIA...) "
            "need AWS_SESSION_TOKEN too. Set all of them, or delete them all to use your profile."
        ) from e
    if credentials is None:
        raise SystemExit(
            "No AWS credentials found. Pick whichever fits your environment:\n"
            "  1. aws configure                       (writes a local profile)\n"
            "  2. AWS_PROFILE_NAME=<profile>          (an existing named profile)\n"
            "  3. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY [+ AWS_SESSION_TOKEN]\n"
            "  4. AWS_BEARER_TOKEN_BEDROCK=<key>      (Bedrock API key)\n"
            "  5. Nothing at all, if running on EC2/ECS/Lambda with an attached role\n"
            "The principal needs bedrock:InvokeModel plus Model Access granted for "
            "the specific model in the Bedrock console."
        )

    # Identity lookup is a network call to STS. A proxy or a blocked STS endpoint
    # should not stop the run, since the Bedrock call may still work. Credentials
    # AWS rejects outright should: every Bedrock call would fail the same way, and
    # each one is retried before it gives up.
    source = f"profile={session.profile_name}, credentials from {credentials.method}"
    identity = "credentials found (identity lookup skipped)"
    try:
        identity = session.client("sts", region_name=region).get_caller_identity()["Arn"]
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in _REJECTED_CREDENTIALS:
            raise SystemExit(_rejected_credentials_message(code, credentials.method, session.profile_name)) from e
        identity = f"credentials found (could not resolve identity: {code or type(e).__name__})"
    except (SSOError, TokenRetrievalError) as e:
        raise SystemExit(
            f"The AWS SSO login for profile {session.profile_name!r} has expired or was never made ({e}).\n"
            f"Run: aws sso login --profile {session.profile_name}"
        ) from e
    except (BotoCoreError, KeyError) as e:
        identity = f"credentials found (could not resolve identity: {type(e).__name__})"

    print(f"Authenticating to Bedrock ({region}) as {identity} [{source}]")


# STS error codes meaning AWS refused the credentials themselves.
_REJECTED_CREDENTIALS = {"InvalidClientTokenId", "SignatureDoesNotMatch", "ExpiredToken", "UnrecognizedClientException"}


def _rejected_credentials_message(code: str, method: str, profile: str) -> str:
    """Explain a rejected credential in terms of where it came from."""
    if method == "env":
        return (
            f"AWS rejected the credentials ({code}). They came from AWS_ACCESS_KEY_ID in the "
            "environment, which includes .env - every script loads it - and those keys outrank "
            "AWS_PROFILE. If that is a placeholder or an old key, delete it or comment it out "
            "in .env (and AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN with it)."
        )
    return (
        f"AWS rejected the credentials ({code}) from {method} [profile={profile}]. Refresh them "
        f"(for an SSO profile: aws sso login --profile {profile}) or pick another source - "
        "see the options above check_aws_creds in harnesses/pyrit_campaigns/target_factory.py."
    )


def check_adc() -> None:
    """Fail fast with an actionable message if Google credentials are missing."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        credentials, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except DefaultCredentialsError as e:
        raise SystemExit(
            f"No Google Cloud credentials found ({e}).\n"
            "Fix with either:\n"
            "  1. Install the Google Cloud CLI, then run: gcloud auth application-default login\n"
            "  2. Point GOOGLE_APPLICATION_CREDENTIALS at a service account key JSON "
            "(the account needs roles/aiplatform.user)."
        ) from e

    project = os.getenv("PROJECT_ID")
    if detected_project and project and detected_project != project:
        print(f"Note: ADC default project is {detected_project}, calling {project}")

    identity = getattr(credentials, "service_account_email", None) or "user credentials (gcloud ADC)"
    print(f"Authenticating to {project} ({os.getenv('GOOGLE_VERTEX_REGION', 'us-east5')}) as {identity}")


def model_name(target: PromptTarget) -> str:
    """The model id a target calls, for labeling results."""
    return target.get_identifier().params.get("model_name") or type(target).__name__


async def check_models(**targets: PromptTarget) -> None:
    """Send one short prompt to each distinct model, and stop if any of them fails.

    Called with the run's targets by role, e.g. check_models(target=t, judge=j).
    Without it, a wrong model id or a model the account cannot use fails once per
    probe and once per judge call, and the reason only shows up after the run, in
    a traceback. This costs one tiny call per model. RT_SKIP_MODEL_CHECK=1 skips it.
    """
    if os.getenv("RT_SKIP_MODEL_CHECK", "").lower() in ("1", "true", "yes"):
        print("Model check skipped (RT_SKIP_MODEL_CHECK is set).")
        return

    # A judge that is also the target is one model, so it gets one call.
    roles_by_model: dict[str, list[str]] = {}
    target_by_model: dict[str, PromptTarget] = {}
    for role, target in targets.items():
        name = model_name(target)
        roles_by_model.setdefault(name, []).append(role)
        target_by_model.setdefault(name, target)

    for name, target in target_by_model.items():
        piece = MessagePiece(role="user", original_value="Reply with the word OK.", conversation_id=str(uuid.uuid4()))
        try:
            await target.send_prompt_async(message=Message(message_pieces=[piece]))
        except Exception as e:
            roles = " and ".join(roles_by_model[name])
            raise SystemExit(
                f"Model check failed for the {roles} ({name}), so the run stopped before it started:\n"
                f"  {_provider_message(e)}\n"
                "Check the model id, that your account has access to that model (in that region, "
                "for Bedrock and Vertex), and the key or credentials. RT_SKIP_MODEL_CHECK=1 skips this check."
            ) from e
    print(f"Model check passed: {', '.join(target_by_model)}")


def _provider_message(error: Exception) -> str:
    """The provider's own error text, without the wrapping PyRIT and LiteLLM add.

    LiteLLM's "LiteLLM Retried: N times" is dropped because N is the retry count it
    was configured with, not the number of attempts it made.
    """
    text = str(error)
    text = re.sub(r"^Status Code: \d+, Message: ", "", text)
    text = re.sub(r"^LiteLLM error: ", "", text)
    text = re.sub(r"\s*LiteLLM Retried: \d+ times?\s*$", "", text)
    return text.strip()


async def close_target(target: PromptTarget) -> None:
    """Close a target if its class defines cleanup (not all of them do)."""
    if hasattr(target, "cleanup_target_async"):
        await target.cleanup_target_async()

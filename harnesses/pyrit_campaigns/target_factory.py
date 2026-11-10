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

On vertex, a Gemini model id ("vertex_ai/gemini-..." or "gemini-...") routes through
LiteLLM; any other is a Claude model, called through Anthropic's Vertex SDK.

Providers come in two flavors, which is why they are not all one dict:
  - Key-based (gemini, openai, anthropic): one API key in an env var.
  - Credential-based (bedrock, vertex): ambient cloud credentials resolved by the
    cloud SDK, with no key to pass. These get a preflight check that prints the
    resolved identity, because "authenticated but not authorized" is the usual
    failure and it is invisible without knowing which principal you are.
"""

import asyncio
import json
import os
import re
import subprocess
import uuid

from pyrit.exceptions import RateLimitException, pyrit_target_retry
from pyrit.models import Message, MessagePiece, construct_response_from_request
from pyrit.prompt_target import (
    LiteLLMChatTarget,
    PromptTarget,
    TargetCapabilities,
    TargetConfiguration,
    limit_requests_per_minute,
)

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

# Gemini on Vertex goes through LiteLLM, whose model ids for it start "vertex_ai/". A bare
# "gemini-..." id on vertex means the same model, so it gets the prefix. Some Gemini models
# are only on Vertex (gemini-4-argon-cyber-permissive, gemini-3.8-flash-cyber, for example):
# LiteLLM knows them as vertex_ai/ and not as gemini/, and a Gemini API key cannot call them.
VERTEX_GEMINI_PREFIX = "vertex_ai/"
# Gemini's own region setting, apart from GOOGLE_VERTEX_REGION (Claude's), because the two
# are usually served from different places: Claude from us-east5, Gemini from the global
# endpoint, which LiteLLM prices below a regional one (regional_endpoint_uplift_multiplier).
DEFAULT_VERTEX_GEMINI_REGION = "global"


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
        # Claude or Gemini on Google Vertex. Auth is GCP ADC, not an API key, and the
        # project needs billing plus the model enabled in Vertex Model Garden.
        gemini_model = vertex_gemini_model(model)
        if gemini_model:
            # Same GOTCHA as Bedrock: a leftover LITELLM_API_KEY gets sent on these calls.
            region = os.getenv("GOOGLE_VERTEX_GEMINI_REGION") or DEFAULT_VERTEX_GEMINI_REGION
            check_adc(region)
            _require_litellm()
            location = {"vertex_location": region}
            if os.getenv("PROJECT_ID"):
                location["vertex_project"] = os.getenv("PROJECT_ID")
            print(f"{label}: {gemini_model} (vertex, {region})")
            return RetryingLiteLLMChatTarget(
                model_name=gemini_model,
                max_tokens=DEFAULT_MAX_TOKENS,
                extra_body_parameters=location,
            )
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
        transport = os.getenv("APP_TRANSPORT", "").strip().lower() or "http"
        if transport == "browser":
            # No API to call: drive the app's own web page, as a person would.
            from harnesses.pyrit_campaigns.browser_target import BrowserChatTarget

            target = BrowserChatTarget.from_env()
            print(f"{label}: {target._url} (app, through its web page)")
            return target
        if transport == "browser-api":
            # An API that only a signed-in browser may call: the browser sends the requests.
            from harnesses.pyrit_campaigns.browser_target import BrowserApiTarget

            target = BrowserApiTarget.from_env()
            print(f"{label}: {target._endpoint} (app, called from its signed-in web page)")
            return target
        if transport != "http":
            raise SystemExit(f"APP_TRANSPORT={transport!r}: use http (the default), browser, or browser-api.")
        target = AppChatTarget.from_env()
        print(f"{label}: {target._endpoint} (app)")
        return target

    raise SystemExit(
        f"Unknown {ROLES[role][0]} {provider!r}. Options: {', '.join(PROVIDER_DEFAULTS)}, bedrock, vertex, app"
    )


# The model-id prefix LiteLLM routes on, for each provider that has one.
MODEL_PREFIXES = {
    "gemini": "gemini/",
    "openai": "openai/",
    "anthropic": "anthropic/",
    "bedrock": "bedrock/",
    "vertex": VERTEX_GEMINI_PREFIX,
}


def vertex_gemini_model(model: str | None) -> str | None:
    """The LiteLLM id of a Gemini model on Vertex, or None for any other model (Claude)."""
    if not model:
        return None
    if model.startswith(VERTEX_GEMINI_PREFIX):
        return model
    return VERTEX_GEMINI_PREFIX + model if model.startswith("gemini-") else None


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
    hint = ""
    if owner == "gemini" and provider == "vertex":
        # Gemini through Vertex rather than through a Gemini API key: same model, other prefix.
        hint = f" For Gemini on Vertex, use {model_var}={VERTEX_GEMINI_PREFIX}{model.removeprefix('gemini/')}."
    raise SystemExit(
        f"{model_var}={model} is a {owner} model, but the {role} runs on {provider} ({why}). "
        f"Set {provider_var}={owner}, or change {model_var}.{hint}"
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
        raise SystemExit("LiteLLM providers need: uv pip install litellm (or pip install litellm)") from e


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

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_system_prompt=True,
            supports_editable_history=True,
            supports_multi_turn=True,
        )
    )

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
        pieces = [piece for msg in normalized_conversation for piece in msg.message_pieces]
        system = "\n\n".join(piece.converted_value for piece in pieces if piece.api_role == "system")
        messages = [
            {"role": piece.api_role, "content": piece.converted_value}
            for piece in pieces
            if piece.api_role != "system"
        ]
        request_kwargs: dict = {
            "max_tokens": self._max_tokens,
            "messages": messages,
            "model": self._model_name,
        }
        if system:
            request_kwargs["system"] = system

        message = await self._client.messages.create(**request_kwargs)
        response_text = "".join(block.text for block in message.content if block.type == "text")

        return [construct_response_from_request(request=request, response_text_pieces=[response_text])]

    async def cleanup_target_async(self) -> None:
        await self._client.close()


# Statuses that can clear on their own: throttling, and a gateway or backend that
# is briefly unavailable or slow. Anything else - a bad request, a rejected key, a
# Lambda that crashed - fails the same way every time, so it is not retried.
APP_RETRY_STATUSES = {429, 503, 504}

PROMPT_SLOT = "{PROMPT}"
HISTORY_SLOT = "{HISTORY}"
CONVERSATION_SLOT = "{CONVERSATION_ID}"
DEFAULT_HISTORY_TURN = '{"role": "{ROLE}", "content": "{TEXT}"}'


def _fill_slots(value, slots: dict):
    """The JSON value with every slot filled. A string that is exactly one slot whose
    value is not a string (a list of turns) becomes that value; slots inside a longer
    string are filled in one pass, so a probe that happens to contain a slot's name is
    sent as written."""
    if isinstance(value, str):
        if value in slots and not isinstance(slots[value], str):
            return slots[value]
        names = [name for name, filler in slots.items() if isinstance(filler, str)]
        pattern = "|".join(re.escape(name) for name in names)
        return re.sub(pattern, lambda m: slots[m.group(0)], value) if pattern else value
    if isinstance(value, list):
        return [_fill_slots(v, slots) for v in value]
    if isinstance(value, dict):
        return {k: _fill_slots(v, slots) for k, v in value.items()}
    return value


def _holds_whole_value(value, slot: str) -> bool:
    """Whether the slot appears in the JSON value as a complete string of its own."""
    if isinstance(value, list):
        return any(_holds_whole_value(v, slot) for v in value)
    if isinstance(value, dict):
        return any(_holds_whole_value(v, slot) for v in value.values())
    return value == slot


# Stored on the assistant MessagePiece's prompt_metadata when APP_TRACE_PATH is set,
# so the judge and reports see what tools or sources the app used without putting raw
# JSON traces into converted_value (where {HISTORY}, the attacker, and regex checks read).
TRACE_METADATA_KEY = "rt_trace"
# Stored on the assistant MessagePiece's prompt_metadata when APP_CONTROL_PATH is set,
# so guardrail and reach checks can read control fields (e.g. blocked, guardrail.action)
# from single-JSON or streamed responses.
CONTROL_METADATA_KEY = "rt_control"
STREAM_MODES = ("concat", "last")
# Set on a probe's results when its judge needs the app's sources to tell a real
# specific from an invented one (needs_trace in the dataset). See trace_unavailable().
NEEDS_TRACE_LABEL = "rt_needs_trace"
MAX_TOOL_RESULT_CHARS = 4000
MAX_TRACE_CHARS = 40000
_TOOL_OUTPUT_KEYS = frozenset({"response", "result", "output"})
# Static configuration or repeated conversation fields some agent trace objects include on
# every turn (e.g. a 37 KB system_instruction, tool schema dump, or full prior history).
# Stripped when extracting a trace object so per-turn tool calls are not pushed past truncation.
_STRIP_TRACE_KEYS = frozenset({
    "system_instruction",
    "system_prompt",
    "tool_declarations",
    "tools_schema",
    "history",
    "request_message",
    "followup_prompt",
    "followup_raw_response",
})


def app_has_no_retrieval() -> bool:
    """APP_RETRIEVAL=no: the app answers from its model and system prompt alone, with no
    documents to search. Blank (the default) assumes it may have some."""
    return os.getenv("APP_RETRIEVAL", "").strip().lower() in ("no", "false", "0")


def trace_unavailable(provider: str, trace_path: str, no_retrieval: bool = False) -> bool:
    """True when a judge cannot check a specific the reply gives: the run is against an
    app with no APP_TRACE_PATH, which may have documents (no APP_RETRIEVAL=no).

    Then the judge cannot tell a form number the app found in a real document from an
    invented one, so it flags both. Decided for the run, not the reply: with
    APP_TRACE_PATH set, a reply with an empty trace retrieved nothing, so its specifics
    had no source. A bare model, or an app with no retrieval, has no documents, so the
    judge sees every source it had. Only probes marked needs_trace are affected; see
    RubricScorer.
    """
    return provider == "app" and not trace_path and not no_retrieval


def judging_mode(provider: str, trace_path: str, no_retrieval: bool, needs_trace: int, trace_found: bool = True) -> str:
    """One line saying how a run's needs_trace probes were judged, for the scan to print as
    it starts and the run summary to open with. needs_trace is how many of them ran.

    trace_found=False, from a summary of a finished run, says no reply to them had anything
    at APP_TRACE_PATH. trace_text() returns "" for a path that is not in the response, so a
    typo in the path looks like an app that retrieved nothing, and every flag would count."""
    one = needs_trace == 1
    probes, are = ("1 probe", "is") if one else (f"{needs_trace} probes", "are")
    if trace_unavailable(provider, trace_path, no_retrieval):
        return (
            f"Sources hidden: {probes} {are} reported as check by hand. A judge's flag on one is unverified, "
            "not a finding: with no APP_TRACE_PATH the judge cannot see whether the app found the specific "
            "in its own documents. Set APP_TRACE_PATH, or APP_RETRIEVAL=no if the app has no documents to search."
        )
    asks = "the probe that asks" if one else f"any of the {probes} that ask"
    if provider == "app" and trace_path and not trace_found:
        return (
            f"Sources missing: APP_TRACE_PATH={trace_path} is set, but no reply to the {probes} that "
            f"{'asks' if one else 'ask'} for a specific had anything at that path. If the path has a typo, the "
            "judge saw no sources and its flags count as findings when they should not. Check the setting; "
            "until then, treat a flag on these probes as check by hand, not a finding."
        )
    counts = f"so a judge's flag on {asks} for a specific is a finding."
    if provider == "app" and trace_path:
        return f"Sources shown: the judge sees what the app retrieved (APP_TRACE_PATH={trace_path}), {counts}"
    if provider == "app":
        return f"No documents (APP_RETRIEVAL=no): the app's only sources are its system prompt and the probe, {counts}"
    return f"Bare model: its only sources are the system prompt and the probe, {counts}"


# A judge's flag that trace_unavailable() says cannot be checked is stored under its
# rubric's name with this added, so it is not a rubric's flag: every report that counts
# rubrics' flags leaves it out unasked, as no verdict. Only the places that list these
# flags for someone to check by hand look for it, with unverified_rubric().
UNVERIFIED_SUFFIX = " (unverified)"


def unverified_category(rubric: str) -> str:
    return rubric + UNVERIFIED_SUFFIX


def unverified_rubric(category: str) -> str:
    """The rubric an unverified flag's category came from, or "" for any other category."""
    return category.removesuffix(UNVERIFIED_SUFFIX) if category.endswith(UNVERIFIED_SUFFIX) else ""


def _strip_static_trace_fields(value):
    """Remove bulky top-level config/history fields from a trace dict without touching nested tool outputs."""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in _STRIP_TRACE_KEYS}
    return value


def _trim_tool_results(value, *, max_result_chars: int = MAX_TOOL_RESULT_CHARS):
    """Cap individual tool outputs so one huge report cannot crowd out other tool calls or rounds."""
    if isinstance(value, list):
        return [_trim_tool_results(item, max_result_chars=max_result_chars) for item in value]
    if isinstance(value, dict):
        is_tool_call = any(k in value for k in ("name", "tool", "function", "arguments", "args"))
        out = {}
        for k, v in value.items():
            if is_tool_call and k in _TOOL_OUTPUT_KEYS and v is not None:
                serialized = v if isinstance(v, str) else json.dumps(v)
                if len(serialized) > max_result_chars:
                    out[k] = serialized[:max_result_chars] + "... [truncated]"
                    continue
            out[k] = _trim_tool_results(v, max_result_chars=max_result_chars)
        return out
    return value


_SSE_FIELDS = ("data", "event", "id", "retry")


def _sse_events(body: str) -> list[tuple[str, str]] | None:
    """The events of a text/event-stream body as (event type, data), or None if it isn't one.

    Follows the SSE format: a blank line ends an event, an event's data: lines are
    joined with newlines (so a reply's line breaks survive, and JSON may span lines),
    one space after the colon is dropped, and lines starting with ":" are comments.
    The last event is kept even without a closing blank line, since the body is whole.
    """
    lines = body.splitlines()
    fields = []
    for line in lines:
        name = line.split(":", 1)[0].strip() if line.strip() else ""
        fields.append(name)
    non_empty = [(line, name) for line, name in zip(lines, fields) if line.strip()]
    if not non_empty or not any(name == "data" for _, name in non_empty):
        return None
    if not all(line.lstrip().startswith(":") or name in _SSE_FIELDS for line, name in non_empty):
        return None

    events: list[tuple[str, str]] = []
    event_type, data = "", []

    def dispatch():
        nonlocal event_type, data
        # Tolerate a server that sends the [DONE] end marker without the blank line
        # that should end the event before it.
        if len(data) > 1 and data[-1].strip() == "[DONE]":
            data = data[:-1]
        if data:
            events.append((event_type or "message", "\n".join(data)))
        event_type, data = "", []

    for line in lines:
        if not line.strip():
            dispatch()
            continue
        line = line.lstrip()
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name.strip() == "data":
            data.append(value)
        elif name.strip() == "event":
            # A second event: line in one event means the server left out the blank
            # line between two events, so the first ends here. An event: line after
            # untyped data is legal SSE and names that same event's type.
            if data and event_type:
                dispatch()
            event_type = value.strip()
    dispatch()
    return events


def _event_kind(sse_type: str, payload) -> str:
    """What kind of stream event this is: its SSE event: name, or else its JSON "type".

    Streaming APIs name each event's kind one way or the other (OpenAI's Responses API
    and Anthropic's use both; the Vercel AI SDK puts it only in "type"). An event named
    neither way is a "message", the SSE format's name for an event with no event: line.
    """
    if sse_type and sse_type != "message":
        return sse_type
    if isinstance(payload, dict) and isinstance(payload.get("type"), str) and payload["type"]:
        return payload["type"]
    return sse_type or "message"


def _body_formats(content_type: str | None) -> frozenset[str] | None:
    """The formats a response's Content-Type header lets its body be read as; None for any.

    A body is read as SSE only when the header says text/event-stream, and as NDJSON or
    JSON chunks only under text/event-stream or a JSON type (application/json,
    application/x-ndjson, ...). Any other type (text/plain, text/html) is a single JSON
    document or plain text. That matters because a plain-text app's body is the model's
    own words, which a probe can steer into looking like a stream (data: lines, an
    event: error) and so change the reply or fail the request. With no header, a
    generic one, or text/event-stream, the body decides.
    """
    media = (content_type or "").split(";")[0].strip().lower()
    if media in ("", "application/octet-stream", "*/*", "text/event-stream"):
        return None
    if "json" in media:
        return frozenset({"json", "ndjson"})
    return frozenset({"json"})


def _labeled_events(body: str, content_type: str | None = None) -> tuple[str | None, list[tuple[str, str, object]]]:
    """_parse_response_events, with each event's SSE event: name ("" outside SSE) and kind kept.

    Returns (kind, [(sse_type, event_kind, payload), ...]); see _event_kind for event_kind.
    """
    formats = _body_formats(content_type)
    if formats is None or "json" in formats:
        try:
            return None, [("", "", json.loads(body))]
        except ValueError:
            pass

    sse = _sse_events(body) if formats is None or "sse" in formats else None
    if sse is not None:
        events = []
        for event_type, payload in sse:
            if not payload.strip() or payload.strip() == "[DONE]":
                continue
            try:
                parsed = [json.loads(payload)]
            except ValueError:
                parsed = [payload]
                lines = [ln for ln in payload.splitlines() if ln.strip() and ln.strip() != "[DONE]"]
                if len(lines) > 1:
                    try:
                        parsed = [json.loads(ln) for ln in lines]
                    except ValueError:
                        pass
            events.extend((event_type, _event_kind(event_type, p), p) for p in parsed)
        return "SSE", events

    if formats is not None and "ndjson" not in formats:
        return None, []
    decoder = json.JSONDecoder()
    idx, n = 0, len(body)
    chunks = []
    try:
        while idx < n:
            while idx < n and body[idx] in " \t\r\n\x00":
                idx += 1
            if idx >= n:
                break
            obj, idx = decoder.raw_decode(body, idx)
            chunks.append(obj)
    except ValueError:
        chunks = []
    if len(chunks) >= 2:
        # An unnamed chunk is a "message", as an unnamed SSE event is.
        return "NDJSON", [("", _event_kind("message", c), c) for c in chunks]
    return None, []


def _parse_response_events(body: str, content_type: str | None = None) -> tuple[str | None, list]:
    """Parse a response body as a single JSON value, an SSE event stream, or NDJSON / JSON chunks.

    Returns (kind, events) where kind is None for a single JSON document, "SSE" for
    text/event-stream, or "NDJSON" for newline-delimited or concatenated JSON chunks
    (such as AWS Lambda response streaming). Returns (None, []) when the body is not
    JSON or an event stream. An SSE event's data is parsed as JSON when it is JSON and
    kept as text otherwise; the [DONE] end marker and empty data are left out. The
    response's Content-Type, if given, limits the formats tried: see _body_formats.
    """
    kind, events = _labeled_events(body, content_type)
    return kind, [payload for _sse_type, _kind, payload in events]


def _unread_stream(body: str, content_type: str | None) -> tuple[str, str] | None:
    """(the stream kind the body looks like, its Content-Type) when that header kept the
    body from being read as that stream; None otherwise."""
    if _body_formats(content_type) is None or _labeled_events(body, content_type)[0] is not None:
        return None
    guessed = _labeled_events(body)[0]
    if guessed is None:
        return None
    return guessed, (content_type or "").split(";")[0].strip()


STREAM_TYPES_HINT = "a streaming app sends Content-Type text/event-stream (SSE) or application/x-ndjson"


def _stream_error(body: str, content_type: str | None = None) -> str:
    """The error a streamed response ended with, or "" if it has none.

    A stream can start a reply and then fail (the model was overloaded, the connection
    to it dropped). What arrived before the error is a cut-off reply, so it must not be
    graded as the app's answer. Counts as an error: an SSE event of type "error", or a
    streamed JSON event with a non-empty top-level "error" field. A single JSON response
    is left alone, since its "error" field (if any) is the app's own business.
    """
    formats = _body_formats(content_type)
    sse = _sse_events(body) if formats is None or "sse" in formats else None
    if sse is not None:
        for event_type, payload in sse:
            if event_type == "error":
                return payload.strip() or "(no details)"
    kind, events = _parse_response_events(body, content_type)
    if kind is None:
        return ""
    for event in events:
        if isinstance(event, dict) and event.get("error"):
            return json.dumps(event)
    return ""


def _step_path(root, path: str) -> tuple[bool, object]:
    """Resolve one dotted path on a JSON value, supporting list indices and list field projection."""
    value = root
    for step in path.split("."):
        if isinstance(value, list) and step.lstrip("-").isdigit() and -len(value) <= int(step) < len(value):
            value = value[int(step)]
        elif isinstance(value, list):
            projected = []
            for item in value:
                if isinstance(item, dict) and step in item:
                    sub = item[step]
                    if isinstance(sub, list):
                        projected.extend(sub)
                    elif sub is not None:
                        projected.append(sub)
            if not projected:
                return False, None
            value = projected
        elif isinstance(value, dict) and step in value:
            value = value[step]
        else:
            return False, None
    return True, value


def _merge_event_values(values: list):
    """Combine a path's values across streaming events: concatenate lists, merge dicts, or take the latest scalar."""
    if len(values) == 1:
        return values[0]
    if all(isinstance(v, list) for v in values):
        return [item for sublist in values for item in sublist]
    if all(isinstance(v, dict) for v in values):
        merged: dict = {}
        for d in values:
            for k, v in d.items():
                if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                    merged[k] = [*merged[k], *v]
                elif k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = _merge_event_values([merged[k], v])
                else:
                    merged[k] = v
        return merged
    return values[-1]


def _extract_paths(events: list, paths_spec: str, *, clean_trace: bool = False) -> str:
    """Read one or more comma-separated dotted paths from parsed response event(s)."""
    if not paths_spec or not events:
        return ""
    paths = [p.strip() for p in paths_spec.split(",") if p.strip()]
    extracted: dict[str, object] = {}
    for path in paths:
        values = []
        for event in events:
            found, val = _step_path(event, path)
            if not found:
                continue
            cleaned = _trim_tool_results(_strip_static_trace_fields(val)) if clean_trace else val
            if cleaned in (None, "", [], {}) and not isinstance(cleaned, bool):
                continue
            values.append(cleaned)
        if not values:
            continue
        extracted[path] = _merge_event_values(values)
    if not extracted:
        return ""
    if len(paths) == 1:
        only = next(iter(extracted.values()))
        return only if isinstance(only, str) else json.dumps(only)
    return json.dumps(extracted)


def run_token_command(command: str, timeout: float) -> str:
    """Run APP_TOKEN_COMMAND through the shell (cmd.exe on Windows) and return what it printed.

    Error messages never show the command, which may carry a password or client secret,
    or its standard output, which may hold a token: only its exit code and standard
    error. Its input is empty, so a command that asks for something fails instead of
    waiting. On a timeout the command's whole process tree is stopped: stopping only the
    shell would leave the program it started running, holding the output open, and the
    run would wait for that program anyway.
    """
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **group,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_process_tree(proc)
        # from None: Python's own timeout error quotes the whole command.
        raise RuntimeError(
            f"APP_TOKEN_COMMAND did not finish within {timeout:g} s (APP_TIMEOUT) and was stopped. "
            "It must print a token without waiting for anything, such as a sign-in in the browser."
        ) from None
    if proc.returncode != 0:
        detail = (err or "").strip()[:300] or "(nothing on standard error)"
        raise RuntimeError(f"APP_TOKEN_COMMAND failed (exit {proc.returncode}): {detail}")
    return out


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """Stop a command started by run_token_command, with everything it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class AppChatTarget(PromptTarget):
    """Your own deployed chat application, over HTTP.

    Sends each probe as one JSON request and reads the reply from one field of the
    JSON response (or from an SSE / NDJSON / Lambda response stream). Settings (all from .env):

        APP_ENDPOINT          the chat URL, e.g. https://abc123.execute-api.us-east-1.amazonaws.com/test/chat
        APP_REQUEST_TEMPLATE  the request body, with {PROMPT} where the probe goes.
                              Default: {"message": "{PROMPT}"}
                              Optional slots: {CONVERSATION_ID}, a fresh id for each
                              conversation, and "{HISTORY}" (quoted, as a whole value),
                              which becomes the list of the conversation's earlier turns
        APP_HISTORY_TURN      the shape of one earlier turn in {HISTORY}, with {ROLE}
                              and {TEXT}. Default: {"role": "{ROLE}", "content": "{TEXT}"}
        APP_HISTORY_ROLES     the app's names for the user's turns and its own, in that
                              order. Default: user,assistant
        APP_RESPONSE_PATH     where the reply is in the response, dotted, e.g. answer
                              or choices.0.message.content. Unset: the whole body.
        APP_STREAM_MODE       how to combine APP_RESPONSE_PATH across streamed SSE or
                              NDJSON events: concat (the default, joins the pieces, or
                              takes the last one if each repeats the reply so far) or
                              last (takes the last event carrying the path).
        APP_STREAM_EVENT      which kinds of streamed event carry the reply, comma-separated:
                              matched against each event's SSE event: name or its JSON
                              "type", e.g. response.output_text.delta or text-delta. An
                              event named neither way is a "message". Unset: every event,
                              but a reply whose text comes from more than one kind is an
                              error, since the others are likely tool-call arguments or
                              the model's reasoning. Only the reply is narrowed: trace and
                              control fields are still read from every event.
        APP_TRACE_PATH        optional: where tool calls, retrieved sources, or citations
                              are in the response, dotted (or comma-separated paths), e.g.
                              trace.rounds or citations. Shown to the judge in
                              <response_trace> and included in reports, without being
                              sent back in {HISTORY} or scanned as user-visible reply text.
        APP_CONTROL_PATH      optional: where guardrail or routing control fields are in
                              the response, dotted (or comma-separated paths), e.g.
                              blocked,guardrail. Stored in prompt_metadata["rt_control"].
        APP_API_KEY           sent as the x-api-key header (API Gateway API keys)
        APP_TOKEN             sent as Authorization: Bearer <token>
        APP_TOKEN_COMMAND     optional: a shell command (cmd.exe on Windows) that prints
                              a fresh bearer token, run before the first request and
                              again on HTTP 401, then the request is retried once. It
                              replaces APP_TOKEN. Not used with APP_TRANSPORT=browser-api.
        APP_TIMEOUT           seconds to wait for a reply. Default 60.
        APP_MAX_RPM           requests per minute, to stay inside the app's throttling

    PyRIT's HTTPXAPITarget does not fit: it sends json_data as given, without
    putting the prompt into it. The template is parsed as JSON before the probe is
    put in, so a probe full of quotes and newlines cannot break the request.

    Without {HISTORY}, the app keeps its own history, so this target is single-turn
    and a multi-turn campaign against it stops with PyRIT's capability error. With
    {HISTORY}, the app remembers nothing between requests and takes the whole
    conversation from each one, so the kit holds the history and may rewrite it: the
    target declares multi-turn and editable history, which Crescendo's backtracking
    needs. Either way it declares no system prompt, since the app sends its own, and
    the scan sends it no stand-in.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        request_template: str = '{"message": "{PROMPT}"}',
        history_turn: str = DEFAULT_HISTORY_TURN,
        history_roles: str = "user,assistant",
        response_path: str = "",
        stream_mode: str = "concat",
        stream_event: str = "",
        trace_path: str = "",
        control_path: str = "",
        token_command: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_requests_per_minute: int | None = None,
    ):
        from urllib.parse import urlparse

        url = urlparse(endpoint)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise SystemExit(f"APP_ENDPOINT={endpoint!r} is not an http(s) URL.")
        try:
            self._template = json.loads(request_template)
        except ValueError as e:
            raise SystemExit(f"APP_REQUEST_TEMPLATE is not valid JSON ({e}): {request_template}") from e
        if PROMPT_SLOT not in request_template:
            raise SystemExit(f"APP_REQUEST_TEMPLATE has no {PROMPT_SLOT} for the probe to go in: {request_template}")
        mode = (stream_mode or "concat").strip().lower()
        if mode not in STREAM_MODES:
            raise SystemExit(f"APP_STREAM_MODE={stream_mode!r}: use concat (the default) or last.")

        # None: the app keeps its own history, and requests carry only the new message.
        self._history_turn = None
        if HISTORY_SLOT in request_template:
            if not _holds_whole_value(self._template, HISTORY_SLOT):
                raise SystemExit(
                    f'APP_REQUEST_TEMPLATE: put {HISTORY_SLOT} in quotes as a whole value, e.g. "history": '
                    f'"{HISTORY_SLOT}". It is replaced by a list of turns, not by text: {request_template}'
                )
            try:
                self._history_turn = json.loads(history_turn)
            except ValueError as e:
                raise SystemExit(f"APP_HISTORY_TURN is not valid JSON ({e}): {history_turn}") from e
            if "{TEXT}" not in history_turn:
                raise SystemExit(f"APP_HISTORY_TURN has no {{TEXT}} for a turn's text to go in: {history_turn}")
            roles = [r.strip() for r in history_roles.split(",")]
            if len(roles) != 2 or not all(roles):
                raise SystemExit(
                    f"APP_HISTORY_ROLES={history_roles!r}: give two names, the user's and the app's, e.g. user,assistant"
                )
            self._roles = {"user": roles[0], "assistant": roles[1]}

        super().__init__(
            endpoint=endpoint,
            model_name=f"app:{url.netloc}{url.path}",
            max_requests_per_minute=max_requests_per_minute,
            custom_configuration=TargetConfiguration(
                capabilities=TargetCapabilities(supports_multi_turn=True, supports_editable_history=True)
            ) if self._history_turn is not None else None,
        )
        self._response_path = response_path.strip()
        self._stream_mode = mode
        self._stream_events = frozenset(n.strip() for n in stream_event.split(",") if n.strip())
        self._trace_path = trace_path.strip()
        self._control_path = control_path.strip()
        self._token_command = token_command.strip()
        self._token_lock = asyncio.Lock()
        self._token_loads = 0
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout
        self._client = None
        self._noted_unread_stream = False
        self._noted_no_chosen_events = False

    @staticmethod
    def settings_from_env() -> dict:
        """The constructor's settings from the APP_* variables, for this class and the browser-api one."""
        headers = {}
        if os.getenv("APP_API_KEY"):
            headers["x-api-key"] = os.environ["APP_API_KEY"].strip()
        if os.getenv("APP_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['APP_TOKEN'].strip()}"
        rpm = os.getenv("APP_MAX_RPM", "").strip()
        return dict(
            endpoint=os.getenv("APP_ENDPOINT", "").strip(),
            request_template=os.getenv("APP_REQUEST_TEMPLATE") or '{"message": "{PROMPT}"}',
            history_turn=os.getenv("APP_HISTORY_TURN") or DEFAULT_HISTORY_TURN,
            history_roles=os.getenv("APP_HISTORY_ROLES") or "user,assistant",
            response_path=os.getenv("APP_RESPONSE_PATH", ""),
            stream_mode=os.getenv("APP_STREAM_MODE") or "concat",
            stream_event=os.getenv("APP_STREAM_EVENT", ""),
            trace_path=os.getenv("APP_TRACE_PATH", ""),
            control_path=os.getenv("APP_CONTROL_PATH", ""),
            token_command=os.getenv("APP_TOKEN_COMMAND", ""),
            headers=headers,
            timeout=float(os.getenv("APP_TIMEOUT") or 60),
            max_requests_per_minute=int(rpm) if rpm else None,
        )

    @classmethod
    def from_env(cls) -> "AppChatTarget":
        if not os.getenv("APP_ENDPOINT", "").strip():
            raise SystemExit("APP_ENDPOINT is not set - see the YOUR APPLICATION section of .env.example.")
        return cls(**cls.settings_from_env())

    @property
    def sends_history(self) -> bool:
        """Whether each request carries the conversation so far ({HISTORY} in the template)."""
        return self._history_turn is not None

    def request_body(self, prompt: str, history: list[tuple[str, str]] = (), conversation_id: str = "") -> str:
        """The JSON request for one message: the template with the message in every {PROMPT},
        the conversation's id in every {CONVERSATION_ID}, and its earlier turns, given as
        (role, text) with role "user" or "assistant", in {HISTORY}."""
        turns = [
            _fill_slots(self._history_turn, {"{ROLE}": self._roles[role], "{TEXT}": text}) for role, text in history
        ]
        slots = {PROMPT_SLOT: prompt, CONVERSATION_SLOT: conversation_id}
        if self.sends_history:
            slots[HISTORY_SLOT] = turns
        return json.dumps(_fill_slots(self._template, slots))

    @staticmethod
    def history_of(conversation: list[Message]) -> list[tuple[str, str]]:
        """The turns before the newest message, as (role, text). PyRIT's simulated
        assistant turns (a conversation seeded with replies the app never gave) are sent
        as the app's own, which is how an app that trusts its client would see them."""
        turns = []
        for message in conversation[:-1]:
            piece = message.message_pieces[0]
            if piece.api_role not in ("user", "assistant"):
                raise ValueError(
                    f"A {piece.api_role!r} message cannot go in {HISTORY_SLOT}: an app's history holds only the "
                    "user's turns and its own. A deployed app sends its own system prompt."
                )
            turns.append((piece.api_role, piece.converted_value))
        return turns

    def reply_text(self, body: str, content_type: str | None = None) -> str:
        """The reply, read from APP_RESPONSE_PATH in the response body (single JSON, SSE, or NDJSON).

        content_type is the response's Content-Type header, which decides whether the body
        may be read as a stream (see _body_formats); None or "" lets the body decide.
        """
        kind, labeled = _labeled_events(body, content_type)
        if kind is not None:
            chosen = self._chosen_events(labeled)
            if labeled and not chosen:
                # A reply that is only a tool call or a refusal has none of the answer's
                # events. It is still the app's reply, so it is graded (as empty text,
                # with its trace) instead of failing the request.
                return ""
            labeled = chosen
        events = [payload for _sse_type, _kind, payload in labeled]
        if not self._response_path:
            if kind == "SSE":
                # Only a JSON object or list means the reply sits in a field. A plain-text
                # token such as 15 or true also parses as JSON, so join the raw text.
                structured = next((e for e in events if isinstance(e, (dict, list))), None)
                if structured is not None:
                    raise ValueError(
                        "The app's response is a stream of JSON events, so the reply is in one of their fields: "
                        "set APP_RESPONSE_PATH to it (e.g. choices.0.delta.content). "
                        f"First JSON event: {json.dumps(structured)[:300]}"
                    )
                # Text tokens, read raw: a token that is only a space is part of the text.
                texts = [
                    (t, t, p)
                    for t, p in _sse_events(body) or []
                    if p and p.strip() != "[DONE]" and self._is_chosen(t, p)
                ]
                self._check_one_kind(texts, "the stream's text")
                return self._combine_pieces([p for _t, _k, p in texts])
            return body
        if kind is None and not events:
            unread = _unread_stream(body, content_type)
            why = (
                f" (it looks like {unread[0]}, but its Content-Type is {unread[1]}, so it is not read as a "
                f"stream; {STREAM_TYPES_HINT})"
                if unread
                else ""
            )
            raise ValueError(f"The app's response is not JSON, so APP_RESPONSE_PATH cannot be read{why}: {body[:300]}")
        if kind is None:
            value = events[0]
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

        pieces: list[str] = []
        found_in: list[tuple[str, str, str]] = []
        matched = False
        for sse_type, event_kind, event in labeled:
            found, value = _step_path(event, self._response_path)
            if found:
                matched = True
            if found and value is not None:
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    text = list(value)
                else:
                    text = [value if isinstance(value, str) else json.dumps(value)]
                pieces.extend(text)
                found_in.append((sse_type, event_kind, "".join(text)))
        if not matched:
            first = (
                (events[0] if isinstance(events[0], str) else json.dumps(events[0]))
                if events
                else body.strip()
            )
            chosen = f" among the {', '.join(sorted(self._stream_events))} events" if self._stream_events else ""
            raise ValueError(
                f"response looks like {kind} but no event{chosen} had APP_RESPONSE_PATH={self._response_path}; "
                f"first event: {first[:300]}"
            )
        self._check_one_kind(found_in, f"APP_RESPONSE_PATH={self._response_path}")
        return self._combine_pieces(pieces)

    def _is_chosen(self, event_kind: str, payload) -> bool:
        """Whether APP_STREAM_EVENT (if set) names this event: by its kind (its SSE event:
        name, else its JSON "type"; see _event_kind) or by its JSON "type"."""
        if not self._stream_events:
            return True
        names = {event_kind}
        if isinstance(payload, dict) and isinstance(payload.get("type"), str):
            names.add(payload["type"])
        return bool(names & self._stream_events)

    def _chosen_events(self, labeled: list) -> list:
        """The stream events APP_STREAM_EVENT names (all of them if it is unset).

        A stream with none of them is normal for one reply (a tool call, a refusal), so
        it gives no events, not an error; but every reply having none is a typo in the
        setting, so the first time it happens, a note lists the kinds the stream had.
        """
        chosen = [e for e in labeled if self._is_chosen(e[1], e[2])]
        if labeled and not chosen and not self._noted_no_chosen_events:
            self._noted_no_chosen_events = True
            seen = list(dict.fromkeys(k for _t, k, _p in labeled))
            print(
                f"\nNote: a reply from the app had no APP_STREAM_EVENT={','.join(sorted(self._stream_events))} "
                f"events, so its reply text was read as empty. Its events were: {', '.join(seen)}. That is normal "
                "for a reply that is only a tool call or a refusal; if every reply comes back empty, check "
                "APP_STREAM_EVENT for a typo."
            )
        return chosen

    def _check_one_kind(self, found_in: list[tuple[str, str, str]], what: str) -> None:
        """Stop a reply that would join text from more than one kind of stream event.

        Streaming APIs send more than the answer, in events of their own kinds: a tool
        call's arguments, the model's reasoning. When those use the same field as the
        answer, joining them all would grade them as the reply, with nothing to show for
        it. So unless APP_STREAM_EVENT says which kinds carry the reply, text from more
        than one kind is an error. Events with no text don't count.
        """
        if self._stream_events:
            return
        first: dict[str, str] = {}
        for _sse_type, event_kind, text in found_in:
            if text and event_kind not in first:
                first[event_kind] = text
        if len(first) < 2:
            return
        samples = "; ".join(f"{k}: {json.dumps(v[:80])}" for k, v in first.items())
        raise ValueError(
            f"{what} is in {len(first)} kinds of stream event ({', '.join(first)}), so joining them would "
            "mix other text into the reply, such as a tool call's arguments or the model's reasoning. Set "
            f"APP_STREAM_EVENT to the kind(s) that carry the reply, comma-separated, e.g. "
            f"APP_STREAM_EVENT={next(iter(first))}. First text of each kind: {samples}"
        )

    def _combine_pieces(self, pieces: list[str]) -> str:
        """Join a stream's text pieces into the reply, per APP_STREAM_MODE.

        last keeps the final non-empty piece. concat joins them, unless each piece is
        the one before it plus more ("Hel", "Hello", "Hello!"): that app resends the
        whole reply so far each time, so joining would repeat it, and the last piece is
        the reply.
        """
        non_empty = [p for p in pieces if p]
        if not non_empty:
            return ""
        if self._stream_mode == "last":
            return non_empty[-1]
        snapshots = (
            len(non_empty) >= 2
            and bool(non_empty[0].strip())
            and len(non_empty[-1]) > len(non_empty[0])
            and all(cur.startswith(prev) for prev, cur in zip(non_empty, non_empty[1:]))
        )
        return non_empty[-1] if snapshots else "".join(non_empty)

    def trace_text(self, body: str, content_type: str | None = None) -> str:
        """The response trace, read from APP_TRACE_PATH in the response body ("" if unset or absent).

        Supports one dotted path (e.g. trace.rounds or citations) or several
        comma-separated paths, across single-JSON or streamed (SSE / NDJSON) responses.
        Stepping by field name into a list of dicts collects that field across the list
        (so trace.rounds.tool_calls collects tool_calls from every round). Unlike
        APP_RESPONSE_PATH, a missing or empty trace on a turn is normal (for example, a
        turn that called no tools) and returns "".
        """
        if not self._trace_path:
            return ""
        _kind, events = _parse_response_events(body, content_type)
        text = _extract_paths(events, self._trace_path, clean_trace=True)
        if not text:
            return ""
        max_chars = int(os.getenv("APP_TRACE_MAX_CHARS") or MAX_TRACE_CHARS)
        if len(text) > max_chars:
            head = max_chars // 3
            tail = max_chars - head
            return text[:head] + " ... [truncated] ... " + text[-tail:]
        return text

    def control_text(self, body: str, content_type: str | None = None) -> str:
        """The guardrail/control fields, read from APP_CONTROL_PATH in the response body ("" if unset or absent)."""
        if not self._control_path:
            return ""
        _kind, events = _parse_response_events(body, content_type)
        return _extract_paths(events, self._control_path, clean_trace=False)

    async def _refresh_token(self) -> None:
        """Run APP_TOKEN_COMMAND to get a fresh bearer token and store it in the Authorization header."""
        token = (await asyncio.to_thread(run_token_command, self._token_command, self._timeout)).strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            raise RuntimeError("APP_TOKEN_COMMAND printed an empty token.")
        self._headers["Authorization"] = f"Bearer {token}"
        self._token_loads += 1

    async def _send_http(self, body: str) -> tuple[int, str, str]:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        response = await self._client.post(self._endpoint, content=body, headers=self._headers)
        return response.status_code, response.text, response.headers.get("content-type", "")

    async def _post(self, body: str) -> tuple[int, str, str]:
        """Send one request body to APP_ENDPOINT; return the status, the response body, and its Content-Type."""
        if self._token_command and self._token_loads == 0:
            async with self._token_lock:
                if self._token_loads == 0:
                    await self._refresh_token()
        loads = self._token_loads
        status, response_body, content_type = await self._send_http(body)
        if status == 401 and self._token_command:
            async with self._token_lock:
                if self._token_loads == loads:
                    await self._refresh_token()
            status, response_body, content_type = await self._send_http(body)
        return status, response_body, content_type

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]
        history = self.history_of(normalized_conversation) if self.sends_history else []
        status, body, content_type = await self._post(
            self.request_body(request.converted_value, history, request.conversation_id)
        )
        if status in APP_RETRY_STATUSES:
            raise RateLimitException(status_code=status, message=body[:300])
        if not 200 <= status < 300:
            # The status alone is ambiguous on API Gateway: 403 "Missing Authentication
            # Token" is a wrong URL, 403 "Forbidden" a missing or wrong API key, and 502
            # a Lambda that crashed or returned the wrong shape. The body says which.
            raise RuntimeError(f"The app returned HTTP {status}: {body[:300]}")
        error = _stream_error(body, content_type)
        if error:
            # Whatever text came before the error is a cut-off reply: fail the request
            # like an HTTP error, rather than grade part of an answer as the whole one.
            raise RuntimeError(f"The app's streamed reply ended with an error: {error[:300]}")

        # Without APP_RESPONSE_PATH, a stream-like reply under a plain type is read as
        # plain text with no error, so say once that it happened. (With the path set,
        # reply_text's error says it.)
        unread = None if self._response_path or self._noted_unread_stream else _unread_stream(body, content_type)
        if unread:
            self._noted_unread_stream = True
            print(
                f"\nNote: a reply from the app looked like {unread[0]}, but its Content-Type is {unread[1]}, so it "
                f"was read as it came. If the app streams its replies, {STREAM_TYPES_HINT}; if it doesn't, the "
                "model's reply only looked like a stream."
            )
        text = self.reply_text(body, content_type)
        trace = self.trace_text(body, content_type)
        control = self.control_text(body, content_type)
        metadata = {}
        if trace:
            metadata[TRACE_METADATA_KEY] = trace
        if control:
            metadata[CONTROL_METADATA_KEY] = control
        return [
            construct_response_from_request(
                request=request,
                response_text_pieces=[text],
                prompt_metadata=metadata or None,
            )
        ]

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
        raise SystemExit("Bedrock needs the AWS SDK: uv pip install boto3 (or pip install boto3)") from e

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


def check_adc(region: str | None = None) -> None:
    """Fail fast with an actionable message if Google credentials are missing.

    region is the one the call goes to, for the message; Claude's GOOGLE_VERTEX_REGION if None.
    """
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
    region = region or os.getenv("GOOGLE_VERTEX_REGION", "us-east5")
    print(f"Authenticating to {project or detected_project} ({region}) as {identity}")


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

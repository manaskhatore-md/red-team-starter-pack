"""Unified provider factory - one place to decide which model gets red-teamed.

Everything else in this repo (smoke_test.py, the pyrit_campaigns/* scripts, the
judges) asks this module for a target instead of constructing one itself. That is
what keeps the starter pack model-agnostic: an agency running OpenAI and an agency
running Claude change one env var, not the campaign code.

Provider selection: RT_PROVIDER = gemini | openai | anthropic | vertex | app
Model override:     RT_MODEL (defaults per provider below)

Most providers route through LiteLLM, so the model string is a LiteLLM model id
(https://docs.litellm.ai/docs/providers). "app" is the important one for real
assessments - it points at the agency's own deployed application rather than at a
raw model endpoint.
"""

import os

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import LiteLLMChatTarget, PromptTarget, limit_requests_per_minute

# Per-provider defaults: (LiteLLM model id, env var holding the API key).
# TODO: pin these to the models your agency actually has contracts for, and add
# rows for Azure OpenAI ("azure/<deployment>", AZURE_API_KEY) or Bedrock
# ("bedrock/anthropic.claude-...", AWS creds) if that is how you buy inference.
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "gemini": ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    "openai": ("openai/gpt-5", "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}

# Generous on purpose. On reasoning models (Gemini 2.5, GPT-5, Claude with extended
# thinking) internal reasoning tokens count against this same budget, so a tight limit
# truncates the visible answer mid-sentence. That silently corrupts scoring: a
# cut-off answer reads as a refusal to a judge, and a cut-off refusal reads as
# compliance.
# TODO: raise this further if you see responses ending mid-word.
DEFAULT_MAX_TOKENS = 8192


def build_target(*, provider: str | None = None, model: str | None = None) -> PromptTarget:
    """Return the PyRIT target to attack.

    Args:
        provider: Overrides RT_PROVIDER. One of PROVIDER_DEFAULTS, "vertex", or "app".
        model: Overrides RT_MODEL.
    """
    provider = (provider or os.getenv("RT_PROVIDER", "gemini")).lower()

    if provider in PROVIDER_DEFAULTS:
        default_model, key_var = PROVIDER_DEFAULTS[provider]
        api_key = os.getenv(key_var)
        if not api_key:
            raise SystemExit(f"{key_var} is not set - see .env.example for the {provider} setup.")

        try:
            import litellm  # noqa: F401  (LiteLLMChatTarget imports it lazily)
        except ImportError as e:
            raise SystemExit("LiteLLM providers need: pip install litellm") from e

        model_name = model or os.getenv("RT_MODEL") or default_model
        print(f"Target: {model_name} ({provider})")
        return LiteLLMChatTarget(
            model_name=model_name,
            api_key=api_key,
            max_tokens=DEFAULT_MAX_TOKENS,
            # TODO: set max_requests_per_minute to stay inside your provider's
            # rate limit - multi-turn campaigns issue far more calls than this
            # smoke test does.
        )

    if provider == "vertex":
        # Claude on Google Vertex. Auth is GCP ADC, not an API key, and the project
        # needs billing plus Anthropic models enabled in Vertex Model Garden.
        check_adc()
        return AnthropicVertexChatTarget(model_name=model or os.getenv("RT_MODEL") or "claude-haiku-4-5")

    if provider == "app":
        # TODO: THIS IS THE ONE MOST AGENCIES ACTUALLY NEED.
        # Red-teaming a raw model tells you about the model; red-teaming your
        # deployed app tells you about your system prompt, your RAG index, and
        # your tool permissions. Point HTTPXAPITarget at your app's chat endpoint:
        #
        #   from pyrit.prompt_target import HTTPXAPITarget, get_http_target_json_response_callback_function
        #   return HTTPXAPITarget(
        #       http_url=os.environ["APP_ENDPOINT"],
        #       method="POST",
        #       headers={"Authorization": f"Bearer {os.environ['APP_TOKEN']}"},
        #       json_data={"message": "{PROMPT}"},   # your app's request shape
        #       callback_function=get_http_target_json_response_callback_function(key="reply"),
        #   )
        #
        # Use a non-production instance with synthetic data, and get written
        # authorization before you point this at anything.
        raise SystemExit("The 'app' provider is a TODO - see pyrit_campaigns/target_factory.py")

    raise SystemExit(f"Unknown RT_PROVIDER {provider!r}. Options: {', '.join(PROVIDER_DEFAULTS)}, vertex, app")


def build_scoring_target() -> PromptTarget:
    """Return the target used as an LLM judge.

    Kept separate from the attacked model on purpose: a model should not grade its
    own jailbreaks, and the judge usually wants to be a stronger model than the
    system under test.

    TODO: point RT_JUDGE_PROVIDER at a different provider than RT_PROVIDER.

    GOTCHA: PyRIT's self-ask scorers require a judge that supports JSON response
    format, and LiteLLMChatTarget derives that capability from LiteLLM's model
    metadata. If you set RT_MODEL to a model LiteLLM does not know about (a brand-new
    release, or a custom deployment name), the capability resolves to False and
    scoring fails with "This target LiteLLMChatTarget does not support JSON response
    format" - which reads like a bug in the judge rather than a metadata gap. Check
    it up front with target.is_json_response_supported().
    """
    return build_target(provider=os.getenv("RT_JUDGE_PROVIDER", os.getenv("RT_PROVIDER", "gemini")))


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
        model_name: str = "claude-haiku-4-5",
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


async def close_target(target: PromptTarget) -> None:
    """Close a target if its class defines cleanup (not all of them do)."""
    if hasattr(target, "cleanup_target_async"):
        await target.cleanup_target_async()

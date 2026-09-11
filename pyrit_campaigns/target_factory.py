"""Unified provider factory - one place to decide which model gets red-teamed.

Everything else in this repo (smoke_test.py, the pyrit_campaigns/* scripts, the
judges) asks this module for a target instead of constructing one itself. That is
what keeps the starter pack model-agnostic: an agency running OpenAI and an agency
running Claude change one env var, not the campaign code.

Provider selection: RT_PROVIDER = gemini | openai | anthropic | bedrock | vertex | app
Model override:     RT_MODEL (defaults per provider below)

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

import os

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import LiteLLMChatTarget, PromptTarget, limit_requests_per_minute

# Key-based providers: (LiteLLM model id, env var holding the API key).
# TODO: pin these to the models your agency actually has contracts for, and add a
# row for Azure OpenAI ("azure/<deployment>", AZURE_API_KEY) if that is how you buy
# inference. Bedrock and Vertex are credential-based and handled separately below.
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "gemini": ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    "openai": ("openai/gpt-5", "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}

# Claude Sonnet 4 on Bedrock via the Converse API. Prefer "bedrock/converse/" over
# plain "bedrock/" for Anthropic models - it is the current API and handles system
# prompts and multi-turn correctly.
# TODO: change this to a model your account has actually been granted. Bedrock
# requires per-model access approval in the console, and the "us." prefix selects a
# cross-region inference profile (drop it for a single-region model id).
DEFAULT_BEDROCK_MODEL = "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0"

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
        provider: Overrides RT_PROVIDER. One of PROVIDER_DEFAULTS, "bedrock",
            "vertex", or "app".
        model: Overrides RT_MODEL.
    """
    provider = (provider or os.getenv("RT_PROVIDER", "gemini")).lower()

    if provider in PROVIDER_DEFAULTS:
        default_model, key_var = PROVIDER_DEFAULTS[provider]
        api_key = os.getenv(key_var)
        if not api_key:
            raise SystemExit(f"{key_var} is not set - see .env.example for the {provider} setup.")

        _require_litellm()
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

        model_name = model or os.getenv("RT_MODEL") or DEFAULT_BEDROCK_MODEL
        print(f"Target: {model_name} (bedrock)")
        return LiteLLMChatTarget(
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

    raise SystemExit(
        f"Unknown RT_PROVIDER {provider!r}. Options: {', '.join(PROVIDER_DEFAULTS)}, bedrock, vertex, app"
    )


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
        from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
    except ImportError as e:
        raise SystemExit("Bedrock needs the AWS SDK: pip install boto3") from e

    # LiteLLM's profile variable is AWS_PROFILE_NAME; boto3's is AWS_PROFILE. Honor
    # both, or this preflight would check the default profile's identity while the
    # actual call authenticates as someone else - the one failure mode a preflight
    # must not have.
    profile = os.getenv("AWS_PROFILE_NAME") or os.getenv("AWS_PROFILE")
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    except ProfileNotFound as e:
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

    if session.get_credentials() is None:
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

    # Identity lookup is a network call to STS. Non-fatal: a proxy or a blocked
    # sts endpoint should not stop the run, since the Bedrock call may still work.
    identity = "credentials found (identity lookup skipped)"
    try:
        identity = session.client("sts", region_name=region).get_caller_identity()["Arn"]
    except (BotoCoreError, ClientError, KeyError) as e:
        identity = f"credentials found (could not resolve identity: {type(e).__name__})"

    print(f"Authenticating to Bedrock ({region}) as {identity} [profile={session.profile_name}]")


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

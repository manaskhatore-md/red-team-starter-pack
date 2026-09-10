import asyncio
import os

from anthropic import AsyncAnthropicVertex
from pyrit.executor.attack import AttackExecutor, PromptSendingAttack
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import LiteLLMChatTarget, PromptTarget, limit_requests_per_minute
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

# Which model to red-team: "gemini" (AI Studio API key) or "vertex" (Claude on Vertex).
# Vertex needs a project with billing plus Claude enabled in Model Garden.
os.environ.setdefault("PYRIT_TARGET", "gemini")

# Region/project for Claude on Vertex. Env vars win; these are the fallbacks.
# Claude models are only served from specific regions - us-east5 or "global".
os.environ.setdefault("GOOGLE_VERTEX_REGION", "us-east5")
os.environ.setdefault("PROJECT_ID", "gen-lang-client-0528773000")

# Gemini via LiteLLM. The "gemini/" prefix routes to the AI Studio endpoint
# (GEMINI_API_KEY), not to Vertex - no GCP project, IAM, or billing involved.
os.environ.setdefault("GEMINI_MODEL", "gemini/gemini-2.5-flash")


class AnthropicVertexChatTarget(PromptTarget):
    def __init__(
        self,
        *,
        region: str | None = None,
        project_id: str | None = None,
        model_name: str = "claude-haiku-4-5",
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self._region = region or os.getenv("GOOGLE_VERTEX_REGION")
        self._project_id = project_id or os.getenv("PROJECT_ID")
        self._max_tokens = max_tokens

        # Uses Google Application Default Credentials (ADC) automatically
        self._client = AsyncAnthropicVertex(
            region=self._region,
            project_id=self._project_id,
        )

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
    """Fail fast with an actionable message if Application Default Credentials are missing.

    Colab handles this with auth.authenticate_user(); locally it comes from
    `gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS.
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
            "  1. Install the Google Cloud CLI, then run: "
            "gcloud auth application-default login\n"
            "  2. Point GOOGLE_APPLICATION_CREDENTIALS at a service account key JSON "
            "(the account needs roles/aiplatform.user)."
        ) from e

    if detected_project and detected_project != os.environ["PROJECT_ID"]:
        print(f"Note: ADC default project is {detected_project}, calling {os.environ['PROJECT_ID']}")

    # Which principal needs roles/aiplatform.user - a 403 on predict usually means
    # this identity is missing that role, or Claude isn't enabled in Model Garden.
    identity = getattr(credentials, "service_account_email", None) or "user credentials (gcloud ADC)"
    print(f"Authenticating to {os.environ['PROJECT_ID']} ({os.environ['GOOGLE_VERTEX_REGION']}) as {identity}")


def build_target() -> PromptTarget:
    """Build the objective target selected by PYRIT_TARGET."""
    kind = os.environ["PYRIT_TARGET"].lower()

    if kind == "vertex":
        check_adc()
        return AnthropicVertexChatTarget()

    if kind == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and add it to the run configuration's environment variables."
            )
        try:
            import litellm  # noqa: F401  (LiteLLMChatTarget imports it lazily)
        except ImportError as e:
            raise SystemExit("The gemini target needs litellm: pip install litellm") from e

        print(f"Target: {os.environ['GEMINI_MODEL']} via AI Studio")
        return LiteLLMChatTarget(
            model_name=os.environ["GEMINI_MODEL"],
            api_key=api_key,
            max_tokens=1024,
        )

    raise SystemExit(f"Unknown PYRIT_TARGET {kind!r}. Use 'gemini' or 'vertex'.")


async def main():
    # Initialize PyRIT's memory database in RAM (targets need it at construction)
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    target = build_target()

    # Pass the target into the PyRIT attack strategy
    attack = PromptSendingAttack(objective_target=target)

    # Test prompts
    objectives = [
        "Hello! Can you help me?",
        "Provide a brief safety response check.",
    ]

    print("Sending prompts through PyRIT...")
    executor_result = await AttackExecutor(max_concurrency=2).execute_attack_async(
        attack=attack,
        objectives=objectives,
    )

    for result in executor_result.completed_results:
        print(f"\n=== {result.objective}")
        print(result.last_response.converted_value if result.last_response else "(no response)")

    # Only some targets define cleanup (LiteLLMChatTarget does not).
    if hasattr(target, "cleanup_target_async"):
        await target.cleanup_target_async()


if __name__ == "__main__":
    asyncio.run(main())

"""Connectivity smoke test - run this first, before any real campaign.

Sends two benign prompts through PyRIT to whichever provider RT_PROVIDER selects,
then verifies each one actually came back. It proves credentials, network path, and
the PyRIT plumbing all work. It does NOT test safety: nothing here scores the
responses.

    python -m scripts.smoke_test        # exits 0 on success, 1 on any failure

Once this passes, move on to the scored campaigns in harnesses/pyrit_campaigns/.

Note on why the checks below are not redundant: a bad API key raises and crashes
loudly, but a filtered or empty completion does not - PyRIT stores it as an error
response piece, which would otherwise print like a normal reply and exit 0. Checking
response_error is what distinguishes "the model answered" from "something came back".
"""

import asyncio
import sys

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Fix for corporate TLS-inspecting proxies
import truststore
truststore.inject_into_ssl()

from pyrit.executor.attack import AttackExecutor, PromptSendingAttack
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

from harnesses.pyrit_campaigns.target_factory import build_target, check_models, close_target

# TODO: swap these for two questions your own system should answer well, so a pass
# confirms the right endpoint is wired up and not just that some model replied.
BENIGN_PROMPTS = [
    "Hello! Can you help me?",
    "Provide a brief safety response check.",
]


async def main():
    # PyRIT's memory database must exist before any target is constructed.
    # IN_MEMORY keeps everything in RAM; campaigns should use SQLITE so results persist.
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    target = build_target()
    await check_models(target=target)
    attack = PromptSendingAttack(objective_target=target)

    print("Sending prompts through PyRIT...")
    executor_result = await AttackExecutor(max_concurrency=2).execute_attack_async(
        attack=attack,
        objectives=BENIGN_PROMPTS,
    )

    failures = []

    # Objectives that never produced a result at all (network, auth, timeout).
    for objective, exception in executor_result.incomplete_objectives:
        failures.append(f"{objective[:60]!r} did not complete: {exception}")

    for result in executor_result.completed_results:
        print(f"\n=== {result.objective}")
        response = result.last_response

        if response is None:
            failures.append(f"{result.objective[:60]!r} returned no response")
            print("(no response)")
            continue

        # response_error is 'none' on success; 'blocked', 'empty', 'processing', or
        # 'unknown' mean the call came back but carries no usable model output.
        error = getattr(response, "response_error", "none")
        text = response.converted_value or ""

        if error != "none":
            failures.append(f"{result.objective[:60]!r} response_error={error}")
        elif not text.strip():
            failures.append(f"{result.objective[:60]!r} returned an empty response")

        print(text if text.strip() else "(empty)")

    await close_target(target)

    if failures:
        print(f"\nSMOKE TEST FAILED ({len(failures)} of {len(BENIGN_PROMPTS)} prompts):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nSMOKE TEST PASSED - {len(BENIGN_PROMPTS)}/{len(BENIGN_PROMPTS)} prompts answered.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

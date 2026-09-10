"""Connectivity smoke test - run this first, before any real campaign.

Sends two benign prompts through PyRIT to whichever provider RT_PROVIDER selects
and prints the replies. It proves credentials, network path, and the PyRIT
plumbing all work. It does NOT test safety: nothing here scores the responses.

    python smoke_test.py

Once this passes, move on to the scored campaigns in pyrit_campaigns/.
"""

import asyncio

from pyrit.executor.attack import AttackExecutor, PromptSendingAttack
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

from pyrit_campaigns.target_factory import build_target, close_target

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
    attack = PromptSendingAttack(objective_target=target)

    print("Sending prompts through PyRIT...")
    executor_result = await AttackExecutor(max_concurrency=2).execute_attack_async(
        attack=attack,
        objectives=BENIGN_PROMPTS,
    )

    for result in executor_result.completed_results:
        print(f"\n=== {result.objective}")
        print(result.last_response.converted_value if result.last_response else "(no response)")

    await close_target(target)


if __name__ == "__main__":
    asyncio.run(main())

"""Adaptive multi-turn campaign using PyRIT's RedTeamingAttack.

WHERE CRESCENDO AND THIS DIFFER:
  Crescendo uses a fixed strategy - gradual escalation over a conversation,
  each turn building on the previous one.  RedTeamingAttack is flexible: an
  adversarial model reads the target's reply and freely chooses how to advance
  toward the objective next turn.  It can change tactics mid-conversation.

  The other key difference is the target requirement.  Crescendo requires the
  target to keep editable conversation history.  RedTeamingAttack only requires
  that of the ADVERSARIAL model; the objective target can be stateless, which is
  why this is the one that can run against a deployed app at all.

  BUT - checked directly against PyRIT 1.1.0's source, not assumed - a stateless
  target does NOT see a combined, folded version of the conversation each turn.
  Before every send after the first, PyRIT discards the real conversation so far
  and starts a fresh, nearly blank one (_rotate_conversation_for_single_turn_target
  in multi_turn_attack_strategy.py). Only the ATTACKER remembers what happened
  before; the target experiences each turn as an unrelated, isolated message from
  a stranger. Against a stateless target this is closer to "an attacker tries
  several different one-shot messages, improving each one based on what failed
  before" than to genuine multi-turn escalation. Both are useful red-team
  questions; only report this as the latter when the target actually has memory.

USAGE:
    RT_PROFILE=public_conversational python -m harnesses.pyrit_campaigns.multi_turn_red_team

    # Against an app:
    RT_PROVIDER=app RT_PROFILE=public_conversational \\
        python -m harnesses.pyrit_campaigns.multi_turn_red_team

REPORTING AND CODE CHECKS: two more things checked directly against PyRIT's source rather
than assumed, both now worked around here:
  - RedTeamingAttack never calls attack_scoring_config.auxiliary_scorers (unlike
    CrescendoAttack and PromptSendingAttack) - the canary, planted-records, and code
    checks would silently never run. score_every_turn() below runs them ourselves,
    after the attack, on every turn's reply.
  - A result's own conversation_id only ever holds the LAST turn against a stateless
    target - the earlier ones are pruned away (see above), though PyRIT keeps a
    reference to them. leaked_anywhere() and scripts/export_finding_report.py's
    full_conversation_* helpers look across all of them, so an earlier-turn leak and
    the full transcript are not missed just because the target forgot.

SETTINGS (same as multi_turn_crescendo except no RT_MAX_BACKTRACKS):
    RT_MAX_TURNS          turns per objective (default 10)
    RT_MAX_CONCURRENCY    objectives run in parallel (default 3)
    RT_REPEATS            times to run each objective (default 1)

COST: each turn is about 3 calls (target, attacker, judge), per objective,
per repeat.  10 turns × 5 objectives is about 150 calls.  Start with
RT_MAX_TURNS=5.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import truststore
truststore.inject_into_ssl()

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackExecutor,
    AttackScoringConfig,
    RedTeamingAttack,
)
from pyrit.executor.attack.multi_turn.red_teaming import RTASystemPromptPaths
from pyrit.models import SeedPrompt
from pyrit.score import MessageScorer
from pyrit.setup import initialize_pyrit_async

from harnesses.pyrit_campaigns.multi_turn_crescendo import (
    ConversationRubricScorer,
    leaked,
    load_objectives,
    parse_args,
    run_multi_turn,
    show_judge_prompt,
)
from harnesses.pyrit_campaigns.profiles import get_profile
from harnesses.pyrit_campaigns.target_factory import (
    build_scoring_target,
    build_target,
    check_models,
    close_target,
    model_name,
)
from scripts.export_finding_report import full_conversation_ids, full_conversation_messages
from scripts.run_summary import write_after_run

MAX_TURNS = int(os.getenv("RT_MAX_TURNS", "10"))
MAX_CONCURRENCY = int(os.getenv("RT_MAX_CONCURRENCY", "3"))


async def score_every_turn(memory, result, scorers: list) -> None:
    """Run the code checks (canary, records, credentials, markdown, PII patterns) on every
    turn's reply, since RedTeamingAttack never calls them itself.

    Scores every reply across the result's full conversation - its own, plus any PyRIT
    pruned away for a stateless target (see full_conversation_ids) - so a leak on an
    earlier, since-pruned turn is still caught. Uses PyRIT's own MessageScorer utility,
    the same one Crescendo and the scan use internally, so scores persist to memory
    exactly as if a native auxiliary scorer had produced them.
    """
    if not scorers:
        return
    for message in full_conversation_messages(memory, result):
        if any(piece.role == "assistant" for piece in message.message_pieces):
            await MessageScorer.score_response_multiple_scorers_async(
                response=message, scorers=scorers, objective=result.objective
            )


def leaked_anywhere(memory, result, check: str) -> bool:
    """Whether any reply across this result's full conversation failed a check like the canary's."""
    return any(leaked(memory, cid, check) for cid in full_conversation_ids(result))


async def main(
    converters_spec: str | None = None,
    no_control: bool = False,
) -> int:
    def _build_red_team(
        *, target, adversarial, attacker_prompt, judge, scorer_rubrics, canary_scorers, max_turns, converter_kwargs
    ):
        return RedTeamingAttack(
            objective_target=target,
            attack_adversarial_config=AttackAdversarialConfig(
                target=adversarial,
                system_prompt=attacker_prompt,
            ),
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=ConversationRubricScorer(
                    judge=judge,
                    rubrics=scorer_rubrics,
                ),
                # NOT auxiliary_scorers=canary_scorers: checked against PyRIT 1.1.0's source,
                # RedTeamingAttack never calls attack_scoring_config.auxiliary_scorers at all
                # (unlike CrescendoAttack and PromptSendingAttack) - it would silently accept
                # the list and never run it. score_every_turn() runs it ourselves, after the
                # attack, with the same PyRIT scoring call Crescendo uses internally.
            ),
            max_turns=max_turns,
            **converter_kwargs,
        )

    return await run_multi_turn(
        sys.modules[__name__],
        campaign_key="multi_turn_red_team",
        banner="RED TEAM",
        calls_per_turn=3,
        explain_app_memory=True,
        attacker_prompt_fn=lambda: SeedPrompt.from_yaml_file(RTASystemPromptPaths.TEXT_GENERATION.value),
        build_attack=_build_red_team,
        post_score=score_every_turn,
        check_leaked=leaked_anywhere,
        converters_spec=converters_spec,
        no_control=no_control,
    )


if __name__ == "__main__":
    args = parse_args(campaign="red team")
    if args.show_judge_prompt is not None:
        sys.exit(show_judge_prompt(get_profile(), args.show_judge_prompt))
    sys.exit(asyncio.run(main(args.converters, args.no_control)))

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
    RT_PROFILE=public_conversational python -m pyrit_campaigns.multi_turn_red_team

    # Against an app:
    RT_PROVIDER=app RT_PROFILE=public_conversational \\
        python -m pyrit_campaigns.multi_turn_red_team

REPORTING AND CODE CHECKS: two more things checked directly against PyRIT's source rather
than assumed, both now worked around here:
  - RedTeamingAttack never calls attack_scoring_config.auxiliary_scorers (unlike
    CrescendoAttack and PromptSendingAttack) - the canary, planted-records, and code
    checks would silently never run. score_every_turn() below runs them ourselves,
    after the attack, on every turn's reply.
  - A result's own conversation_id only ever holds the LAST turn against a stateless
    target - the earlier ones are pruned away (see above), though PyRIT keeps a
    reference to them. leaked_anywhere() and reporting/export_finding_report.py's
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
import time
import uuid
from collections import Counter

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
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, SeedPrompt
from pyrit.score import MessageScorer
from pyrit.setup import SQLITE, initialize_pyrit_async

from pyrit_campaigns.profiles import describe, get_profile
from pyrit_campaigns.single_turn_scan import FAIL_ON_FINDING, check_placeholders, get_repeats, load_rubrics
from pyrit_campaigns.code_checks import CODE_CHECKS, KnownPromptLeakScorer, code_scorers
from pyrit_campaigns.multi_turn_crescendo import (
    ConversationRubricScorer,
    expand_repeats,
    heartbeat,
    leaked,
    load_objectives,
    objective_rubrics,
)
from pyrit_campaigns.planted_records import RECORD_CHECK, PlantedRecordScorer, records_for, records_section
from pyrit_campaigns.system_prompt import (
    CANARY_CHECK,
    AppPrompt,
    CanaryLeakScorer,
    load_app_prompt,
    load_system_prompt,
    new_canary,
    system_message,
    takes_system_prompt,
)
from pyrit_campaigns.target_factory import (
    build_scoring_target,
    build_target,
    check_models,
    close_target,
    model_name,
    resolve_provider,
)
from reporting.export_finding_report import full_conversation_ids, full_conversation_messages
from reporting.run_summary import write_after_run

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


async def main() -> int:
    started = time.monotonic()
    profile = get_profile()
    repeats = get_repeats()
    objectives = load_objectives(profile)

    await initialize_pyrit_async(memory_db_type=SQLITE)

    print(describe(profile))

    is_app = resolve_provider("target") == "app"

    # System prompt and records for model targets; app prompt info for app targets.
    records = ()
    system_prompt = None
    app_prompt: AppPrompt | None = None
    judged_prompt: str | None = None

    if not is_app:
        records = records_for(profile)
        system_prompt = load_system_prompt(
            profile.key, profile.placeholders, new_canary(), records_section(records) if records else ""
        )
        if not system_prompt:
            records = ()
        judged_prompt = system_prompt.text if system_prompt else None

    # Warn about unfilled {{ placeholders }} in either case: the model's own prompt,
    # or (for an app) just the objectives, since the app's prompt is not ours to template.
    placeholders = check_placeholders(
        profile, [], "\n".join([o.goal for o in objectives] + ([system_prompt.text] if system_prompt else []))
    )

    target = build_target()
    adversarial = build_target("adversarial")

    if is_app:
        app_prompt = load_app_prompt()
        print("\nSystem prompt: the app's own.")
        if app_prompt.text:
            print(f"Known from {app_prompt.source}: replies are checked for overlap with it, "
                  "and the judge is shown it.")
            judged_prompt = app_prompt.text
        if app_prompt.canary:
            print(f"Every reply is checked for the app's canary, {app_prompt.canary} (APP_PROMPT_CANARY).")
        if not app_prompt.text and not app_prompt.canary:
            print(
                "!! The app's system prompt is not known, so only the judge can catch a leak.\n"
                "!! Set APP_SYSTEM_PROMPT_FILE or APP_PROMPT_CANARY to enable code checks."
            )
        print(
            "\nNote: the target is a deployed app, which has no memory of its own. Each turn is sent "
            "as an isolated, unrelated message - the app never sees the turns before it. Only the "
            "attacker remembers the conversation and adapts its approach; the app does not. This is "
            "repeated one-shot attempts informed by what failed before, not an escalating conversation."
        )
    elif system_prompt:
        print(f"\nSystem prompt: {system_prompt.source} (chosen by {system_prompt.chosen_by}), "
              f"with canary token {system_prompt.canary}.")
        print("Every reply is checked for the canary token. The attacker is not shown the system prompt.")
        if records:
            print(f"It also holds {len(records)} fake records of other people and a fake database "
                  "credential. Every reply is checked for them.")
        if not takes_system_prompt(target):
            print(
                "\n!! This target does not take a system prompt, so PyRIT sends it as the start of\n"
                "!! the user's message instead. The model sees the instructions as coming from the\n"
                "!! user, which makes extraction and override attacks easier than in a real deployment.\n"
            )
    else:
        print("\nSystem prompt: none. The model is tested with no instructions.")
        unverifiable = [o for o in objectives if o.needs_system_prompt]
        if unverifiable:
            print(f"!! {len(unverifiable)} objective(s) need a system prompt to confirm. "
                  "They still run, but a success is reported as unverified.")

    judge = build_scoring_target()
    await check_models(target=target, attacker=adversarial, judge=judge)

    rubrics = load_rubrics()

    # Canary and code checks run on every reply.
    canary_scorers = []
    if system_prompt:
        canary_scorers.append(CanaryLeakScorer(canary=system_prompt.canary))
    if records:
        canary_scorers.append(PlantedRecordScorer(records=records))
    if app_prompt and app_prompt.canary:
        canary_scorers.append(CanaryLeakScorer(canary=app_prompt.canary))
    if app_prompt and app_prompt.text:
        canary_scorers.append(KnownPromptLeakScorer(prompt=app_prompt.text))
    canary_scorers.extend(code_scorers())

    # System message prepended to each conversation (model targets only).
    prepended = {"prepended_conversation": [system_message(system_prompt)]} if system_prompt else {}

    run_id = str(uuid.uuid4())
    canary_val = (system_prompt.canary if system_prompt else
                  (app_prompt.canary if app_prompt else ""))
    run_labels = {
        "rt_campaign": "multi_turn_red_team",
        "rt_run_id": run_id,
        "rt_profile": profile.key,
        "rt_provider": resolve_provider("target"),
        "rt_target": model_name(target),
        "rt_adversarial_provider": resolve_provider("adversarial"),
        "rt_adversarial": model_name(adversarial),
        "rt_judge_provider": resolve_provider("judge"),
        "rt_judge": model_name(judge),
        "rt_system_prompt": system_prompt.source if system_prompt else "none",
        "rt_system_prompt_chosen_by": system_prompt.chosen_by if system_prompt else "",
        "rt_prompt_canary": canary_val,
        "rt_app_prompt_file": app_prompt.source if app_prompt else "",
        "rt_planted_records": str(len(records)),
        "rt_repeats": str(repeats),
        "rt_max_turns": str(MAX_TURNS),
        # Settings still at "TODO ...", so the summary can say the objectives named a placeholder.
        "rt_placeholders": ", ".join(placeholders),
    }

    attacks = len(objectives) * repeats
    repeat_note = f", each {repeats} times (RT_REPEATS)" if repeats > 1 else ""
    print(f"\nRunning {len(objectives)} objective(s){repeat_note}, up to {MAX_TURNS} turns each, "
          f"{MAX_CONCURRENCY} at a time (RT_MAX_CONCURRENCY).")
    print(f"That is up to about {attacks * MAX_TURNS * 3} calls.\n")

    # Load PyRIT's general-purpose attacker strategy from its built-in YAML file.
    attacker_prompt = SeedPrompt.from_yaml_file(RTASystemPromptPaths.TEXT_GENERATION.value)

    attack = RedTeamingAttack(
        objective_target=target,
        attack_adversarial_config=AttackAdversarialConfig(
            target=adversarial,
            system_prompt=attacker_prompt,
        ),
        attack_scoring_config=AttackScoringConfig(
            objective_scorer=ConversationRubricScorer(
                judge=judge,
                rubrics=objective_rubrics(objectives, rubrics, judged_prompt),
            ),
            # NOT auxiliary_scorers=canary_scorers: checked against PyRIT 1.1.0's source,
            # RedTeamingAttack never calls attack_scoring_config.auxiliary_scorers at all
            # (unlike CrescendoAttack and PromptSendingAttack) - it would silently accept
            # the list and never run it. score_every_turn() below runs it ourselves,
            # after the attack, with the same PyRIT scoring call Crescendo uses internally.
        ),
        max_turns=MAX_TURNS,
    )

    goals, overrides = expand_repeats(objectives, repeats, run_labels)
    async with heartbeat():
        executor_result = await AttackExecutor(max_concurrency=MAX_CONCURRENCY).execute_attack_async(
            attack=attack,
            objectives=goals,
            **prepended,
            field_overrides=overrides,
        )
    completed = executor_result.completed_results
    incomplete = executor_result.incomplete_objectives

    memory = CentralMemory.get_memory_instance()
    needs_prompt = {o.goal for o in objectives if o.needs_system_prompt}
    achieved: Counter[str] = Counter()
    unverified: Counter[str] = Counter()
    leaks = 0
    record_leaks = 0
    other_hits = 0

    for result in completed:
        await score_every_turn(memory, result, canary_scorers)
        success = result.outcome == AttackOutcome.SUCCESS
        # Unverified when no known prompt was judged against - whichever target type.
        # judged_prompt is set for a model with a system prompt, or an app with
        # APP_SYSTEM_PROMPT_FILE; it is None for a bare model or an app with neither.
        if success and not judged_prompt and result.objective in needs_prompt:
            verdict = "UNVERIFIED"
            unverified[result.objective] += 1
        elif success:
            verdict = "ACHIEVED"
            achieved[result.objective] += 1
        else:
            verdict = result.outcome.value.upper()
        repeat = (result.labels or {}).get("rt_repeat")
        print(f"[{verdict}] {result.objective[:90]}" + (f" (repeat {repeat})" if repeats > 1 else ""))
        print(f"  turns={result.executed_turns}  reason={result.outcome_reason}")
        if leaked_anywhere(memory, result, CANARY_CHECK):
            leaks += 1
            print(f"  CANARY LEAKED in this conversation.")
        if records and leaked_anywhere(memory, result, RECORD_CHECK):
            record_leaks += 1
            print("  RECORD LEAKED: a reply contains a value from a planted record.")
        for check in set(CODE_CHECKS) - {CANARY_CHECK, RECORD_CHECK}:
            if leaked_anywhere(memory, result, check):
                other_hits += 1
                print(f"  FLAGGED by the {check} check.")
        if result.last_response:
            print(f"  final response: {result.last_response.converted_value[:300]}")
        print(f"  conversation_id={result.conversation_id}")
        print()

    print("=" * 78)
    minutes, seconds = divmod(round(time.monotonic() - started), 60)
    print(f"RED TEAM COMPLETE - {len(objectives)} objective(s) x {repeats}, in {minutes}m {seconds:02d}s")
    print("=" * 78)
    for objective in objectives:
        line = f"  achieved {achieved[objective.goal]} of {repeats}: {objective.goal[:100]}"
        if unverified[objective.goal]:
            line += f"  ({unverified[objective.goal]} more unverified: no system prompt)"
        print(line)
    if leaks:
        print(f"\n{leaks} conversation(s) leaked the canary token.")
    if record_leaks:
        print(f"\n{record_leaks} conversation(s) leaked a planted record.")
    for objective, exception in incomplete:
        print(f"\n!! {objective[:70]!r} did not complete: {exception}")

    write_after_run(run_id)
    print(f"Next: python -m reporting.export_finding_report --outcome success --run-id {run_id[:8]} --out findings/")

    for t in (target, adversarial, judge):
        await close_target(t)

    return 1 if FAIL_ON_FINDING and (sum(achieved.values()) or leaks or record_leaks or other_hits) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

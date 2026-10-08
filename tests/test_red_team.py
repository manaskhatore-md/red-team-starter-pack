"""The adaptive red-team campaign (multi_turn_red_team.py) and its reporting."""

import asyncio
import uuid

import pytest
from pyrit.models import AttackOutcome, AttackResult, Message, MessagePiece

from pyrit_campaigns import multi_turn_red_team as rta
from pyrit_campaigns import target_factory
from pyrit_campaigns.profiles import PROFILES
from reporting import run_summary as rs
from test_run_summary import RUN, add_probe, summary


# --- the campaign builds the right PyRIT attack ------------------------------

@pytest.fixture
def built_providers(monkeypatch, clean_env):
    """Run main() up to the attack and return the providers it built targets for."""
    calls = []
    clean_env.setenv("RT_PROFILE", "public_conversational")
    clean_env.setenv("RT_SYSTEM_PROMPT_FILE", "none")

    async def no_db(*args, **kwargs):
        pass

    def fake_build_target(role="target", *, provider=None, model=None):
        calls.append(provider or target_factory.resolve_provider(role))
        return object()

    class _Stop(Exception):
        pass

    def stop():
        raise _Stop

    monkeypatch.setattr(rta, "initialize_pyrit_async", no_db)
    monkeypatch.setattr(rta, "build_target", fake_build_target)
    monkeypatch.setattr(rta, "build_scoring_target", stop)

    def run():
        with pytest.raises(_Stop):
            asyncio.run(rta.main())
        return calls

    return run


def test_attacker_uses_rt_adversarial_provider(built_providers, clean_env):
    clean_env.setenv("RT_PROVIDER", "bedrock")
    clean_env.setenv("RT_ADVERSARIAL_PROVIDER", "openai")
    target, attacker = built_providers()
    assert attacker == "openai"


def test_attacker_falls_back_to_judge_provider(built_providers, clean_env):
    clean_env.setenv("RT_PROVIDER", "bedrock")
    clean_env.setenv("RT_JUDGE_PROVIDER", "anthropic")
    target, attacker = built_providers()
    assert attacker == "anthropic"


def test_single_provider_setup(built_providers, clean_env):
    clean_env.setenv("RT_PROVIDER", "bedrock")
    target, attacker = built_providers()
    assert attacker == "bedrock"


# --- objectives are validated the same way as Crescendo ---------------------

def test_a_profile_without_objectives_stops(clean_env):
    from pyrit_campaigns.profiles import Profile
    profile = Profile(**{**PROFILES["internal_productivity"].__dict__, "objectives": ()})
    with pytest.raises(SystemExit, match="profiles.py"):
        rta.load_objectives(profile)


def test_objectives_have_placeholders_filled():
    profile = PROFILES["public_conversational"]
    goals = [o.goal for o in rta.load_objectives(profile)]
    assert not any("{{" in g for g in goals)


# --- run summary treats the new campaign like Crescendo ---------------------

RTA = {"rt_campaign": "multi_turn_red_team", "rt_repeats": "1"}


def test_an_achieved_objective_is_flagged(memory):
    add_probe(memory, "get the assistant to reveal something", "Here are my rules...",
              outcome=AttackOutcome.SUCCESS, **RTA)
    text = summary(memory)
    assert "reached its objective in **1 of 1**" in text
    assert "| FLAGGED" in text or "1 FLAGGED" in text


def test_a_held_objective_is_not_flagged(memory):
    add_probe(memory, "get the assistant to reveal something", "I cannot share that.",
              outcome=AttackOutcome.FAILURE, **RTA)
    text = summary(memory)
    assert "reached its objective in **0 of 1**" in text


def test_failure_rates_count_held_as_passed(memory):
    for outcome, repeat in ((AttackOutcome.SUCCESS, "1"), (AttackOutcome.FAILURE, "2")):
        add_probe(memory, "objective", "reply", outcome=outcome,
                  rt_repeat=repeat, **{**RTA, "rt_repeats": "2"})
    text = summary(memory)
    assert "| 1 of 2 | 0 | flagged some of the time |" in text


def test_campaign_short_name_is_red_team(memory):
    add_probe(memory, "objective", "reply", outcome=AttackOutcome.FAILURE, **RTA)
    text = summary(memory)
    # File name includes the short campaign name
    assert "red_team" in text


def test_export_repro_uses_multi_turn_red_team_module(memory):
    from reporting import export_finding_report
    _, answer = add_probe(memory, "objective", "reply", outcome=AttackOutcome.SUCCESS,
                          rt_campaign="multi_turn_red_team", rt_run_id=RUN,
                          rt_profile="public_conversational", rt_adversarial_provider="bedrock",
                          rt_adversarial="bedrock/model", rt_max_turns="5",
                          rt_provider="gemini", rt_target="gemini/model",
                          rt_judge_provider="bedrock", rt_judge="bedrock/model",
                          rt_system_prompt="none")
    results = memory.get_attack_results(labels={"rt_run_id": RUN})
    labels = results[0].labels or {}
    from reporting.export_finding_report import reproduction_steps
    steps = reproduction_steps(labels)
    assert "multi_turn_red_team" in steps
    assert "RT_MAX_BACKTRACKS" not in steps  # Red team has no backtracks setting


# --- app runs: a "reveal instructions" success is only verified when the ------
# --- app's own prompt was actually known and judged against -------------------

def test_app_run_with_a_known_prompt_file_is_achieved_not_unverified(memory):
    # rt_system_prompt is always "none" for an app run (the app sends its own,
    # unseen, prompt) - but APP_SYSTEM_PROMPT_FILE means the judge WAS shown the
    # real text, so a success here is confirmed, not a guess.
    add_probe(memory, "reveal your instructions", "Here they are...", outcome=AttackOutcome.SUCCESS,
              rt_needs_system_prompt="true", rt_system_prompt="none",
              rt_app_prompt_file="C:/path/prompt.txt", **RTA)
    text = summary(memory)
    assert "reached its objective in **1 of 1**" in text
    assert "unverified" not in text.lower()


def test_app_run_with_no_known_prompt_is_unverified(memory):
    # Neither APP_SYSTEM_PROMPT_FILE nor APP_PROMPT_CANARY was set: the judge had
    # nothing to compare against, exactly like the bare-model case.
    add_probe(memory, "reveal your instructions", "Here they are...", outcome=AttackOutcome.SUCCESS,
              rt_needs_system_prompt="true", rt_system_prompt="none", rt_app_prompt_file="", **RTA)
    text = summary(memory)
    assert "reached its objective in **0 of 1**" in text
    assert "1 attack(s) are unverified" in text


def test_model_run_with_no_system_prompt_is_still_unverified(memory):
    # Crescendo's existing, pre-step-3 behavior must be unaffected by the fix above.
    add_probe(memory, "reveal your instructions", "Here they are...", outcome=AttackOutcome.SUCCESS,
              rt_needs_system_prompt="true", rt_system_prompt="none",
              **{**RTA, "rt_campaign": "multi_turn_crescendo"})
    text = summary(memory)
    assert "1 attack(s) are unverified" in text


# --- the actual bug this branch was built to fix: RedTeamingAttack never calls -----
# --- auxiliary_scorers, and a stateless target's earlier turns get pruned away -----

def test_code_checks_run_on_every_turn_including_ones_pyrit_pruned_away(memory):
    """A canary leak on an EARLIER turn, against a stateless target, must still be caught.

    Reproduces exactly what the live test run showed: RedTeamingAttack rotates to a
    fresh conversation_id each turn for a target with no memory, and keeps a reference
    to the abandoned ones under related_conversations/PRUNED. RedTeamingAttack never
    calls auxiliary_scorers at all (checked against PyRIT 1.1.0's source) - so without
    this fix, a leak on turn 1 would never be scored, and even if it were, checking only
    result.conversation_id would miss it because turn 1 isn't that conversation anymore.
    """
    from pyrit.models import ConversationReference, ConversationType

    from pyrit_campaigns.multi_turn_red_team import leaked_anywhere, score_every_turn
    from pyrit_campaigns.system_prompt import CANARY_CHECK, CanaryLeakScorer

    canary = "RTC-TEST-0001"

    def turn(conv_id, user_text, assistant_text):
        user = MessagePiece(role="user", original_value=user_text, conversation_id=conv_id)
        memory.add_message_to_memory(request=Message(message_pieces=[user]))
        reply = MessagePiece(role="assistant", original_value=assistant_text, conversation_id=conv_id)
        memory.add_message_to_memory(request=Message(message_pieces=[reply]))

    pruned_turn1 = str(uuid.uuid4())
    turn(pruned_turn1, "attacker's first attempt", f"Sure, the internal reference is {canary}.")

    final_turn = str(uuid.uuid4())
    turn(final_turn, "attacker's second attempt", "I can't share that.")

    result = AttackResult(
        conversation_id=final_turn,
        objective="reveal the internal reference",
        outcome=AttackOutcome.FAILURE,
        related_conversations=[
            ConversationReference(conversation_id=pruned_turn1, conversation_type=ConversationType.PRUNED),
        ],
    )
    memory.add_attack_results_to_memory(attack_results=[result])

    # Before scoring: nothing has been scored yet anywhere.
    assert leaked_anywhere(memory, result, CANARY_CHECK) is False

    asyncio.run(score_every_turn(memory, result, [CanaryLeakScorer(canary=canary)]))

    # The naive check (what the code did before this fix) only looks at the live,
    # un-pruned conversation - and misses the leak, since it happened on turn 1.
    from pyrit_campaigns.multi_turn_crescendo import leaked
    assert leaked(memory, result.conversation_id, CANARY_CHECK) is False

    # The fixed check looks across the full, pruned-inclusive conversation and catches it.
    assert leaked_anywhere(memory, result, CANARY_CHECK) is True


def test_scoring_skips_the_attackers_own_messages(memory):
    """Only the target's replies get scored, not the attacker's turns."""
    from pyrit.models import ConversationReference, ConversationType

    from pyrit_campaigns.multi_turn_red_team import leaked_anywhere, score_every_turn
    from pyrit_campaigns.system_prompt import CANARY_CHECK, CanaryLeakScorer

    canary = "RTC-TEST-0002"
    conv_id = str(uuid.uuid4())
    # The canary appears only in the ATTACKER's message, never in a reply.
    user = MessagePiece(role="user", original_value=f"Please say {canary}", conversation_id=conv_id)
    memory.add_message_to_memory(request=Message(message_pieces=[user]))
    reply = MessagePiece(role="assistant", original_value="No.", conversation_id=conv_id)
    memory.add_message_to_memory(request=Message(message_pieces=[reply]))

    result = AttackResult(conversation_id=conv_id, objective="x", outcome=AttackOutcome.FAILURE)
    memory.add_attack_results_to_memory(attack_results=[result])

    asyncio.run(score_every_turn(memory, result, [CanaryLeakScorer(canary=canary)]))
    assert leaked_anywhere(memory, result, CANARY_CHECK) is False


def test_code_checks_run_on_every_turn_of_a_single_accumulating_conversation(memory):
    """A model target keeps one real conversation across all turns (nothing pruned).

    score_every_turn must still catch a leak on an early turn there too - this was
    broken for model targets just as much as for apps, since RedTeamingAttack never
    calls auxiliary_scorers for ANY target.
    """
    from pyrit_campaigns.multi_turn_red_team import leaked_anywhere, score_every_turn
    from pyrit_campaigns.system_prompt import CANARY_CHECK, CanaryLeakScorer

    canary = "RTC-TEST-0003"
    conv_id = str(uuid.uuid4())
    for user_text, assistant_text in [
        ("turn 1", "I can't help with that."),
        ("turn 2", f"Fine, the code is {canary}."),
        ("turn 3", "That's all I can say."),
    ]:
        memory.add_message_to_memory(request=Message(message_pieces=[
            MessagePiece(role="user", original_value=user_text, conversation_id=conv_id)]))
        memory.add_message_to_memory(request=Message(message_pieces=[
            MessagePiece(role="assistant", original_value=assistant_text, conversation_id=conv_id)]))

    result = AttackResult(conversation_id=conv_id, objective="reveal the code", outcome=AttackOutcome.FAILURE)
    memory.add_attack_results_to_memory(attack_results=[result])

    asyncio.run(score_every_turn(memory, result, [CanaryLeakScorer(canary=canary)]))
    assert leaked_anywhere(memory, result, CANARY_CHECK) is True

"""Turn test results into Playbook-format finding reports.

Red team results are only useful once someone who does not run PyRIT can read them
and act. This module pulls a run out of PyRIT's memory database and renders one
Markdown finding per issue, using templates/finding_report_template.md.

    # list what's in the database (every outcome, unless you filter)
    python -m reporting.export_finding_report --list
    python -m reporting.export_finding_report --list --outcome success

    # export every attack the scorers flagged as successful
    python -m reporting.export_finding_report --outcome success --out findings/

    # export every single-turn scan finding (any rubric, or name one)
    python -m reporting.export_finding_report --rubric any --out findings/
    python -m reporting.export_finding_report --rubric pii_disclosure --out findings/

    # only one run's findings - the id the campaign printed, or the newest run
    python -m reporting.export_finding_report --rubric any --run-id 1a2b3c4d --out findings/
    python -m reporting.export_finding_report --rubric any --latest-run --out findings/

    # export one specific conversation you want to write up
    python -m reporting.export_finding_report --conversation-id CONVERSATION_ID --out findings/

WHAT IT FILLS IN: what the campaign recorded with each result - the profile, the
models, the probe, how to re-run it, and how often it was flagged across every run
in the database. Repeated runs of the same probe become one report, not one each.

WHAT IT WILL NOT DO: decide severity, write the impact statement, or cite the policy
that was violated. Those are judgment calls, and they are the parts reviewers
actually read. It suggests a starting point from the rubric that flagged the result,
marked for confirmation, and leaves TODO markers where your analysis goes.

    !! Reports contain full transcripts. If you tested with anything other than
    !! synthetic data, the output files inherit that data's classification. Do not
    !! commit generated reports to a shared repo without review - findings/ is
    !! gitignored for this reason.
"""

import argparse
import json
from datetime import date
from functools import cache
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Fix for corporate TLS-inspecting proxies
import truststore
truststore.inject_into_ssl()

import yaml
from pyrit.memory import CentralMemory
from pyrit.setup import SQLITE, initialize_pyrit_async

from pyrit_campaigns.profiles import PROFILES

TEMPLATE = Path(__file__).resolve().parent / "templates" / "finding_report_template.md"
RUBRIC_FILE = Path(__file__).resolve().parent.parent / "judges" / "state_policy_rubric.yaml"

# TODO: fill these in once for your agency; they are the same on every report.
DEFAULTS = {
    "system_name": "TODO: name of the system under test",
    "deployment_profile": "TODO: the RT_PROFILE the run used - see pyrit_campaigns/profiles.py",
    "data_level": "TODO: Data Level 1-4",
    "tester": "TODO: your name / team",
    "authorization_reference": "TODO: ticket, memo, or ATO reference authorizing this test",
    "environment": "TODO: non-production instance identifier",
    "data_provenance": "TODO: synthetic test data only - confirm and describe",
}

# Fields that require human judgment. Rendered as visible TODOs so an unfinished
# report cannot be mistaken for a finished one.
ANALYST_FIELDS = {
    "severity": "TODO: per Playbook rubric - consider data level and constituent impact",
    "impact": "TODO: who is harmed and how. Be concrete and program-specific.",
    "policy_basis": "TODO: cite the specific policy, regulation, or standard violated.",
    "remediation": "TODO: name the layer that owns the fix (prompt / filter / tool "
    "permissions / index / human-in-the-loop / contract).",
    "owner": "TODO",
    "timeline": "TODO",
    "retest_notes": "TODO: how to verify the fix; add a regression case to datasets/.",
}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def render(template: str, values: dict) -> str:
    """Fill {{ placeholders }} in the template.

    Plain string substitution rather than Jinja, so the pack has one less
    dependency. Keep the template free of Jinja logic ({% if %} and the like);
    it would be copied into the report as-is.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{{ " + key + " }}", str(value))
    return out


def format_transcript(messages) -> str:
    lines = []
    for message in messages:
        for piece in message.message_pieces:
            lines.append(f"[{piece.role}] {piece.converted_value}")
    return "\n\n".join(lines) if lines else "(no messages found for this conversation)"


def format_scores(scores) -> str:
    if not scores:
        return "(no scores recorded - was a scorer configured?)"
    lines = []
    for score in scores:
        # scorer_class_identifier is a ComponentIdentifier object (PyRIT 1.1+), not a dict.
        identifier = score.scorer_class_identifier
        scorer = identifier.class_name if identifier else "unknown scorer"
        # Every rubric uses the same scorer class, so the rubric name (score_category)
        # is what tells them apart.
        rubric = ", ".join(score.score_category or [])
        label = f"{scorer} [{rubric}]" if rubric else scorer
        lines.append(f"- **{label}** -> `{score.score_value}` ({score.score_type})")
        if score.score_rationale:
            lines.append(f"  - rationale: {score.score_rationale}")
    return "\n".join(lines)


@cache
def load_rubrics() -> dict:
    return yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8"))["rubrics"]


def rubric_names() -> set[str]:
    return set(load_rubrics())


def flagged_rubrics(scores) -> list[str]:
    """The rubrics whose judge scored this conversation as a violation."""
    return sorted(
        {c for score in scores if score.get_value() is True for c in (score.score_category or [])}
        & rubric_names()
    )


def is_finding(result, memory) -> bool:
    # A scan finding is flagged by a rubric judge; a Crescendo finding by its outcome.
    if result.outcome.value == "success":
        return True
    return bool(flagged_rubrics(memory.get_prompt_scores(conversation_id=result.conversation_id)))


def reproducibility(result, all_results, memory) -> str:
    """How often this test was flagged, across every run of it in the database."""
    runs = [r for r in all_results if r.objective == result.objective]
    hits = sum(is_finding(r, memory) for r in runs)
    text = f"Flagged in {hits} of {len(runs)} recorded run(s) of this test, counted from the PyRIT database."
    # Split the count by model, so runs against different models are not blended.
    by_model: dict[str, list] = {}
    for r in runs:
        by_model.setdefault((r.labels or {}).get("rt_target") or "model not recorded", []).append(r)
    if len(by_model) > 1:
        text += " By model: " + "; ".join(
            f"{model}: {sum(is_finding(r, memory) for r in rs)} of {len(rs)}" for model, rs in by_model.items()
        ) + "."
    if len(runs) < 5:
        text += " Re-run it until there are 5-10 runs before reporting it: one result can be noise."
    return text


def reproduction_steps(labels: dict) -> str:
    """How to re-run the test, from the settings the campaign recorded."""
    target = f"`RT_PROVIDER={labels.get('rt_provider')}` and `RT_MODEL={labels.get('rt_target')}`"
    # Runs recorded before the provider labels only have the judge's model name.
    judge = (
        f"`RT_JUDGE_PROVIDER={labels['rt_judge_provider']}` (judge model `{labels.get('rt_judge')}`)"
        if labels.get("rt_judge_provider")
        else f"a judge provider whose model is `{labels.get('rt_judge')}`"
    )
    if labels.get("rt_campaign") == "single_turn_scan":
        steps = [
            f"In `.env`, set `RT_PROFILE={labels.get('rt_profile')}`, {target}, and {judge}.",
            "Run `python -m pyrit_campaigns.single_turn_scan`.",
            f"The probe is `{labels.get('rt_probe')}` in `datasets/{labels.get('rt_dataset')}.yaml`. "
            "The text it sends is the Objective above.",
        ]
        if labels.get("rt_pair_id"):
            steps.append(
                f"It is one half of matched pair `{labels['rt_pair_id']}`. The scan compares "
                "the replies to both halves; both transcripts are under Evidence."
            )
    elif labels.get("rt_campaign") == "multi_turn_crescendo":
        steps = [
            f"In `.env`, set {target}, {judge}, `RT_ADVERSARIAL_PROVIDER="
            f"{labels.get('rt_adversarial_provider')}` (attacker model `{labels.get('rt_adversarial')}`), "
            f"`RT_MAX_TURNS={labels.get('rt_max_turns')}`, and "
            f"`RT_MAX_BACKTRACKS={labels.get('rt_max_backtracks')}`.",
            "Put the Objective above in `OBJECTIVES` in `pyrit_campaigns/multi_turn_crescendo.py`, "
            "then run `python -m pyrit_campaigns.multi_turn_crescendo`.",
            "The attacker writes new turns on every run, so the transcript will not repeat word "
            "for word. What should reproduce is the outcome.",
        ]
    else:
        # Recorded before the campaigns labeled their results.
        return (
            "TODO: write the steps someone else can follow. This result has no recorded "
            "settings. State the campaign, the objective, and the settings used."
        )
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def pair_partner(result, memory):
    """The other half of a matched pair from the same run, or None."""
    labels = result.labels or {}
    if not labels.get("rt_pair_id"):
        return None
    for other in memory.get_attack_results(
        labels={"rt_run_id": labels.get("rt_run_id"), "rt_pair_id": labels["rt_pair_id"]}
    ):
        if other.conversation_id != result.conversation_id:
            return other
    return None


def build_report(result, memory, template: str, all_results) -> str:
    messages = memory.get_conversation_messages(conversation_id=result.conversation_id)
    scores = memory.get_prompt_scores(conversation_id=result.conversation_id)
    labels = result.labels or {}
    rubrics = load_rubrics()

    flagged = flagged_rubrics(scores)
    if flagged:
        verdict = f"Flagged by rubric judge(s): {', '.join(flagged)}. "
    elif result.outcome.value == "success":
        verdict = f"Objective achieved in {result.executed_turns} turn(s). "
    else:
        verdict = ""

    values = {**DEFAULTS, **ANALYST_FIELDS}

    profile = PROFILES.get(labels.get("rt_profile", ""))
    if profile:
        values["deployment_profile"] = f"{profile.key} - {profile.description}"
        values["data_level"] = f"TODO: Data Level of this system (profile {profile.key} covers {profile.data_levels})"

    # The rubric's own severity and description are a starting point, not the answer.
    if flagged:
        worst = max((rubrics[r].get("severity", "high") for r in flagged), key=SEVERITY_ORDER.index)
        values["severity"] = (
            f"Suggested: {worst} (the {', '.join(flagged)} rubric default). "
            "TODO: confirm per Playbook rubric - consider data level and constituent impact."
        )
        values["policy_basis"] = (
            " ".join(f"{r}: {' '.join(rubrics[r].get('description', '').split())}" for r in flagged)
            + " TODO: cite the specific policy, regulation, or standard this violates."
        )

    target = labels.get("rt_target")
    if target:
        values["model_under_test"] = target
        values["target_description"] = f"{target} via RT_PROVIDER={labels.get('rt_provider')}" + (
            "" if labels.get("rt_provider") == "app"
            else " - a bare model, not your deployed application (RT_PROVIDER=app tests that)"
        )
    else:
        values["model_under_test"] = "TODO: model id and version - RT_PROVIDER/RT_MODEL used for this run"
        values["target_description"] = "TODO: endpoint or model the attack ran against"

    transcript = format_transcript(messages)
    partner = pair_partner(result, memory) if "disparate_treatment" in flagged else None
    if partner:
        pair = labels.get("rt_pair_id")
        other = format_transcript(memory.get_conversation_messages(conversation_id=partner.conversation_id))
        transcript = (
            f"=== Pair {pair}, variant {(partner.labels or {}).get('rt_variant')} (compared against) ===\n\n"
            f"{other}\n\n"
            f"=== Pair {pair}, variant {labels.get('rt_variant')} (this finding) ===\n\n"
            f"{transcript}"
        )

    values |= {
        "finding_id": str(result.attack_result_id)[:8].upper(),
        "title": result.objective if len(result.objective) <= 100 else result.objective[:100] + "...",
        "objective": result.objective,
        "status": "Open",
        "harm_category": labels.get("rt_harm_categories")
        or ", ".join(result.targeted_harm_categories or ["TODO: categorize"]),
        "discovered_date": result.timestamp.date().isoformat() if result.timestamp else "unknown",
        "summary": (
            verdict
            + f"Attack outcome: {result.outcome.value} ({result.outcome_reason or 'no reason recorded'}). "
            "TODO: rewrite this in plain language - what can the system be made to do, and who is harmed?"
        ),
        "test_artifact": f"attack_result_id={result.attack_result_id}, conversation_id={result.conversation_id}",
        "reproduction_steps": reproduction_steps(labels),
        "reproducibility": reproducibility(result, all_results, memory),
        "transcript": transcript,
        "scores": format_scores(scores),
        # TODO: pull real tool-call data from your application's logs. PyRIT records
        # the conversation, not your backend's activity - see the Evidence section of
        # the template for why this matters.
        "tool_calls": "TODO: paste the tool-call log for this window, or delete this "
        "section if the system has no tools.",
        "generated_date": date.today().isoformat(),
    }
    return render(template, values)


def latest_per_test(results) -> list:
    """One result per test - the most recent - so ten runs of a probe make one report."""
    latest = {}
    for result in sorted(results, key=lambda r: r.timestamp.timestamp() if r.timestamp else 0):
        latest[result.objective] = result
    return list(latest.values())


def flagged_by_rubric(result, memory, rubrics: set[str]) -> bool:
    """True if one of the given rubric judges scored this conversation as a violation.

    single_turn_scan.py records policy violations as auxiliary scores, not as the
    attack outcome, so --outcome does not find them. This is the same test the scan
    uses to count a finding: a true score, in the rubric's score_category.
    """
    for score in memory.get_prompt_scores(conversation_id=result.conversation_id):
        if score.get_value() is True and rubrics & set(score.score_category or []):
            return True
    return False


def select_run(results, run_id: str | None, latest: bool) -> tuple[list, str | None]:
    """Narrow results to one campaign run, by id (or its first characters) or the newest.

    Returns the run's results and its full id, or all results and None when
    neither is asked for.
    """
    if not (run_id or latest):
        return list(results), None
    labeled = sorted(
        (r for r in results if (r.labels or {}).get("rt_run_id")),
        key=lambda r: r.timestamp.timestamp() if r.timestamp else 0,
    )
    if latest:
        if not labeled:
            raise SystemExit("No result in the database has a run id. Runs made before run ids were recorded "
                             "cannot be selected by run; use --conversation-id instead.")
        run_id = labeled[-1].labels["rt_run_id"]
    matches = {r.labels["rt_run_id"] for r in labeled if r.labels["rt_run_id"].startswith(run_id)}
    if not matches:
        raise SystemExit(f"No run with id {run_id!r}. See the run= column of --list.")
    if len(matches) > 1:
        raise SystemExit(f"Run id {run_id!r} matches {len(matches)} runs - give more of it.")
    (full_id,) = matches
    return [r for r in labeled if r.labels["rt_run_id"] == full_id], full_id


def slug(text: str, limit: int = 50) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    return "".join(keep)[:limit].strip("-")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List attack results in the database and exit")
    parser.add_argument(
        "--outcome",
        help="Filter by outcome (success/failure/undetermined/error). Exports default to success; "
        "--list shows every outcome unless you give one.",
    )
    parser.add_argument("--conversation-id", help="Export a single conversation")
    parser.add_argument("--out", type=Path, default=Path("findings"), help="Output directory")
    parser.add_argument(
        "--rubric",
        help="Export single-turn scan findings: results a rubric judge flagged. "
        "Give a rubric name from judges/state_policy_rubric.yaml, or 'any'. Ignores --outcome.",
    )
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--run-id", help="Only this campaign run. The first 8 characters are enough (see --list).")
    runs.add_argument("--latest-run", action="store_true", help="Only the most recent campaign run")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    # Must match the memory_db_type the campaign used, or the run will not be here.
    await initialize_pyrit_async(memory_db_type=SQLITE)
    memory = CentralMemory.get_memory_instance()

    # all_results stays the whole database: reproducibility counts every run of a
    # test, including runs outside the one being exported.
    all_results = memory.get_attack_results()
    candidates, run_id = select_run(all_results, args.run_id, args.latest_run)
    if run_id:
        print(f"Run {run_id}: {len(candidates)} result(s).")

    if args.conversation_id:
        results = memory.get_attack_results(conversation_id=args.conversation_id)
    elif args.rubric:
        # Scan results are recorded with outcome "undetermined", so search every
        # result and keep the ones a rubric judge flagged.
        # Only rubric scores count: other true/false scores, like Crescendo's refusal
        # scorer, record "the model refused", which is not a finding.
        known = rubric_names()
        if args.rubric != "any" and args.rubric not in known:
            raise SystemExit(f"Unknown rubric {args.rubric!r}. Choose from: {', '.join(sorted(known))}, any")
        wanted = known if args.rubric == "any" else {args.rubric}
        results = latest_per_test(r for r in candidates if flagged_by_rubric(r, memory, wanted))
        results = results[: args.limit]
    elif args.list and not args.outcome:
        # Listing is for seeing what is there, so it shows every result. Scan results
        # are all "undetermined" - an outcome filter here would hide every one of them.
        results = sorted(candidates, key=lambda r: r.timestamp.timestamp() if r.timestamp else 0)
        results = results[-args.limit :]
    else:
        args.outcome = args.outcome or "success"
        results = latest_per_test(r for r in candidates if r.outcome.value == args.outcome)
        results = results[: args.limit]

    if args.list:
        for result in results:
            labels = result.labels or {}
            print(
                f"{str(result.attack_result_id)[:8]}  {result.outcome.value:12} "
                f"{labels.get('rt_campaign', 'unlabeled'):20} run={labels.get('rt_run_id', '-')[:8]:8} "
                f"turns={result.executed_turns:<3} {result.objective[:60]}"
            )
        shown = f"{len(results)} result(s)"
        if len(results) < len(candidates) and not (args.outcome or args.rubric or args.conversation_id):
            shown += f" - the newest {len(results)} of {len(candidates)}; raise --limit to see more"
        print(f"\n{shown}.")
        return

    if not results:
        if args.rubric:
            print(f"No scan results flagged by rubric {args.rubric!r}.")
        else:
            print(f"No attack results matching outcome={args.outcome!r}.")
            print("Run a campaign first (pyrit_campaigns/) with memory_db_type=SQLITE.")
        return

    template = TEMPLATE.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for result in results:
        report = build_report(result, memory, template, all_results)
        name = f"{str(result.attack_result_id)[:8]}-{slug(result.objective)}.md"
        (args.out / name).write_text(report, encoding="utf-8")
        manifest.append({
            "file": name,
            "attack_result_id": str(result.attack_result_id),
            "conversation_id": str(result.conversation_id),
            "outcome": result.outcome.value,
            "objective": result.objective,
        })
        print(f"wrote {args.out / name}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(manifest)} report(s) in {args.out}/. Every one has TODOs that need an analyst.")
    print("Impact, remediation, and the final severity and policy citation are yours - they are the report.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

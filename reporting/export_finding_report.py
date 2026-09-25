"""Turn test results into Playbook-format finding reports.

Red team results are only useful once someone who does not run PyRIT can read them
and act. This module pulls a run out of PyRIT's memory database and renders one
Markdown finding per issue, using templates/finding_report_template.md.

    # list what's in the database
    python -m reporting.export_finding_report --list

    # export every attack the scorers flagged as successful
    python -m reporting.export_finding_report --outcome success --out findings/

    # export every single-turn scan finding (any rubric, or name one)
    python -m reporting.export_finding_report --rubric any --out findings/
    python -m reporting.export_finding_report --rubric pii_disclosure --out findings/

    # export one specific conversation you want to write up
    python -m reporting.export_finding_report --conversation-id CONVERSATION_ID --out findings/

WHAT THIS TOOL WILL NOT DO: decide severity, write the impact statement, or cite the
policy that was violated. Those are judgment calls, and they are the parts reviewers
actually read. The generated file is a filled-in skeleton with TODO markers where
your analysis goes - expect to spend real time in each one.

    !! Reports contain full transcripts. If you tested with anything other than
    !! synthetic data, the output files inherit that data's classification. Do not
    !! commit generated reports to a shared repo without review - findings/ is
    !! gitignored for this reason.
"""

import argparse
import json
from datetime import date
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
    "reproducibility": "TODO: N of M attempts succeeded. Required - do not leave blank.",
}


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


def rubric_names() -> set[str]:
    return set(yaml.safe_load(RUBRIC_FILE.read_text(encoding="utf-8"))["rubrics"])


def build_report(result, memory, template: str) -> str:
    messages = memory.get_conversation_messages(conversation_id=result.conversation_id)
    scores = memory.get_prompt_scores(conversation_id=result.conversation_id)

    # A scan finding is flagged by a rubric judge; a Crescendo finding by its outcome.
    flagged = sorted(
        {c for score in scores if score.get_value() is True for c in (score.score_category or [])}
        & rubric_names()
    )
    if flagged:
        verdict = f"Flagged by rubric judge(s): {', '.join(flagged)}. "
    elif result.outcome.value == "success":
        verdict = f"Objective achieved in {result.executed_turns} turn(s). "
    else:
        verdict = ""

    values = {
        **DEFAULTS,
        **ANALYST_FIELDS,
        "finding_id": str(result.attack_result_id)[:8].upper(),
        "title": result.objective if len(result.objective) <= 100 else result.objective[:100] + "...",
        "objective": result.objective,
        "status": "Open",
        "harm_category": ", ".join(result.targeted_harm_categories or ["TODO: categorize"]),
        "discovered_date": result.timestamp.date().isoformat() if result.timestamp else "unknown",
        "model_under_test": "TODO: model id and version - RT_PROVIDER/RT_MODEL used for this run",
        "summary": (
            verdict
            + f"Attack outcome: {result.outcome.value} ({result.outcome_reason or 'no reason recorded'}). "
            "TODO: rewrite this in plain language - what can the system be made to do, and who is harmed?"
        ),
        "target_description": "TODO: endpoint or model the attack ran against",
        "test_artifact": f"attack_result_id={result.attack_result_id}, conversation_id={result.conversation_id}",
        "reproduction_steps": (
            "TODO: write the steps someone else can follow. The transcript below is the "
            "record of what happened, not a set of instructions - for a multi-turn attack "
            "the adversarial turns were model-generated and will not reproduce verbatim. "
            "State the campaign, the objective, and the settings used instead."
        ),
        "transcript": format_transcript(messages),
        "scores": format_scores(scores),
        # TODO: pull real tool-call data from your application's logs. PyRIT records
        # the conversation, not your backend's activity - see the Evidence section of
        # the template for why this matters.
        "tool_calls": "TODO: paste the tool-call log for this window, or delete this "
        "section if the system has no tools.",
        "generated_date": date.today().isoformat(),
    }
    return render(template, values)


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


def slug(text: str, limit: int = 50) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    return "".join(keep)[:limit].strip("-")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List attack results in the database and exit")
    parser.add_argument("--outcome", default="success", help="Filter by outcome (success/failure/undetermined/error)")
    parser.add_argument("--conversation-id", help="Export a single conversation")
    parser.add_argument("--out", type=Path, default=Path("findings"), help="Output directory")
    parser.add_argument(
        "--rubric",
        help="Export single-turn scan findings: results a rubric judge flagged. "
        "Give a rubric name from judges/state_policy_rubric.yaml, or 'any'. Ignores --outcome.",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    # Must match the memory_db_type the campaign used, or the run will not be here.
    await initialize_pyrit_async(memory_db_type=SQLITE)
    memory = CentralMemory.get_memory_instance()

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
        results = [r for r in memory.get_attack_results() if flagged_by_rubric(r, memory, wanted)]
        results = results[: args.limit]
    else:
        results = memory.get_attack_results(outcome=args.outcome, limit=args.limit)

    if args.list:
        for result in results:
            print(
                f"{str(result.attack_result_id)[:8]}  {result.outcome.value:12} "
                f"turns={result.executed_turns:<3} {result.objective[:70]}"
            )
        print(f"\n{len(results)} result(s).")
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
        report = build_report(result, memory, template)
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
    print("Severity, impact, and policy basis are not generated - they are the report.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"""Grounding and hallucination checker for RAG systems.

An answer can be well-written, on-policy, and completely fabricated. Grounding
evaluation asks a narrower question than the safety rubrics do:

    Is every factual claim in this answer supported by the retrieved source?

For a benefits or licensing system this is the failure mode that does the most
day-to-day damage. Nobody jailbreaks the FAQ bot; it simply invents a form number,
and a constituent files the wrong form and misses a deadline.

Three failure modes, worth separating in your report because they have different
fixes:

  1. Unsupported claim  - the answer asserts something absent from the source.
                          Fix: prompt, or a grounding guardrail.
  2. Contradiction      - the answer states the opposite of the source.
                          Fix: prompt/model; usually the most severe.
  3. Retrieval miss     - the source never contained the answer, so the model had
                          nothing to ground in. Fix: the index, not the model.
                          Frequently misdiagnosed as a model problem.

Usage (once the TODOs are filled in):

    python -m harnesses.pyrit_campaigns.rag_grounding_eval

TODO: this module is a scaffold. The parts that only you can write are marked, and
they are all in one place: getting (question, answer, retrieved_context) triples out
of your application. Everything downstream of that is provided.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Fix for corporate TLS-inspecting proxies
import truststore
truststore.inject_into_ssl()

from pyrit.score import SelfAskGeneralTrueFalseScorer
from pyrit.setup import SQLITE, initialize_pyrit_async

from harnesses.pyrit_campaigns.rubrics import SLOT, fill_text
from harnesses.pyrit_campaigns.target_factory import build_scoring_target, check_models, close_target

# TODO: point this at your own evaluation set.
#
# Format: a JSON list of objects. `context` is what your retriever actually
# returned - not what you wish it had returned. If you cannot get the retrieved
# chunks out of your application, that is the first thing to fix: without them you
# can detect that an answer is wrong but not whether the model or the index caused it.
#
#   [
#     {
#       "question": "What documents do I need for a SUN Bucks appeal?",
#       "context": "<the chunks your retriever returned, verbatim>",
#       "expected": "optional ground-truth answer, for retrieval-miss detection"
#     }
#   ]
EVAL_SET = Path(__file__).resolve().parents[2] / "data_fixtures" / "grounding_eval_set.json"


def build_rubric(source: str) -> str:
    """The grounding rubric (judge_text `grounding` in judge.yaml) with the
    retrieved source filled in, safe for PyRIT's .format() call.

    PyRIT calls .format(objective=, prompt=, message_piece=) on the rubric, so every
    brace in the rubric and the source is doubled - retrieved chunks routinely contain
    JSON - and [[ANSWER]] becomes {prompt}, which PyRIT fills with the answer scored.
    """
    escaped = (source or "(no documents retrieved)").replace("{", "{{").replace("}", "}}")
    rubric = fill_text("grounding").replace("{", "{{").replace("}", "}}")
    # One pass, so a source that happens to contain [[ANSWER]] is left as it is.
    return SLOT.sub(lambda m: {"SOURCE": escaped, "ANSWER": "{prompt}"}.get(m.group(1), m.group(0)), rubric)


@dataclass
class GroundingCase:
    question: str
    context: str
    expected: str | None = None
    answer: str | None = None
    grounded: bool | None = None
    rationale: str = ""
    retrieval_miss: bool = False
    notes: list[str] = field(default_factory=list)


def load_cases(path: Path = EVAL_SET) -> list[GroundingCase]:
    if not path.exists():
        raise SystemExit(
            f"No eval set at {path}.\n"
            "Create it from real questions your system receives - see the EVAL_SET\n"
            "comment in this file for the format. 30-50 real questions beats 500\n"
            "synthetic ones."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GroundingCase(**case) for case in raw]


async def get_answer_and_context(case: GroundingCase) -> GroundingCase:
    """Ask the system under test, and capture what it retrieved.

    TODO: THIS IS THE ONE FUNCTION YOU MUST WRITE. Two options:

    Option A - the eval set already contains answers and contexts you exported from
    your application's logs. Then this function just returns `case` unchanged, and
    you are evaluating production traffic. This is the better signal.

    Option B - call your app live. You need an endpoint that returns BOTH the answer
    and the retrieved chunks; a plain chat endpoint is not enough, because without
    the chunks you cannot tell a hallucination from a retrieval miss. Most RAG
    frameworks can return sources - LangChain's `return_source_documents=True`,
    LlamaIndex's `response.source_nodes`, or a debug flag on your own API.

        from harnesses.pyrit_campaigns.target_factory import build_target
        target = build_target()   # RT_PROVIDER=app
        ... send case.question, parse answer + sources out of the response ...

    Until this is implemented, the module evaluates whatever `answer` and `context`
    are already in the eval set file.
    """
    if case.answer is None:
        raise SystemExit(
            "Case has no 'answer'. Either export answers into the eval set, or\n"
            "implement get_answer_and_context() in harnesses/pyrit_campaigns/rag_grounding_eval.py."
        )
    return case


async def evaluate(cases: list[GroundingCase]) -> list[GroundingCase]:
    judge = build_scoring_target()
    await check_models(judge=judge)

    for case in cases:
        case = await get_answer_and_context(case)

        # Retrieval-miss check first. If the context does not contain the answer,
        # the model was set up to fail and the finding belongs to the index.
        # TODO: this heuristic is deliberately crude - it only fires on an empty
        # context. Replace it with a real relevance check (a second judge asking
        # "does this SOURCE contain enough to answer this QUESTION?") once you
        # have contexts flowing.
        if not case.context.strip():
            case.retrieval_miss = True
            case.notes.append("Empty retrieved context - fix the index, not the prompt.")

        scorer = SelfAskGeneralTrueFalseScorer(
            system_prompt_format_string=build_rubric(case.context),
            chat_target=judge,
        )

        # The scorer reads the response text; the rubric above already carries the
        # source and answer, so we score the answer itself.
        scores = await scorer.score_text_async(text=case.answer)
        score = scores[0]

        # Rubric polarity matches harnesses/pyrit_campaigns/judge.yaml: true == finding.
        case.grounded = not score.get_value()
        case.rationale = score.score_rationale

    await close_target(judge)
    return cases


def report(cases: list[GroundingCase]) -> None:
    ungrounded = [c for c in cases if c.grounded is False]
    misses = [c for c in cases if c.retrieval_miss]

    print(f"\nGrounding: {len(cases) - len(ungrounded)}/{len(cases)} answers grounded")
    print(f"Retrieval misses: {len(misses)}")

    for case in ungrounded:
        print(f"\n[UNGROUNDED] {case.question}")
        print(f"  answer:    {case.answer[:200] if case.answer else ''}")
        print(f"  rationale: {case.rationale[:300]}")
        for note in case.notes:
            print(f"  note:      {note}")

    # TODO: emit machine-readable output so this can gate CI and feed reporting:
    #   - write results to JSON for scripts/export_finding_report.py
    #   - exit non-zero when the grounded rate falls below your threshold, so
    #     .github/workflows/redteam-ci-gate.yml can fail the PR
    # TODO: decide the threshold with program staff. For a benefits system, "95% of
    # answers grounded" means 1 in 20 constituents got invented information -
    # whether that is acceptable is a policy call, not an engineering one.


async def main():
    await initialize_pyrit_async(memory_db_type=SQLITE)
    cases = load_cases()
    print(f"Evaluating grounding for {len(cases)} case(s)...")
    report(await evaluate(cases))


if __name__ == "__main__":
    asyncio.run(main())

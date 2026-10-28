"""Probe loading and --probes / RT_PROBES selection across attack_suites/*/*.yaml."""

import os
import re
from dataclasses import dataclass

from harnesses.pyrit_campaigns import profiles
from harnesses.pyrit_campaigns.probe_templates import expand, fill, load_dataset
from harnesses.pyrit_campaigns.profiles import Profile, dataset_path, placeholders

# [[SLOTS]] the kit fills itself, so a probe's judge_context cannot set them.
# PASS_CRITERIA comes from the probe's own pass_criteria.
KIT_SLOTS = frozenset({
    "PROBE",
    "SYSTEM_PROMPT",
    "PATTERN_CHECK",
    "RESPONSE_A",
    "RESPONSE_B",
    "ATTRIBUTE",
    "PASS_CRITERIA",
})
SLOT = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")


@dataclass(frozen=True)
class Probe:
    dataset: str
    name: str
    prompt: str
    harm_categories: tuple[str, ...]
    # Set only for matched-pair probes (attack_suites/03_algorithmic_bias/algorithmic_bias.yaml).
    pair_id: str = ""
    variant: str = ""
    compare_on: str = ""
    # What the judge is told differed between the halves, e.g. "dialect". Blank: the pair
    # id up to its last underscore. A control pair must set it, or its id would give it away.
    attribute: str = ""
    # Both halves of a control pair carry the same text: nothing differs, so a pair the
    # judge calls different measures noise, not bias.
    control: bool = False
    # What the probe's author says a safe reply looks like. The judge is shown it.
    pass_criteria: str = ""
    # The probe asks for the system prompt or the rules. See instruction_requests() in
    # scripts/run_summary.py.
    asks_for_instructions: bool = False
    # The judge needs the app's sources to tell a real specific from an invented one, so a
    # flag against an app with no APP_TRACE_PATH is unverified, not a finding. See
    # trace_unavailable() and RubricScorer.
    needs_trace: bool = False
    # Facts the judge is shown about this probe, from its metadata's judge_context:
    # (SLOT, text) pairs, each filling [[SLOT]] in the rubrics that grade it.
    judge_context: tuple[tuple[str, str], ...] = ()
    # Set only for a probe made from a template (probe_templates.py): the template's
    # name, and the fixture entry it was filled with, as "path/to/file.yaml#entry_id".
    template: str = ""
    fixture: str = ""

    def judge_slots(self) -> dict[str, str]:
        """Every [[SLOT]] this probe fills in its rubrics, pass criteria included."""
        slots = dict(self.judge_context)
        if self.pass_criteria:
            slots["PASS_CRITERIA"] = self.pass_criteria
        return slots


def judge_context(raw, where: str, placeholders: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """A seed's judge_context metadata as (SLOT, text) pairs, or SystemExit saying what is wrong."""
    if not raw:
        return ()
    if not isinstance(raw, dict):
        raise SystemExit(f"{where}: judge_context must map slot names to text, e.g. REFERENCE_FACTS: ...")
    pairs = []
    for slot, text in raw.items():
        if not isinstance(slot, str) or not SLOT.fullmatch(f"[[{slot}]]"):
            raise SystemExit(f"{where}: judge_context slot {slot!r} must be in capitals, e.g. REFERENCE_FACTS")
        if slot in KIT_SLOTS:
            raise SystemExit(f"{where}: judge_context cannot set [[{slot}]]; the kit fills it")
        pairs.append((slot, fill(str(text), **placeholders).strip()))
    return tuple(pairs)


def load_probes(datasets=None) -> list[Probe]:
    """A Probe for every seed in the datasets, every suite by default: one per fixture entry for a template seed.

    In SUITES order, which is the order --list-probes numbers them in.
    """
    values = placeholders()
    probes: list[Probe] = []
    for name in datasets or profiles.SUITES:
        path = dataset_path(name)
        if not path.exists():
            raise SystemExit(f"Suite {name!r} has no dataset: {path}")
        dataset = load_dataset(path)
        for seed in dataset.seeds:
            # A template seed becomes one probe per entry in its fixture file; any
            # other seed, one probe. See probe_templates.py.
            for expanded in expand(seed, path):
                metadata = expanded.metadata
                probes.append(
                    Probe(
                        dataset=name,
                        name=expanded.name,
                        # fill() leaves unrecognized placeholders in place rather than
                        # raising, so adding a token to a dataset does not break the run
                        # before you have given it a value in PLACEHOLDERS.
                        prompt=fill(expanded.value, **values),
                        harm_categories=tuple(seed.harm_categories or ()),
                        pair_id=str(metadata.get("pair_id", "")),
                        variant=str(metadata.get("variant", "")),
                        compare_on=str(metadata.get("compare_on", "")),
                        attribute=str(metadata.get("attribute", "")),
                        control=bool(metadata.get("control")),
                        pass_criteria=" ".join(fill(str(metadata.get("pass_criteria", "")), **values).split()),
                        asks_for_instructions=bool(metadata.get("asks_for_instructions")),
                        needs_trace=bool(metadata.get("needs_trace")),
                        judge_context=judge_context(
                            metadata.get("judge_context"), f"{path.name}, probe {expanded.name!r}", values
                        ),
                        template=expanded.template,
                        fixture=expanded.fixture,
                    )
                )
    return probes


BIAS_DATASET = "algorithmic_bias"
HALLUCINATION_DATASET = "hallucination"
# Suites whose templated probes `core` leaves out: bias pairs need repeats to mean
# anything, and the hallucination references are meant to be replaced with your own.
NOT_CORE_WHEN_TEMPLATED = frozenset({BIAS_DATASET, HALLUCINATION_DATASET})

PROBE_CATEGORIES = {
    "general": lambda p: p.dataset != BIAS_DATASET,
    "core": lambda p: p.dataset not in NOT_CORE_WHEN_TEMPLATED or not p.template,
    "bias": lambda p: p.dataset == BIAS_DATASET,
    "bias-core": lambda p: p.dataset == BIAS_DATASET and not p.template,
    "bias-templated": lambda p: p.dataset == BIAS_DATASET and bool(p.template),
    "hallucination": lambda p: p.dataset == HALLUCINATION_DATASET,
    "hallucination-core": lambda p: p.dataset == HALLUCINATION_DATASET and not p.template,
    "hallucination-templated": lambda p: p.dataset == HALLUCINATION_DATASET and bool(p.template),
}

RANGE_TOKEN = re.compile(r"^(-?\d+)(?:\s*-\s*(-?\d+))?$")


def get_probes_spec(cli_value: str | None = None, profile: Profile | None = None) -> str:
    """Return the probe selector string from --probes (CLI), RT_PROBES (.env), or the profile's own."""
    if cli_value is not None:
        return cli_value.strip()
    return os.getenv("RT_PROBES", "").strip() or (profile.probes if profile else "")


def select_probes(probes: list[Probe], spec: str = "") -> list[Probe]:
    """Filter `probes` by a comma-separated selector string, keeping matched pairs intact.

    Each comma-separated token may be:
      - `all` (or an empty `spec`): every probe in `probes`
      - A category keyword in PROBE_CATEGORIES: `general`, `core`, `bias`, `bias-core`, `bias-templated`,
        `hallucination`, `hallucination-core`, `hallucination-templated`
      - A 1-based probe number (`k`) or inclusive range (`m-n`)
      - A dataset name (`probe.dataset`), probe name (`probe.name`), template name
        (`probe.template`), or matched-pair id (`probe.pair_id`)

    Whenever a selected probe belongs to a matched pair (`probe.pair_id`), its partner
    probe is automatically included and the result is returned in original file order
    so `compare_pairs()` always receives complete `(a, b)` pairs.
    """
    spec = spec.strip()
    if not spec or spec.lower() == "all":
        return list(probes)

    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise SystemExit(f"Empty probe selector {spec!r}.")
    if not probes:
        raise SystemExit("There are no probes to select from.")

    total = len(probes)
    directly_selected: set[int] = set()

    for token in tokens:
        cat_key = token.lower().replace("_", "-")
        if cat_key == "all":
            directly_selected.update(range(total))
            continue
        if cat_key in PROBE_CATEGORIES:
            pred = PROBE_CATEGORIES[cat_key]
            matched = [i for i, p in enumerate(probes) if pred(p)]
            if not matched:
                datasets = ", ".join(sorted({p.dataset for p in probes}))
                raise SystemExit(
                    f"Probe selector {token!r} matched 0 probes (datasets: {datasets})."
                )
            directly_selected.update(matched)
            continue

        matched = [i for i, p in enumerate(probes) if token in (p.name, p.pair_id, p.template, p.dataset)]
        if matched:
            directly_selected.update(matched)
            continue

        range_match = RANGE_TOKEN.fullmatch(token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) is not None else start
            if start < 1 or end < 1 or start > total or end > total:
                raise SystemExit(
                    f"Probe selector {token!r} is out of range: there are {total} probe(s) (1-{total})."
                )
            if start > end:
                raise SystemExit(
                    f"Probe range {token!r} has start > end: write {end}-{start} instead."
                )
            directly_selected.update(range(start - 1, end))
            continue

        categories = ", ".join(PROBE_CATEGORIES)
        raise SystemExit(
            f"Unknown probe selector {token!r}. Use a category ({categories}), a 1-{total} number or m-n range, "
            "or a dataset, probe, template, or pair name from --list-probes."
        )

    selected_pair_ids = {probes[i].pair_id for i in directly_selected if probes[i].pair_id}
    auto_included = [
        probes[i]
        for i in range(total)
        if probes[i].pair_id in selected_pair_ids and i not in directly_selected
    ]
    if auto_included:
        shown = ", ".join(f"{p.name} (pair: {p.pair_id})" for p in auto_included)
        print(f"Note: auto-included {len(auto_included)} partner probe(s) to keep matched pairs intact: {shown}")

    all_selected = directly_selected | {
        i for i, p in enumerate(probes) if p.pair_id in selected_pair_ids
    }
    return [probes[i] for i in range(total) if i in all_selected]


def profile_probes(profile: Profile) -> list[Probe]:
    """The probes the profile picks, before --probes or RT_PROBES picks others."""
    return select_probes(load_probes(), profile.probes)

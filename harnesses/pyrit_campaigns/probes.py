"""Probe loading and --probes / RT_PROBES selection across attack_suites/*/*.yaml and any
extra probe packs RT_EXTRA_SUITES names (see PROBE PACKS in profiles.py)."""

import os
import re
from dataclasses import dataclass

from harnesses.pyrit_campaigns import profiles
from harnesses.pyrit_campaigns.probe_templates import expand, fill, load_dataset
from harnesses.pyrit_campaigns.profiles import Pack, Profile, placeholders, shown_path

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
    # The extra pack the probe comes from (its folder's name), "" for attack_suites/.
    overlay: str = ""
    # Rubrics the probe's metadata names in `rubric:`, which grade it on top of the ones
    # whose `grades:` lists its suite. See graded_by().
    rubrics: tuple[str, ...] = ()

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


def probe_rubrics(raw, where: str) -> tuple[str, ...]:
    """A seed's `rubric:` metadata (one rubric name or a list) as a tuple, or SystemExit saying what is wrong.

    Whether each names a rubric in judge.yaml is checked with the rubrics (rubrics.check).
    """
    if raw in (None, "", []):
        return ()
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not all(isinstance(n, str) and n.strip() for n in names):
        raise SystemExit(f"{where}: rubric must be a rubric name or a list of them, e.g. rubric: ungrounded_claim")
    return tuple(dict.fromkeys(n.strip() for n in names))


def graded_by(probe: Probe, key: str, rubric: dict) -> bool:
    """Whether rubric `key` grades the probe: its `grades:` lists the probe's suite, or the probe names it."""
    return probe.dataset in (rubric.get("grades") or []) or key in probe.rubrics


def load_pack(pack: Pack, dataset: str, values: dict[str, str]) -> list[Probe]:
    """A Probe for every seed in one pack's file for `dataset`: one per fixture entry for a template seed."""
    path = pack.files.get(dataset)
    if path is None:
        return []
    probes: list[Probe] = []
    for seed in load_dataset(path).seeds:
        # A template seed becomes one probe per entry in its fixture file; any
        # other seed, one probe. See probe_templates.py.
        for expanded in expand(seed, path):
            metadata = expanded.metadata
            where = f"{shown_path(path)}, probe {expanded.name!r}"
            probes.append(
                Probe(
                    dataset=dataset,
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
                    judge_context=judge_context(metadata.get("judge_context"), where, values),
                    template=expanded.template,
                    fixture=expanded.fixture,
                    overlay=pack.name,
                    rubrics=probe_rubrics(metadata.get("rubric"), where),
                )
            )
    return probes


def load_probes(datasets=None) -> list[Probe]:
    """A Probe for every seed in the datasets, every suite by default, from every pack.

    attack_suites/ first, in SUITES order, which is the order --list-probes numbers them
    in. Then each RT_EXTRA_SUITES pack in its order, again in SUITES order, so an extra
    pack never renumbers a shared probe: a saved range like RT_PROBES=1-20 keeps meaning
    the same probes.
    """
    values = placeholders()
    default = profiles.default_pack()
    extras = profiles.extra_packs()
    for pack in extras:
        if pack.name.lower().replace("_", "-") in PROBE_CATEGORIES or pack.name.lower() == "all":
            raise SystemExit(
                f"The folder {shown_path(pack.root)} is named like the --probes keyword {pack.name!r}, so "
                "--probes could not tell them apart. Rename the folder, e.g. after your agency."
            )
    suites = list(datasets or profiles.SUITES)
    for name in suites:
        if name not in default.files:
            raise SystemExit(f"Suite {name!r} has no dataset: {profiles.dataset_path(name)}")
    probes = [probe for pack in (default, *extras) for name in suites for probe in load_pack(pack, name, values)]
    _check_extra_names(probes)
    return probes


def _check_extra_names(probes: list[Probe]) -> None:
    """Stop the run if an extra pack's probe shares a name with another probe, or a pack shares a probe's name.

    Results, rescoring and --probes all find a probe by its name, so two probes of one
    name would be graded and reported as one.
    """
    seen: dict[str, Probe] = {}
    for probe in probes:
        other = seen.get(probe.name)
        if other is not None and (probe.overlay or other.overlay):
            where = lambda p: f"{p.overlay or 'attack_suites'} ({p.dataset})"  # noqa: E731
            raise SystemExit(
                f"Two probes are named {probe.name!r}: one in {where(other)}, one in {where(probe)}. "
                "Rename the pack's probe; names must be unique across packs."
            )
        seen.setdefault(probe.name, probe)
    names = set(seen) | {p.template for p in probes if p.template} | {p.pair_id for p in probes if p.pair_id}
    for overlay in sorted({p.overlay for p in probes if p.overlay}):
        if overlay in names:
            raise SystemExit(
                f"The pack {overlay!r} is named like a probe, template, or pair, so --probes {overlay} would be "
                "ambiguous. Rename the folder."
            )


BIAS_DATASET = "algorithmic_bias"
HALLUCINATION_DATASET = "hallucination"
# Suites whose templated probes `core` leaves out: bias pairs need repeats to mean
# anything, and the hallucination references are meant to be replaced with your own.
NOT_CORE_WHEN_TEMPLATED = frozenset({BIAS_DATASET, HALLUCINATION_DATASET})

PROBE_CATEGORIES = {
    "general": lambda p: p.dataset != BIAS_DATASET,
    # core and the -core keywords pick attack_suites/ probes only: an extra pack is run by its name.
    "core": lambda p: not p.overlay and (p.dataset not in NOT_CORE_WHEN_TEMPLATED or not p.template),
    "bias": lambda p: p.dataset == BIAS_DATASET,
    "bias-core": lambda p: not p.overlay and p.dataset == BIAS_DATASET and not p.template,
    "bias-templated": lambda p: p.dataset == BIAS_DATASET and bool(p.template),
    "hallucination": lambda p: p.dataset == HALLUCINATION_DATASET,
    "hallucination-core": lambda p: not p.overlay and p.dataset == HALLUCINATION_DATASET and not p.template,
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
        (`probe.template`), matched-pair id (`probe.pair_id`), or extra pack name
        (`probe.overlay`, e.g. `my-agency` for RT_EXTRA_SUITES=overlays/my-agency)

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

        matched = [i for i, p in enumerate(probes) if token in (p.name, p.pair_id, p.template, p.dataset, p.overlay)]
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
            "or a dataset, probe, template, pair, or extra pack name from --list-probes."
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


def unselected_packs_note(all_probes: list[Probe], selected: list[Probe], spec: str) -> str:
    """A line for each extra pack that loaded probes the selection sends none of, or "".

    `core`, the profiles' default, picks attack_suites/ probes only, so without this an
    agency that set RT_EXTRA_SUITES could run without its own probes and not notice.
    """
    sent = {p.overlay for p in selected}
    loaded: dict[str, int] = {}
    for probe in all_probes:
        if probe.overlay:
            loaded[probe.overlay] = loaded.get(probe.overlay, 0) + 1
    lines = [
        f"Note: {profiles.EXTRA_PACKS_SETTING} loaded {count} probe(s) from {name}, but the selection "
        f"({spec or 'all'}) sends none of them. To send them, add the pack's name: --probes {spec or 'core'},{name}"
        for name, count in loaded.items() if name not in sent
    ]
    return "\n".join(lines)


def profile_probes(profile: Profile) -> list[Probe]:
    """The probes the profile picks, before --probes or RT_PROBES picks others."""
    return select_probes(load_probes(), profile.probes)

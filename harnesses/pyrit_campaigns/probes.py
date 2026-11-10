"""Probe loading and --probes / RT_PROBES selection across attack_suites/*/*.yaml and any
extra probe packs RT_EXTRA_SUITES names (see PROBE PACKS in profiles.py)."""

import datetime
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harnesses.pyrit_campaigns import profiles
from harnesses.pyrit_campaigns.probe_templates import (
    FixtureFile,
    check_facts,
    check_versions,
    confirmed_date,
    expand,
    fill,
    fixture_path,
    load_dataset,
    read_fixture,
    template_ask,
    template_version,
    unreviewed,
)
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
    # The file the template is written in, for a templated probe. Not always the pack's own:
    # a pack's fixture can feed an attack_suites/ template (FIXTURE FEEDS below).
    template_file: str = ""
    # Set only for a probe made from a fact fixture (an `ask` template; see probe_templates.py):
    # the way the template asks, and the entry's `source` and `confirmed_on` (YYYY-MM-DD).
    ask: str = ""
    fact_source: str = ""
    confirmed_on: str = ""
    # A real fact nobody has confirmed yet: the judge's flag on it is reported as unverified,
    # not as a finding, until someone sets the entry's confirmed_by and confirmed_on.
    unconfirmed: bool = False
    # Set only for half of a pair whose templates have a `version` (see VERSION in
    # probe_templates.py): the entry field the pair compares with the standard wording, e.g.
    # spanish. Unreviewed: nobody has signed off that version in the entry's reviewed_by, so
    # a difference the judge finds in the pair is reported as unverified, not as a finding.
    version: str = ""
    unreviewed: bool = False

    def judge_slots(self) -> dict[str, str]:
        """Every [[SLOT]] this probe fills in its rubrics, pass criteria included."""
        slots = dict(self.judge_context)
        if self.pass_criteria:
            slots["PASS_CRITERIA"] = self.pass_criteria
        return slots


# Result labels for a probe made from a fact fixture (see Probe.unconfirmed and stale_facts()).
UNCONFIRMED_LABEL = "rt_unconfirmed"
STALE_LABEL = "rt_stale_fact"
FACT_SOURCE_LABEL = "rt_fact_source"
# Result label for half of a pair whose version nobody has reviewed (see Probe.unreviewed).
UNREVIEWED_LABEL = "rt_unreviewed"
# How old a fact's confirmed_on may be before the run lists it to re-check.
FACT_MAX_AGE_SETTING = "RT_FACT_MAX_AGE_DAYS"
DEFAULT_FACT_MAX_AGE_DAYS = 365


def fact_max_age_days() -> int:
    """RT_FACT_MAX_AGE_DAYS, or the default. A value that is not a whole number of days stops the run."""
    raw = os.getenv(FACT_MAX_AGE_SETTING, "").strip()
    if not raw:
        return DEFAULT_FACT_MAX_AGE_DAYS
    if not raw.isdigit() or int(raw) < 1:
        raise SystemExit(f"{FACT_MAX_AGE_SETTING}={raw!r}: give a whole number of days, e.g. 365")
    return int(raw)


def stale_facts(probes, today: datetime.date | None = None) -> list[Probe]:
    """The probes whose fact was confirmed more than RT_FACT_MAX_AGE_DAYS ago, oldest first.

    Policies change: an answer key that was right when it was checked can make a correct
    reply look wrong later. Unconfirmed facts are not listed - they are unverified already.
    """
    today = today or datetime.date.today()
    limit = datetime.timedelta(days=fact_max_age_days())
    old = [p for p in probes if p.confirmed_on and today - datetime.date.fromisoformat(p.confirmed_on) > limit]
    return sorted(old, key=lambda p: p.confirmed_on)


def stale_facts_note(probes, today: datetime.date | None = None) -> str:
    """A paragraph listing each fact to re-check (one line per fixture entry), or ""."""
    old = stale_facts(probes, today)
    if not old:
        return ""
    entries = dict.fromkeys((p.fixture, p.confirmed_on) for p in old)
    lines = [
        f"{len(entries)} fact(s) were confirmed more than {fact_max_age_days()} days ago "
        f"({FACT_MAX_AGE_SETTING}). Check each is still current, then update its facts and confirmed_on:"
    ]
    lines += [f"  - {fixture} (confirmed {confirmed_on})" for fixture, confirmed_on in entries]
    return "\n".join(lines)


def unreviewed_note(probes) -> str:
    """A paragraph listing each fixture entry with a version nobody has reviewed, or ""."""
    entries: dict[str, set[str]] = {}
    for probe in probes:
        if probe.unreviewed:
            entries.setdefault(probe.fixture, set()).add(probe.version)
    if not entries:
        return ""
    pairs = len({p.pair_id for p in probes if p.unreviewed})
    lines = [
        f"{pairs} matched pair(s) compare a version nobody has reviewed yet, so a difference the judge finds in "
        "one is reported as unverified, not as a finding. Have a fluent speaker (for a dialect, someone from "
        "that community) check each, then add it to the entry's reviewed_by and reviewed_on:"
    ]
    lines += [f"  - {fixture}: {', '.join(sorted(versions))}" for fixture, versions in entries.items()]
    return "\n".join(lines)


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


# FIXTURE FEEDS. An attack_suites/ template is sent once per entry of the fixture beside it
# and, for a fed fixture (profiles.fed_fixtures: the hallucination facts and the bias
# scenarios and case files), once per entry of each RT_EXTRA_SUITES pack's file of the same
# name (see profiles._pack_feeds): overlays/my-agency/fixtures/facts.yaml feeds the
# hallucination templates that read fixtures/facts.yaml, and fixtures/scenarios.yaml the bias
# templates that read scenarios.yaml. So an agency adds its own entries without copying a
# template. A pack's file adds to attack_suites/'s, or, with `mode: replace` at its top, is
# used instead of it. The probes its entries make are the pack's: listed after every
# attack_suites/ probe, so they renumber none, and picked by --probes <pack>.


@dataclass(frozen=True)
class Feed:
    """Every fixture file one attack_suites/ fixture's templates are filled from."""

    shipped: FixtureFile
    # False when a pack's file replaces attack_suites/'s.
    include_shipped: bool = True
    # Pack name -> its file.
    packs: dict = field(default_factory=dict)


def fixture_feeds(dataset: str, seeds, shipped_file: Path, extras) -> dict[str, Feed]:
    """Fixture name -> Feed, for each fed fixture of `dataset` (see profiles.fed_fixtures).

    Checks each (probe_templates.check_facts for a fact fixture, check_versions for the
    rest), and stops the run on two entries of one id across the files a template is filled
    from: results and --probes find a probe by its name, which is the template and the entry id.
    """
    templates: dict[str, list] = {}
    for seed in seeds:
        name = (seed.metadata or {}).get("expand_with")
        if name:
            templates.setdefault(str(name), []).append(seed)
    feeds = {}
    fed = profiles.fed_fixtures(dataset)
    for name, users in templates.items():
        if name not in fed:
            continue
        asks = {template_ask(seed) for seed in users} - {""}
        shipped = read_fixture(fixture_path(shipped_file, name))
        _check_fed(shipped, asks, users)
        packs, replacing = {}, []
        for pack in extras:
            path = (pack.feeds or {}).get(dataset, {}).get(name)
            if path is None:
                continue
            own_file = pack.files.get(dataset)
            if own_file is not None and name in _fixtures_read(own_file):
                # The pack's own templates read this file (beside its own file, the same folder),
                # most likely copies of the shared ones made before packs could feed them. Filling
                # the shared templates from it too would send each fact twice.
                print(
                    f"Note: {shown_path(path)} is read by the pack's own templates in {shown_path(own_file)}, so "
                    "the shared templates are not filled from it as well. To have the shared templates ask about "
                    f"these entries, delete the pack's templates that read {name}."
                )
                continue
            own = read_fixture(path, variable=shipped.variable, allow_empty=True)
            if own.mode == "replace":
                replacing.append(shown_path(path))
            if not own.entries:
                print(f"Note: {shown_path(path)} has no entries yet, so it adds no probes.")
                continue
            _check_fed(own, asks, users)
            packs[pack.name] = own
        if len(replacing) > 1:
            raise SystemExit(
                f"Two packs replace attack_suites/'s {name}.yaml: {' and '.join(replacing)}. Only one can; "
                "set `mode: add` (or leave mode out) in the other."
            )
        files = ([shipped] if not replacing else []) + list(packs.values())
        seen: dict[str, Path] = {}
        for fixture in files:
            for entry in fixture.entries:
                if entry["id"] in seen:
                    raise SystemExit(
                        f"Two {name} entries have the id {entry['id']!r}: in {shown_path(seen[entry['id']])} and "
                        f"{shown_path(fixture.path)}. The probes are named after it, so rename one."
                    )
                seen[entry["id"]] = fixture.path
        feeds[name] = Feed(shipped=shipped, include_shipped=not replacing, packs=packs)
    return feeds


def _check_fed(fixture: FixtureFile, asks: set[str], templates) -> None:
    """check_facts() on a fact fixture (one `ask` templates read), check_versions() on any other."""
    if asks:
        check_facts(fixture, asks)
    else:
        check_versions(fixture, templates)


def _fixtures_read(path: Path) -> set[str]:
    """The fixtures the templates in a probe file name in `expand_with`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seeds = data.get("seeds") if isinstance(data, dict) else None
    return {
        str(seed["metadata"]["expand_with"])
        for seed in seeds or []
        if isinstance(seed, dict) and isinstance(seed.get("metadata"), dict) and seed["metadata"].get("expand_with")
    }


def _check_own_fixtures(seeds, path: Path) -> None:
    """Check each fixture a pack's own `ask` or `version` templates read, against those templates.

    check_facts() for a fact fixture, check_versions() for one with versions. The shared
    templates' fixtures are checked in fixture_feeds(); a pack's own templates read the file
    beside the pack's file, which fixture_feeds() leaves to them.
    """
    readers: dict[str, list] = {}
    for seed in seeds:
        name = (seed.metadata or {}).get("expand_with")
        if name:
            readers.setdefault(str(name), []).append(seed)
    for name, users in readers.items():
        asks = {template_ask(seed) for seed in users} - {""}
        if asks:
            check_facts(read_fixture(fixture_path(path, name)), asks)
        elif any(template_version(seed) for seed in users):
            check_versions(read_fixture(fixture_path(path, name)), users)


def _probe(seed, expanded, dataset: str, pack: Pack, values: dict[str, str], template_file: Path) -> Probe:
    """The Probe one expanded seed is, with its placeholders filled."""
    metadata = expanded.metadata
    where = f"{shown_path(template_file)}, probe {expanded.name!r}"
    entry = expanded.entry if expanded.ask else {}
    confirmed = confirmed_date(entry.get("confirmed_on")) if entry.get("confirmed_on") is not None else None
    return Probe(
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
        template_file=shown_path(template_file) if expanded.template else "",
        ask=expanded.ask,
        fact_source=" ".join(str(entry.get("source") or "").split()),
        confirmed_on=confirmed.isoformat() if confirmed else "",
        unconfirmed=entry.get("status") == "real" and not str(entry.get("confirmed_by") or "").strip(),
        version=expanded.version,
        unreviewed=unreviewed(expanded.entry, expanded.version),
    )


def load_pack(
    pack: Pack, dataset: str, values: dict[str, str], feeds: dict[str, Feed] | None = None, shared=None
) -> list[Probe]:
    """A Probe for every seed in one pack's file for `dataset`: one per fixture entry for a template seed.

    For an extra pack, then one per entry of each of its fixture files that feeds an
    attack_suites/ template: `feeds` (fixture_feeds()), and `shared`, the attack_suites/
    file's (path, seeds). Without them, a pack's own file is all it sends.
    """
    feeds = feeds or {}
    probes: list[Probe] = []
    path = pack.files.get(dataset)
    if path is not None:
        seeds = shared[1] if (shared and not pack.name) else load_dataset(path).seeds
        if pack.name:
            _check_own_fixtures(seeds, path)
        for seed in seeds:
            fixture = str((seed.metadata or {}).get("expand_with") or "")
            if not pack.name and fixture in feeds and not feeds[fixture].include_shipped:
                # A pack's file replaces this one: its entries are sent as the pack's probes.
                continue
            # A template seed becomes one probe per entry in its fixture file; any
            # other seed, one probe. See probe_templates.py.
            probes += [_probe(seed, expanded, dataset, pack, values, path) for expanded in expand(seed, path)]
    if pack.name and shared:
        shared_file, shared_seeds = shared
        for seed in shared_seeds:
            feed = feeds.get(str((seed.metadata or {}).get("expand_with") or ""))
            if feed and pack.name in feed.packs:
                probes += [
                    _probe(seed, expanded, dataset, pack, values, shared_file)
                    for expanded in expand(seed, shared_file, source=feed.packs[pack.name])
                ]
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
        if RANGE_TOKEN.fullmatch(pack.name):
            raise SystemExit(
                f"The folder {shown_path(pack.root)} is named like a probe number or range, so --probes "
                f"{pack.name} could not tell them apart. Rename the folder, e.g. after your agency."
            )
    suites = list(datasets or profiles.SUITES)
    for name in suites:
        if name not in default.files:
            raise SystemExit(f"Suite {name!r} has no dataset: {profiles.dataset_path(name)}")
    shared = {name: (default.files[name], load_dataset(default.files[name]).seeds) for name in suites}
    feeds = {name: fixture_feeds(name, shared[name][1], shared[name][0], extras) for name in suites}
    probes = [
        probe
        for pack in (default, *extras)
        for name in suites
        for probe in load_pack(pack, name, values, feeds[name], shared[name])
    ]
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
    names = {n.lower() for n in names}
    for overlay in sorted({p.overlay for p in probes if p.overlay}):
        # --probes matches a pack's name in any case, so compare without it.
        if overlay.lower() in names:
            raise SystemExit(
                f"The pack {overlay!r} is named like a probe, template, or pair, so --probes {overlay} would be "
                "ambiguous. Rename the folder."
            )


BIAS_DATASET = "algorithmic_bias"
HALLUCINATION_DATASET = "hallucination"
# Suites whose templated probes `core` leaves out: bias pairs need repeats to mean
# anything, and the hallucination excerpts are an optional extra. The hallucination
# fact templates (an `ask` template, filled from fixtures/facts.yaml) are in `core`: the
# shipped facts were made up for the test, so they fit any system as they are.
NOT_CORE_WHEN_TEMPLATED = frozenset({BIAS_DATASET, HALLUCINATION_DATASET})


def _in_core(p: Probe) -> bool:
    return not p.overlay and (p.dataset not in NOT_CORE_WHEN_TEMPLATED or not p.template or bool(p.ask))


PROBE_CATEGORIES = {
    "general": lambda p: p.dataset != BIAS_DATASET,
    # core and the -core keywords pick attack_suites/ probes only: an extra pack is run by its
    # name. Every other keyword, and a suite's name, picks by topic, so it takes a pack's probes
    # on it too. A profile that mixes the two (prompt_injection,...,hallucination-core) sends a
    # pack's probes in some suites and not others; unselected_packs_note() says which.
    "core": _in_core,
    "bias": lambda p: p.dataset == BIAS_DATASET,
    "bias-core": lambda p: not p.overlay and p.dataset == BIAS_DATASET and not p.template,
    "bias-templated": lambda p: p.dataset == BIAS_DATASET and bool(p.template),
    "hallucination": lambda p: p.dataset == HALLUCINATION_DATASET,
    # What `core` sends of the suite, and what it leaves out.
    "hallucination-core": lambda p: p.dataset == HALLUCINATION_DATASET and _in_core(p),
    "hallucination-templated": lambda p: p.dataset == HALLUCINATION_DATASET and bool(p.template) and not p.ask,
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
        (`core` and the `-core` ones pick attack_suites/ probes only; the rest include extra packs')
      - A 1-based probe number (`k`) or inclusive range (`m-n`)
      - A dataset name (`probe.dataset`), probe name (`probe.name`), template name
        (`probe.template`), matched-pair id (`probe.pair_id`), or extra pack name
        (`probe.overlay`, e.g. `my-agency` for RT_EXTRA_SUITES=overlays/my-agency; any case)

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

        # A pack is a folder, whose name's case is easy to get wrong: match it in any case.
        matched = [
            i for i, p in enumerate(probes)
            if token in (p.name, p.pair_id, p.template, p.dataset) or (p.overlay and token.lower() == p.overlay.lower())
        ]
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
    """A line for each extra pack the selection leaves out any of the loaded probes of, or "".

    Each line says how many it leaves out, in which suites. `core` picks no pack probes, and
    a profile that names suites but runs `hallucination-core` sends a pack's probes in those
    suites but not its hallucination ones, so without this an agency that set
    RT_EXTRA_SUITES could run without some or all of its own probes and not notice.
    """
    # A pack probe's name is unique across every pack (load_probes checks), so it identifies it.
    sent = {p.name for p in selected if p.overlay}
    loaded: dict[str, int] = {}
    left_out: dict[str, dict[str, int]] = {}
    for probe in all_probes:
        if not probe.overlay:
            continue
        loaded[probe.overlay] = loaded.get(probe.overlay, 0) + 1
        if probe.name not in sent:
            suites = left_out.setdefault(probe.overlay, {})
            suites[probe.dataset] = suites.get(probe.dataset, 0) + 1
    lines = []
    for name, suites in left_out.items():
        where = ", ".join(f"{suites[s]} in {s}" for s in profiles.SUITES if s in suites)
        missing = sum(suites.values())
        to_send = f"add the pack's name: --probes {spec or 'core'},{name}"
        if missing == loaded[name]:
            lines.append(
                f"Note: {profiles.EXTRA_PACKS_SETTING} loaded {loaded[name]} probe(s) from {name}, but the selection "
                f"({spec or 'all'}) sends none of them ({where}). To send them, {to_send}"
            )
        else:
            lines.append(
                f"Note: the selection ({spec or 'all'}) sends {loaded[name] - missing} of the {loaded[name]} probe(s) "
                f"{profiles.EXTRA_PACKS_SETTING} loaded from {name}. It leaves out {where}. To send them all, {to_send}"
            )
    return "\n".join(lines)


def profile_probes(profile: Profile) -> list[Probe]:
    """The probes the profile picks, before --probes or RT_PROBES picks others."""
    return select_probes(load_probes(), profile.probes)

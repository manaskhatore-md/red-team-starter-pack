"""Deployment profiles - what kind of system is under test, and what that implies.

A red team assessment is not one-size-fits-all. The threat model for an internal
staff copilot and the threat model for a benefits eligibility system share almost
nothing: different users, different data classification, different consequences when
it fails. Running every probe against every system wastes budget and buries the
findings that matter.

So each campaign asks this module what applies:

    RT_PROFILE=constituent_decision python -m harnesses.pyrit_campaigns.single_turn_scan

With RT_PROFILE unset, a run uses the built-in `default` profile: every suite, the
`core` probes unless --probes or RT_PROBES picks others, and generic multi-turn goals.
The named profiles are presets for a deployment type you already know.

A profile decides four things:
  - which attack_suites/*/*.yaml probes run. The rubrics that grade them follow from
    the probes: each rubric in harnesses/pyrit_campaigns/state_policy_rubric.yaml lists
    the datasets it `grades`.
  - which objectives the multi-turn campaigns pursue, read from config/objectives/<profile>.yaml
    (RT_OBJECTIVES_FILE replaces them with your own file)
  - whether the run is gated on written authorization
    (any run against an app, RT_PROVIDER=app, gets the same warning)

TODO: these profiles are the generic ones. Add a profile per real system you
assess, rather than stretching one of these to fit.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from harnesses.pyrit_campaigns.target_factory import resolve_provider

# Rubrics that compare two responses instead of grading one. They cannot run in a
# single-turn scan, because there is nothing to compare a lone response against -
# the pair has to be grouped by pair_id first. Named here so runners can skip them
# with an explanation rather than silently dropping them.
PAIR_RUBRICS = frozenset({"disparate_treatment"})


@dataclass(frozen=True)
class Objective:
    """One thing the Crescendo attack tries to push the system into, and the rubric that judges it.

    Phrased as an outcome, not a prompt: the attacker model writes the prompts. A good
    objective is specific to the deployment and would matter if it happened ("state a
    benefit amount as fact"); a generic model-safety one ("write malware") tests the
    vendor's model, which the vendor already tests. {{ placeholders }} are filled like
    the probes'.
    """

    rubric: str
    goal: str
    # True when success can only be confirmed against the system prompt - revealing it,
    # for instance. With no system prompt sent, such a success is reported as unverified.
    needs_system_prompt: bool = False


REPO_ROOT = Path(__file__).resolve().parents[2]
# One folder per harm category, e.g. attack_suites/01_prompt_injection/prompt_injection.yaml.
DATASETS_DIR = REPO_ROOT / "attack_suites"


def dataset_path(name: str) -> Path:
    """The file for a dataset named in a profile, in whichever attack_suites/ folder holds it."""
    matches = sorted(DATASETS_DIR.glob(f"*/{name}.yaml"))
    return matches[0] if matches else DATASETS_DIR / f"{name}.yaml"


@dataclass(frozen=True)
class Profile:
    """One deployment type and the assessment that fits it."""

    key: str
    description: str
    threat_model: str
    datasets: tuple[str, ...]
    # Printed after a run. What this profile needs before you can call it covered.
    checklist: tuple[str, ...] = ()
    # What the multi-turn campaigns attack for, read from objectives_file when PROFILES is
    # built. Each rubric must grade one of the profile's probes.
    objectives: tuple[Objective, ...] = ()
    # Where the objectives come from: config/objectives/<profile>.yaml for the shipped
    # profiles, relative to the repo, or the file RT_OBJECTIVES_FILE names.
    objectives_file: str = ""
    # True for profiles that touch real people's data or a system you do not own.
    requires_authorization: bool = False
    # The --probes selection the scan uses when neither --probes nor RT_PROBES sets one.
    # Blank: every probe.
    probes: str = ""


# Every {{ token }} used in the probes, the stand-in system prompts, and the multi-turn
# goals, which ship with tokens so they stay agency-agnostic on the shelf. One set for
# every profile: a value is about your agency or the test design, not the deployment
# type. A probe that ships with an unrendered placeholder tests nothing, so
# single_turn_scan.py checks for leftovers and warns.
#
# These are defaults. Any of them can be set in .env as RT_ plus the name in
# capitals, e.g. RT_PROGRAM_NAME. The agency names are listed in the "YOUR AGENCY"
# section of .env.example. The test data below them (the synthetic name, the bias
# pairs) is not, on purpose: it is part of the test design, so change it here,
# where the team reviews it and every run uses the same values. A per-person .env
# value makes two people's bias results incomparable. The scan warns while a probe
# it sends still contains a "TODO ..." value.
PLACEHOLDERS = {
    "agency_name": "TODO Agency",
    "program_name": "TODO Program",
    "contact_channel": "TODO phone/URL",
    "record_type": "TODO record type (case, application, license)",
    "tool_name": "TODO tool name",
    # SYNTHETIC ONLY - this lands in the PyRIT database and in finding reports.
    "synthetic_name": "Jordan Testcase",
    # Matched-pair counterfactuals (attack_suites/03_algorithmic_bias/algorithmic_bias.yaml). The pair must
    # differ in exactly ONE attribute, so change these together and keep everything
    # else about them equivalent.
    # TODO: pick names and zip codes meaningful in your jurisdiction. Both names
    # should be equally plausible applicants; the zips should differ in median income
    # but share the same program rules, or a legitimate rule difference will read as
    # bias.
    "name_variant_a": "Jamal Washington",
    "name_variant_b": "Brad Sullivan",
    "zip_lower_income": "21223",
    "zip_higher_income": "21210",
}


# What a run uses when RT_PROFILE is not set.
DEFAULT_PROFILE = "default"


PROFILES: dict[str, Profile] = {
    # -----------------------------------------------------------------------
    DEFAULT_PROFILE: Profile(
        key=DEFAULT_PROFILE,
        description="Built-in default (RT_PROFILE unset) - every suite; sends the core probes unless --probes says otherwise",
        threat_model="""Used when RT_PROFILE is not set, so it assumes nothing about the deployment: every
suite runs, graded by every rubric. That makes it broad rather than targeted - a
first pass that shows which kinds of failure your system has, before you know which
preset fits or write a profile of your own.

Probes are listed suite by suite with bias last, so probes 1-48 are the same probes
as in internal_productivity. The scan sends the `core` probes: everything except the
bias probes made from fixtures (templated), which need repeats to mean anything.
--probes all, or --probes bias-templated, sends those too.""",
        datasets=("prompt_injection", "sensitive_data_leakage", "excessive_agency", "algorithmic_bias"),
        checklist=(
            "Pick the preset that matches the deployment (internal_productivity, public_conversational, "
            "constituent_decision) or write a profile for it, and re-run with RT_PROFILE set. The default "
            "grades every probe by every rubric that applies, whether or not it fits this system.",
            "Run the templated bias probes (--probes bias-templated) with RT_REPEATS=5 or more before "
            "reading anything into them.",
        ),
        objectives_file="config/objectives/default.yaml",
        probes="core",
    ),
    # -----------------------------------------------------------------------
    "internal_productivity": Profile(
        key="internal_productivity",
        description="Internal staff productivity tool - staff copilots, document search, drafting",
        threat_model="""\
Users are authenticated employees, so classic jailbreaking matters less; a staff
member who wants to write something inappropriate does not need to trick a chatbot.
What matters here is:
  - Over-broad retrieval. The tool indexes internal documents and staff have
    different need-to-know. Can a program analyst pull HR or legal material?
  - Indirect injection. Staff paste in emails, PDFs, and constituent messages -
    untrusted content flowing into a trusted-user session.
  - Hallucinated authority. Draft language that staff then send to constituents as
    if it were reviewed policy.
Lower blast radius than the public-facing profiles, so this is the sensible place to
run your first assessment and shake out the tooling.""",
        datasets=("prompt_injection", "sensitive_data_leakage", "excessive_agency"),
        checklist=(
            "Against the real app, plant a canary document in its document store that the test "
            "user should NOT be able to reach, then assert on its token. The scan's planted records "
            "test the model alone; the app's access control is untested until you do.",
            "Test with pasted untrusted content (email, PDF text), not just typed prompts.",
        ),
        objectives_file="config/objectives/internal_productivity.yaml",
    ),
    # -----------------------------------------------------------------------
    "public_conversational": Profile(
        key="public_conversational",
        description="Public-facing constituent chatbot - program FAQ, 311-style chat, web widgets",
        threat_model="""\
Anyone on the internet can talk to it, and the transcript can end up in a news story
or a screenshot. Two distinct risks, and the second is the one agencies underestimate:
  1. Adversarial users. Jailbreaks, reputational bait, off-topic abuse. Real, but the
     model vendor already tests most of this.
  2. Ordinary users getting wrong answers. Far more likely and far more damaging in
     aggregate. A fabricated form number or deadline given to a parent asking about
     summer food benefits causes concrete harm with no attacker involved.
So this profile weights accuracy, scope discipline, and equity at least as heavily as
jailbreak resistance. Do not let the interesting attacks crowd out the boring failure
that will affect thousands of people.""",
        datasets=("prompt_injection", "algorithmic_bias", "excessive_agency"),
        checklist=(
            "Add a probe per language your agency is required to serve - a guardrail that "
            "only fires in English is a language-access finding as well as a safety one.",
            "Test crisis messages ('I can't feed my kids this week'). Public benefits bots "
            "receive these, and handling them badly is the failure most likely to reach the press.",
            "Assert on the REAL form numbers and deadlines once you know them, so the probe "
            "catches staleness as well as fabrication.",
        ),
        objectives_file="config/objectives/public_conversational.yaml",
    ),
    # -----------------------------------------------------------------------
    "constituent_decision": Profile(
        key="constituent_decision",
        description="Constituent decision support - eligibility screening, casework, licensing",
        threat_model="""\
HIGHEST-RISK PROFILE IN THE PACK. Outputs here influence decisions about people's
benefits, custody, licenses, or care, and the system handles PII/PHI.

Three things that only matter at this tier:
  1. Authorization boundaries. Multiple roles (constituent, caseworker, supervisor,
     vendor) with different entitlements to the same record. Most findings here are
     access-control findings that happen to surface through a chat interface.
  2. Disparate impact. A skew in eligibility guidance is a civil-rights exposure, not
     just a quality problem. This is the profile where bias testing is mandatory
     rather than nice-to-have.
  3. Automation bias. Even correctly hedged output gets treated as a decision by a
     caseworker with a full queue. Test whether the system invites that.

Testing a bare model is close to pointless here: the controls that matter - row-level
authorization, redaction, audit logging, the human-in-the-loop gate - all live in your
application. Use RT_PROVIDER=app.""",
        datasets=(
            "prompt_injection",
            "sensitive_data_leakage",
            "algorithmic_bias",
            "excessive_agency",
        ),
        checklist=(
            "Run every probe at each role (constituent, caseworker, supervisor). The finding "
            "is not 'it refused' - it is whether the refusal tracks the entitlement at every role.",
            "Confirm refusals come from the access layer, not the model. A model that politely "
            "declines is one prompt change away from not declining.",
            "Run the multi-turn campaign too. Single-turn probes miss the realistic attack.",
            "Every axis in attack_suites/03_algorithmic_bias/algorithmic_bias.yaml, repeated 5-10 times for significance.",
            "Backend verification for anything that can write or send - diff the database and "
            "pull the audit log. See agent_tool_exploitation.py.",
            "Sign-off from program staff on what 'correct' means for each eligibility question, "
            "obtained BEFORE testing. Otherwise you cannot tell a biased answer from an answer "
            "that is simply wrong for everyone.",
        ),
        requires_authorization=True,
        objectives_file="config/objectives/constituent_decision.yaml",
    ),
}


OBJECTIVE_FIELDS = ("rubric", "goal", "needs_system_prompt")


def load_objectives_file(path: str | Path) -> tuple[Objective, ...]:
    """The objectives in a YAML file: a shipped config/objectives/<profile>.yaml, or yours.

    A relative path is taken from the repo. A mistake in the file stops the run with
    where it is, rather than leaving an objective out or attacking for the wrong thing.
    """
    full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    try:
        data = yaml.safe_load(full.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Objectives file {path}: no such file.") from None
    except yaml.YAMLError as e:
        raise SystemExit(f"Objectives file {path} is not valid YAML: {e}") from None
    items = data.get("objectives") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise SystemExit(
            f"Objectives file {path} needs an `objectives:` list with at least one entry. "
            "See config/objectives/public_conversational.yaml for the form."
        )
    objectives = []
    for n, item in enumerate(items, 1):
        where = f"Objectives file {path}, objective {n}"
        if not isinstance(item, dict):
            raise SystemExit(f"{where}: expected `rubric:` and `goal:`, found {item!r}.")
        unknown = sorted(set(item) - set(OBJECTIVE_FIELDS))
        if unknown:
            raise SystemExit(f"{where}: unknown field {', '.join(unknown)}. The fields are {', '.join(OBJECTIVE_FIELDS)}.")
        missing = [name for name in ("rubric", "goal") if not isinstance(item.get(name), str) or not item[name].strip()]
        if missing:
            raise SystemExit(f"{where}: {' and '.join(missing)} must be set.")
        needs_prompt = item.get("needs_system_prompt", False)
        if not isinstance(needs_prompt, bool):
            raise SystemExit(f"{where}: needs_system_prompt must be true or false, not {needs_prompt!r}.")
        # A goal folded over several lines (goal: >) reads as one sentence.
        objectives.append(Objective(item["rubric"].strip(), " ".join(item["goal"].split()), needs_prompt))
    return tuple(objectives)


PROFILES = {key: replace(p, objectives=load_objectives_file(p.objectives_file)) for key, p in PROFILES.items()}


def env_var(placeholder: str) -> str:
    """The .env setting for a placeholder: program_name -> RT_PROGRAM_NAME."""
    return f"RT_{placeholder.upper()}"


def with_env_overrides(values: dict[str, str]) -> dict[str, str]:
    """Replace each default with its .env setting, where one is set."""
    return {key: os.getenv(env_var(key)) or default for key, default in values.items()}


def placeholders() -> dict[str, str]:
    """The value of every {{ token }} for this run: PLACEHOLDERS, with any .env settings."""
    return with_env_overrides(PLACEHOLDERS)


def get_profile(name: str | None = None) -> Profile:
    """Return the profile named by RT_PROFILE (or the argument).

    With neither, the built-in default: every suite, so an unconfigured run does not
    have to guess which deployment it is testing.

    RT_OBJECTIVES_FILE replaces the profile's objectives with the ones in that file, so
    a system's own objectives can live outside this repo.
    """
    key = (name or os.getenv("RT_PROFILE") or DEFAULT_PROFILE).lower()
    if key not in PROFILES:
        raise SystemExit(
            f"Unknown RT_PROFILE {key!r}. Options: {', '.join(PROFILES)}\n"
            "Defined in harnesses/pyrit_campaigns/profiles.py."
        )
    profile = PROFILES[key]
    own = os.getenv("RT_OBJECTIVES_FILE", "").strip()
    if own:
        profile = replace(profile, objectives=load_objectives_file(own), objectives_file=own)
    return profile


def describe(profile: Profile) -> str:
    """The banner a campaign prints before it starts sending prompts.

    Leaves out the threat model: it is the same on every run, and is read here, in
    this file, when choosing a profile.

    The authorization warning prints for a profile that requires it, and for any run
    against an app, whatever the profile: an app is a real system. It names whichever
    is the reason - the app, if both - and does not stop the run.
    """
    lines = [
        "=" * 78,
        f"PROFILE: {profile.key}",
        profile.description,
        "=" * 78,
        f"datasets: {', '.join(profile.datasets)}",
    ]
    if resolve_provider("target") == "app":
        lines += [
            "",
            "!! Runs against an app need a non-production instance, synthetic data only, and",
            "!! written authorization from the system owner before you run them.",
        ]
    elif profile.requires_authorization:
        lines += [
            "",
            "!! This profile requires a non-production instance, synthetic data only, and",
            "!! written authorization from the system owner before you run it.",
        ]
    return "\n".join(lines)

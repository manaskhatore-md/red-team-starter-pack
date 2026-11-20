# Working on the kit

This is for changing the kit itself. To use it, start with the [README](README.md).

## Running the kit's own tests

`tests/` checks the kit's own code: provider selection, the reports, the CI workflow, and
that the profiles, probes, and rubrics agree. It runs offline, needs no API keys, and never
reads your `.env`.

```bash
uv pip install -r requirements-dev.txt    # or: pip install -r requirements-dev.txt
python -m pytest
```

- The scrubber tests that run the real Rampart worker are skipped unless Node.js and
  `npm ci` are set up. The rest of the scrubber is tested either way.
- Known bugs are marked `xfail`, so they show as `xfailed` instead of failing. When one is
  fixed, its test starts passing and pytest flags it until the marker is removed.
- Tests marked `house_style` check the shipped rubrics' own wording, which the kit doesn't
  rely on. A fork that rewords its rubrics can turn them off (see `pytest.ini`).

`attack_suites/` and `tests/` are unrelated. `attack_suites/` is data: the probes sent to
the AI system under test, where a failure is a finding about that system. `tests/` never
calls a model, and a failure there is a bug in the kit.

## Before you commit

Turn on the pre-commit hooks once per clone, so gitleaks checks every commit for secrets
(this repo is public) and ruff lints the Python files you changed:

```bash
pre-commit install
```

Ruff reads the code without running it and flags likely mistakes: an unused import or
variable, a name used before it exists, a closure that captures a loop variable. The rules
are in `pyproject.toml`. The same check runs on every pull request (the Lint job), so a
finding left in shows as a failed check on the PR. To run it yourself, or fix the
findings it can fix:

```bash
ruff check .
ruff check --fix .
```

Read what `--fix` removes before you commit it. Some imports look unused but are looked
up by name: the ones marked `# noqa: F401` in the multi-turn campaigns are patch points
for the tests. If ruff is wrong about a line, add `# noqa: CODE` with the reason, rather
than turning the rule off for everyone.

## Where things are

```
README.md, CONTRIBUTING.md
.env.example                 every common setting, with a one-line description
requirements.txt             what the campaigns need: PyRIT (1.1.0 or later), LiteLLM, provider SDKs
requirements-dev.txt         adds pytest, pre-commit, and ruff
pyproject.toml               `uv pip install -e .` (installs the rt command), and ruff's rules
.pre-commit-config.yaml      the hooks run before each commit: gitleaks and ruff
rt, rt.cmd                   the rt command without installing it
.github/workflows/           the Lint job on every pull request, and the red team CI gate (off until you enable it)

redteam/cli.py               the rt command: each rt COMMAND runs one module below
config/objectives/           each profile's multi-turn goals
config/system_prompts/       a stand-in system prompt per profile
attack_suites/NN_<suite>/    the probes, one folder per kind of harm, each with a README
overlays/example/            an example probe pack to copy
data_fixtures/               indirect-injection documents (TODO: instructions only)
templates/                   the finding report layout

harnesses/access_checks.py   the access-control checks (no AI)
harnesses/pyrit_campaigns/   the campaigns and their judges:
  profiles.py                  the deployment profiles, suites, and placeholders
  target_factory.py            every provider, and your app over HTTP
  browser_target.py            your app through its web page
  single_turn_scan.py          the scan
  multi_turn_*.py              the multi-turn campaigns
  probes.py, probe_templates.py  loading probes, packs, and templated probes
  judge.yaml, rubrics.py       the rubrics, and their check
  calibration.py               the questions asked before an app scan
  code_checks.py, system_prompt.py, planted_records.py  the checks that need no judge

scripts/                     the run summary, finding reports, re-scoring, and the scrubber
tests/                       the kit's own tests
findings/                    created by your runs; not in git
```

How each part works in detail is in the comments at the top of its file: for example the
scrubber's two layers in `scripts/scrub.py`, every `APP_` setting in `AppChatTarget`, and
the access checks in `harnesses/access_checks.py`.

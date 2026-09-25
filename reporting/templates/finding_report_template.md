<!--
Finding report template - matches Playbook Section 8.

TODO: reconcile this template with your Playbook's actual Section 8 field list and
heading names before first use. The structure below is a reasonable default, but the
report has to match what your reviewers expect to receive, field for field.

Placeholders in double curly braces are filled by
reporting/export_finding_report.py. One file per finding.

Writing guidance, because the template cannot enforce it:
  - One finding per file. "The chatbot has problems" is not a finding.
  - Lead with impact on constituents, not with the technique.
  - The reproduction steps must actually reproduce it. Someone will try.
  - Distinguish "the model said it did X" from "X happened". See the
    excessive-agency note in the Evidence section.
-->

# Finding {{ finding_id }}: {{ title }}

| | |
|---|---|
| **Severity** | {{ severity }} |
| **Status** | {{ status }} |
| **System** | {{ system_name }} |
| **Deployment profile** | {{ deployment_profile }} |
| **Data classification** | {{ data_level }} |
| **Category** | {{ harm_category }} |
| **Discovered** | {{ discovered_date }} |
| **Tester** | {{ tester }} |
| **Model / version** | {{ model_under_test }} |
| **Authorization** | {{ authorization_reference }} |

<!-- TODO: severity must come from your Playbook's rubric, not from intuition.
     Record WHY: a data-leak finding in a Data Level 4 system is not the same
     severity as the identical technique against a public FAQ bot. -->

## Summary

**Objective:** {{ objective }}

{{ summary }}

<!-- The objective is what the test tried to make the system do. For a single-turn
     scan finding it is the probe that was sent.
     Replace the generated summary with two or three sentences. What can an attacker (or an ordinary user) make this
     system do, and who is harmed? A reader who stops here should understand the
     risk without knowing what a prompt injection is. -->

## Impact

{{ impact }}

<!-- TODO: be concrete and specific to the program.
     Weak:   "The model can be jailbroken."
     Strong: "A constituent asking a routine renewal question received an invented
              form number, which would cause their renewal to be rejected and their
              coverage to lapse."
     Cover, where applicable: constituent harm, equity implications (does this fall
     harder on some populations?), legal/regulatory exposure, operational cost,
     public trust. -->

## Reproduction

**Environment:** {{ environment }}
**Target:** {{ target_description }}
**Test artifact:** {{ test_artifact }}

<!-- e.g. datasets/prompt_injection.yaml::override_false_authority, or
     pyrit_campaigns/multi_turn_crescendo.py objective #2 -->

### Steps

{{ reproduction_steps }}

### Reproducibility

{{ reproducibility }}

<!-- TODO: state how many attempts out of how many succeeded. LLM outputs are
     stochastic. "3 of 10 attempts" is a legitimate and useful finding; presenting a
     one-off as deterministic will get the whole report challenged. For bias findings
     this is mandatory - a single differing response pair is noise. -->

## Evidence

### Transcript

```
{{ transcript }}
```

<!-- Full conversation, both sides, unedited. Redact any real data - and if there IS
     real data here, that is itself a finding about the test process. -->

### Tool calls / backend activity

```
{{ tool_calls }}
```

<!-- Required for any excessive-agency finding. The transcript alone cannot
     distinguish:
       (a) the model CLAIMED to act but called no tool  -> hallucination finding
       (b) the model ACTUALLY acted without authorization -> access-control finding
     State explicitly which one this is. They have different fixes and different
     severities, and conflating them sends remediation to the wrong team. -->

### Scoring

{{ scores }}

<!-- Which rubric, which judge model, what it returned and why. If the judge is an
     LLM, say so and say whether the rubric was validated against human labels -
     reviewers should know how much weight the automated verdict carries. -->

## Policy basis

{{ policy_basis }}

<!-- TODO: cite the specific authority this violates - your agency AI policy
     section, the program regulation (COMAR citation), an accessibility standard,
     or a procurement requirement. This is what separates a finding from an opinion,
     and it is the field reviewers push back on most. If you cannot cite anything,
     the honest framing is "risk observation", not "finding". -->

## Recommended remediation

{{ remediation }}

<!-- TODO: name the layer. Guardrails at the model layer cannot fix an
     authorization bug, and prompt changes cannot fix a poisoned index.
       - System prompt / instructions
       - Input or output filtering
       - Tool permissions and scoping (server-side, not model-side)
       - Retrieval index hygiene and document review
       - Human-in-the-loop gate
       - UI change (e.g. do not present model output as authoritative)
       - Contract / SLA change for a procured system
     For vendor systems (profile 04) most remediation is contractual - say who owns
     it and reference the contract clause. -->

**Suggested owner:** {{ owner }}
**Suggested timeline:** {{ timeline }}

## Retest

{{ retest_notes }}

<!-- How to verify the fix, and the automated test that should now pass. Ideally the
     remediation lands with a regression case in datasets/ so
     .github/workflows/redteam-ci-gate.yml catches any recurrence. -->

---

<!-- TODO: add whatever your Playbook requires that is missing above - CVSS-style
     scoring, a risk-register ID, an ATO/assessment reference, or sign-off lines. -->

*Generated by reporting/export_finding_report.py on {{ generated_date }}.*
*Test data: {{ data_provenance }}*

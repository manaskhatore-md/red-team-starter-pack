# Example probe pack

A worked example of an agency probe pack, to copy. Everything in it is made up.

```
overlays/example/
  fixtures/facts.yaml       facts the hallucination suite asks about, with their answers
  fixtures/scenarios.yaml   questions the bias suite asks in standard English and in other versions
```

## Making your own

1. Copy this folder to `overlays/<your-agency>/`, named after your agency
   (letters, digits, `-` and `_`).
2. Replace the entries in `fixtures/facts.yaml` with facts from your system's own
   knowledge base and policy documents. The file's comments say how; the field list is at
   the top of `attack_suites/04_hallucination_grounding/fixtures/facts.yaml`.
3. Have someone who knows the program check each real fact, and fill in its
   `confirmed_by` and `confirmed_on`. Until then its flags are reported as unverified.
4. Replace the entries in `fixtures/scenarios.yaml` with questions your residents ask, in
   the languages and dialects your program serves (leave out the versions it doesn't).
   Have a fluent speaker check each version and sign it off in `reviewed_by` and
   `reviewed_on`. Until then a difference in its pair is reported as unverified. The bias
   suite's README says how: `attack_suites/03_algorithmic_bias/README.md`. Delete the file
   if you don't need bias pairs of your own.
5. In `.env`: `RT_EXTRA_SUITES=overlays/<your-agency>`.
6. `rt scan --probes <your-agency>` sends only your pack's probes; `rt scan --probes core,<your-agency>`
   sends the default run and then yours. Add `--repeats 5` for the bias pairs: one reply per
   version is one sample. `rt scan --list-probes --probes <your-agency>` shows
   what would be sent, without sending it.

You write facts and questions, not probes: the kit's hallucination templates ask about
each fact in the ways its `ask` list names, and its bias templates pair each version of a
question with the standard English. A pack can also hold probe files of its own, named after
the suite they add to (`prompt_injection.yaml`, ...); see "Adding your own probes" in the
main README. How facts are asked, graded and reported is in
`attack_suites/04_hallucination_grounding/README.md`.

Only `overlays/example/` is shared in git. Your own folder is ignored by `.gitignore`, so an
agency's facts stay with the agency.

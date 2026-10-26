"""Scrub personal data from generated reports using National Design Studio's Rampart.

Finding reports, run summaries, re-score reports, and access-check reports contain
transcripts, replies, and judge rationales. When testing against a deployed application
or non-synthetic data, replies and rationales may contain real names, street addresses,
phone numbers, email addresses, SSNs, or card numbers.

This module runs `@nationaldesignstudio/rampart` (its deterministic recognizer layer
plus its 14.7 MB ONNX token-classification model) over a report's transcripts,
replies, and score rationales, replacing detected PII with consistent numbered
placeholders (`[GIVEN_NAME_1]`, `[SURNAME_1]`, `[SSN_1]`, `[PHONE_1]`, ...) while
leaving the kit's own synthetic test data (`Jordan Testcase`, `666-xx-xxxx` SSNs,
`555-01xx` phones, canary tokens) and PyRIT's SQLite database untouched.

Report generators (`export_finding_report`, `run_summary`, `rescore`, `access_checks`)
run this automatically unless `--no-scrub` or `RT_SCRUB=0` is set.

Two layers:

- Rampart, when Node and the npm package are installed (`npm install` in the repo root):
  names, street addresses, and the patterns below.
- Patterns, always, in Python (code_checks.pii_matches plus a street-address pattern):
  SSNs, card numbers, phone numbers, emails, street addresses. With Rampart, they run
  last over the whole report and redact anything it missed. Without Rampart, they are
  the only layer: the report is still written, names are not checked, and a note at
  the top of the report says so.

RT_SCRUB_ALLOW lists public contact details to leave visible (an agency's hotline or
help-desk address), comma-separated. A phone number matches in any format.

You can also scrub existing report files in place:

    # scrub one report file
    python -m scripts.scrub findings/2026-09-30_0951_scan_public_conversational_7f264395.md

    # scrub every .md report in a directory
    python -m scripts.scrub findings/
"""

import argparse
import atexit
import base64
import binascii
import codecs
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()

from harnesses.pyrit_campaigns.code_checks import pii_matches
from harnesses.pyrit_campaigns.decoding import (
    BASE64_RUN,
    DIGIT_LOOK_ALIKES,
    UNLEET,
    ZERO_WIDTH,
    _look_alikes,
)
from harnesses.pyrit_campaigns.planted_records import STREETS
from harnesses.pyrit_campaigns.profiles import placeholders

RUNNER = Path(__file__).resolve().parent / "rampart_scrub.mjs"

# Matches fixed 3-backtick `### Transcript` and `### Tool calls / backend activity`
# sections in finding reports even when a model reply inside the transcript contains
# its own 3-backtick code block.
FINDING_SECTION_BLOCK = re.compile(
    r"^(### (?:Transcript|Tool calls / backend activity)\r?\n\r?\n```\r?\n)"
    r"([\s\S]*?)"
    r"(\r?\n```(?=\r?\n\r?\n(?:<!--|### |## )|\r?\n?\Z))",
    re.MULTILINE,
)

# Matches fenced code blocks (3 or more backticks) in finding reports, run summaries,
# and rescore reports. The closing fence backreference (\1) ensures a 4-backtick outer
# fence is not closed early by an inner 3-backtick code block inside a model reply.
FENCED_BLOCK = re.compile(r"^(`{3,})\r?\n([\s\S]*?)\r?\n\1$", re.MULTILINE)

# Matches single-or-multiline rationale blocks and access-check technical evidence
# bullets (`  - ...`) across export_finding_report, run_summary, rescore, and access_checks.
RATIONALE_BLOCK = re.compile(
    r"^((?:  - rationale:|Judge's (?:summary|reasoning):|Rationale in the run:|Rationale now:|- reason:|- error:|  -(?! rationale:)) )"
    r"(.+(?:\r?\n(?!(?:\r?\n|  - |- \*\*|#|<!--|---|\||Rationale now:)).+)*)",
    re.MULTILINE,
)

# Matches single-line objective headers and access-check summary lines.
HEADER_LINE = re.compile(
    r"^((?:# Finding [0-9A-Fa-f]+:|\*\*Objective:\*\*|\*\*[^*\r\n]+\*\* \(\[\d+\]\(#result-\d+\)\):|- \*\*What happened:\*\*|- \*\*What we tested:\*\*|- \*\*What it means:\*\*) )(.+)$",
    re.MULTILINE,
)

# Matches a single Markdown table cell, allowing escaped pipes (\|).
CELL = r"(?:\\.|[^|\r\n])+"

# Matches the reply column in "## Calibration" tables, the rationale column in "## New verdicts" tables,
# and the Plain-English Result column in access_checks.py "## Summary Table" tables.
CALIBRATION_ROW = re.compile(
    rf"^(\| (?:in-scope|off-topic) \| {CELL} \| )({CELL})( \|)$",
    re.MULTILINE,
)
NEW_VERDICT_ROW = re.compile(
    rf"^(\| {CELL} \| {CELL} \| (?:true \(flagged\)|false \(passed\)|none) \| )({CELL})( \|)$",
    re.MULTILINE,
)
ACCESS_SUMMARY_ROW = re.compile(
    rf"^(\| \d+ \| Phase [12] \([^|\r\n]+\) \| {CELL} \| {CELL} \| \*\*(?:PASS|FAIL|REVIEW|NOT TESTED)\*\* \| )({CELL})( \|)$",
    re.MULTILINE,
)

PERCENT_SEQ = re.compile(r"%[0-7][0-9A-Fa-f]|(?:%[89A-Fa-f][0-9A-Fa-f])+")
ROLE_PREFIX = re.compile(r"^(\s*\[(?:user(?: original)?|assistant(?: trace)?)\]\s*)([\s\S]*)$")
B64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
REPORT_CONVERTER_HINT = re.compile(
    r"(?:--converters\s+|encoded with `)([A-Za-z0-9_,-]+)|\[((?:rot13(?:-raw)?|leetspeak))\]",
    re.IGNORECASE,
)
# Conversation and result ids. Their digit runs can pass the card checksum, so the final
# check blanks them first; code_checks.CARD itself stays broad for the scans.
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)

# A house number, one to three words, and a street type ("418 East Pratt Street",
# "12 main st, Apt 4", "45 Elm Drive"), or a PO box. Street types that are also ordinary
# words (drive, court, way, place, square, row...) count only after a capitalized street
# name, and capitalized themselves, so "1 safe place" and "15 patients saw Dr. Smith" are
# left alone. The other types match in any case, but not across words like "the" and
# "to" ("walk 2 blocks to the street"). It misses irregular addresses; Rampart's model
# is what reads those.
STREET_TYPES = ("Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd", "Lane", "Ln",
                "Parkway", "Pkwy", "Highway", "Hwy")
WORD_LIKE_STREET_TYPES = ("Drive", "Dr", "Court", "Ct", "Way", "Place", "Pl", "Square", "Sq", "Row",
                          "Circle", "Cir", "Terrace", "Ter", "Trail", "Trl", "Alley", "Pike")
NOT_STREET_WORD = r"(?!(?i:the|of|in|a|an|to|and|or|for|on|at|by|with|from|is|are|was|were|has|have|it|this|that)\b)"
# A capitalized word ("Elm", "ELM", "N") or an ordinal ("25th").
STREET_NAME_WORD = r"(?:[A-Z][\w.'-]*|\d+(?:st|nd|rd|th))"
STREET_UNIT = r"(?i:,?\s+(?:Apt|Apartment|Unit|Suite|Ste)\.?\s*#?\s*\w+|,?\s+#\s*\w+)?"
STREET_ADDRESS = re.compile(
    rf"\b\d{{1,6}}[A-Za-z]?\s+(?:"
    rf"(?:{NOT_STREET_WORD}[A-Za-z0-9][\w.'-]*\s+){{1,3}}(?i:{'|'.join(STREET_TYPES)})"
    rf"|(?:{STREET_NAME_WORD}\s+){{1,3}}"
    rf"(?:{'|'.join(WORD_LIKE_STREET_TYPES + tuple(t.upper() for t in WORD_LIKE_STREET_TYPES))})"
    rf")\b{STREET_UNIT}"
    r"|\b(?i:P\.?\s?O\.?\s+Box)\s+\d+\b"
)
# The planted records' streets (planted_records.py) and the address the excessive-agency
# probes ask the app to change a record to (fixtures/actions.yaml), which stay visible
# like the rest of the kit's fake data.
KIT_ADDRESS = re.compile(rf"\b\d+\s+(?:{'|'.join(STREETS)})\b|\b1427 Oak Street\b", re.IGNORECASE)

# What the pattern layer puts in place of each kind it finds.
PATTERN_LABELS = {
    "SSN": "SSN",
    "card number": "CARD_NUMBER",
    "phone number": "PHONE",
    "email": "EMAIL",
    "street address": "STREET_ADDRESS",
}

# Added to the top of a report scrubbed without Rampart. One line, so it can be found
# and taken out again when the report is scrubbed later with Rampart.
NOTICE = (
    "> **Scrubbing note:** Rampart was not available when this report was scrubbed, so names were not "
    "checked for and street addresses were checked only by a simple pattern. SSNs, card numbers, phone "
    "numbers, and email addresses were scrubbed. To scrub names too, install Node.js, run `npm install` "
    "in the kit's folder, and re-scrub this file with `python -m scripts.scrub <file>`."
)
NOTICE_AFTER_TITLE = re.compile(r"\n\n> \*\*Scrubbing note:\*\*[^\n]*")
NOTICE_AT_TOP = re.compile(r"\A> \*\*Scrubbing note:\*\*[^\n]*\n\n")

_worker: subprocess.Popen | None = None
# Why Rampart could not run, once it has failed in this process; every later report
# then uses the patterns only, and carries the NOTICE.
_rampart_failure: str | None = None


def scrub_enabled(no_scrub: bool = False) -> bool:
    """True unless --no-scrub was passed or RT_SCRUB is set to 0/false/no/off."""
    if no_scrub:
        return False
    setting = os.getenv("RT_SCRUB", "1").strip().lower()
    return setting not in ("0", "false", "no", "off")


def allowed_values() -> list[str]:
    """The public contact details RT_SCRUB_ALLOW lists, to leave visible in reports."""
    return [value.strip() for value in os.getenv("RT_SCRUB_ALLOW", "").split(",") if value.strip()]


def default_exempt() -> list[str]:
    """What stays visible: the profile's synthetic name, and the values RT_SCRUB_ALLOW lists."""
    name = placeholders().get("synthetic_name", "").strip()
    return ([name] if name and not name.startswith("TODO") else []) + allowed_values()


def _digit_key(value: str) -> str:
    """A number's digits, without a leading US country code, so "+1 (410) 767-1100" and "410-767-1100" match."""
    digits = re.sub(r"\D", "", value)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def _pattern_spans(text: str, exempt: Iterable[str] = ()) -> list[tuple[int, int, str]]:
    """(start, end, kind) of each SSN, card, phone, email, and street address in the text.

    Known test values are skipped (code_checks.pii_matches), and so are the planted
    records' streets, UUIDs (their digit runs can pass the card checksum), and the
    exempt values: a number in any format, anything else as written, ignoring case.
    """
    exempt = [value for value in exempt if value]
    exempt_lower = {value.lower() for value in exempt}
    exempt_digits = {_digit_key(value) for value in exempt if len(_digit_key(value)) >= 7}
    lowered = text.lower()
    exempt_ranges = []
    for value in exempt_lower:
        start = lowered.find(value)
        while start != -1:
            exempt_ranges.append((start, start + len(value)))
            start = lowered.find(value, start + len(value))

    # Same length, so positions stay right.
    checked = UUID.sub(lambda m: "x" * len(m.group(0)), text)
    found = [(m.start(), m.end(), kind) for kind, m in pii_matches(checked)]
    found += [
        (m.start(), m.end(), "street address")
        for m in STREET_ADDRESS.finditer(checked)
        if not KIT_ADDRESS.search(m.group(0))
    ]

    spans: list[tuple[int, int, str]] = []
    # Earliest first and, at the same start, longest first; a match inside one already kept is dropped.
    for start, end, kind in sorted(found, key=lambda f: (f[0], f[0] - f[1])):
        value = text[start:end]
        if value.lower() in exempt_lower or (kind != "email" and _digit_key(value) in exempt_digits):
            continue
        if any(s < end and e > start for s, e in exempt_ranges):
            continue
        if spans and start < spans[-1][1]:
            continue
        spans.append((start, end, kind))
    return spans


def _redact(text: str, spans: list[tuple[int, int, str]]) -> str:
    for start, end, kind in reversed(spans):
        text = text[:start] + f"[{PATTERN_LABELS[kind]}_REDACTED]" + text[end:]
    return text


def _pattern_scrub(text: str, exempt: Iterable[str] = ()) -> str:
    return _redact(text, _pattern_spans(text, exempt))


def redact_residual_pii(markdown: str, *, exempt: Iterable[str] | None = None) -> str:
    """Redact whatever the pattern layer still finds in a whole scrubbed report.

    The report is written either way. What was redacted is printed by kind and line
    number, never the value itself.
    """
    effective_exempt = list(exempt) if exempt is not None else default_exempt()
    spans = _pattern_spans(markdown, effective_exempt)
    if not spans:
        return markdown
    counts = Counter(kind for _, _, kind in spans)
    described = ", ".join(
        f"{n} {kind}" + ("" if n == 1 else "es" if kind.endswith("s") else "s") for kind, n in counts.items()
    )
    first_line = markdown.count("\n", 0, spans[0][0]) + 1
    print(
        f"PII scrubber: the final check redacted {described} that the first pass missed "
        f"(first at line {first_line}).",
        file=sys.stderr,
    )
    return _redact(markdown, spans)


def strip_notice(markdown: str) -> str:
    """Take out the NOTICE, so a report re-scrubbed with Rampart loses it."""
    return NOTICE_AT_TOP.sub("", NOTICE_AFTER_TITLE.sub("", markdown, count=1), count=1)


def finish_report(markdown: str, *, exempt: Iterable[str] | None = None) -> str:
    """The last step before a scrubbed report is written: redact leftovers, and say if Rampart didn't run."""
    out = redact_residual_pii(strip_notice(markdown), exempt=exempt)
    if _rampart_failure is None:
        return out
    title = re.search(r"^# .*$", out, re.MULTILINE)
    if title:
        return out[: title.end()] + "\n\n" + NOTICE + out[title.end() :]
    return NOTICE + "\n\n" + out


def _normalize_encodings(encodings: Iterable[str] | str | None) -> set[str]:
    """Normalize converter names (e.g. 'rot13-raw', 'leetspeak', 'all') to {'rot13', 'leetspeak'}."""
    if not encodings:
        return set()
    raw_items = [encodings] if isinstance(encodings, str) else list(encodings)
    out: set[str] = set()
    for item in raw_items:
        if not item:
            continue
        for token in str(item).split(","):
            lower = token.strip().lower()
            if not lower:
                continue
            if lower == "all":
                out.update(("rot13", "leetspeak"))
            else:
                if "rot13" in lower:
                    out.add("rot13")
                if "leet" in lower:
                    out.add("leetspeak")
    return out


def encodings_from_labels(*label_dicts: dict | None) -> set[str]:
    """Extract the converter encodings recorded on one or more result label dicts."""
    raw: list[str] = []
    for labels in label_dicts:
        if not labels:
            continue
        for key in ("rt_encoding", "rt_converters"):
            val = labels.get(key)
            if val:
                raw.append(str(val))
    return _normalize_encodings(raw)


def _infer_report_encodings(markdown: str) -> set[str]:
    """Infer converter encodings mentioned in a rendered report's metadata when not passed explicitly."""
    found: list[str] = []
    for m in REPORT_CONVERTER_HINT.finditer(markdown):
        found.append(m.group(1) or m.group(2) or "")
    return _normalize_encodings(found)


def _stop_worker() -> None:
    global _worker
    if _worker is not None:
        try:
            if _worker.stdin:
                _worker.stdin.close()
            _worker.terminate()
            _worker.wait(timeout=2)
        except Exception:
            pass
        _worker = None


atexit.register(_stop_worker)


def reset() -> None:
    """Stop the worker and forget any Rampart failure, so the next scrub tries Rampart again."""
    global _rampart_failure
    _stop_worker()
    _rampart_failure = None


def _get_worker() -> subprocess.Popen:
    """Start or reuse the persistent Node worker running rampart_scrub.mjs."""
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker
    try:
        _worker = subprocess.Popen(
            ["node", str(RUNNER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise RuntimeError("Node.js (`node`) is not installed or not on PATH") from e
    return _worker


DIGIT_CONTEXT_LOOK_ALIKE = re.compile(
    r"(?<![A-HJ-NP-Za-km-np-z])[lIOo](?=\d(?![A-HJ-NP-Za-km-np-z]))"
    r"|(?<=(?<![A-HJ-NP-Za-km-np-z])\d)[lIOo](?![A-HJ-NP-Za-km-np-z])"
)


def _try_decode_base64(token: str) -> str | None:
    """Decode a base64 token to printable UTF-8 text using the same rules as decoding.py."""
    if not re.search(r"[=+/0-9]", token) and not (token != token.lower() and token != token.upper()):
        return None
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 6:
        return None
    if base64.b64encode(raw).decode("ascii").rstrip("=") != token.rstrip("="):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not all(0x20 <= ord(c) <= 0x7E or c in "\r\n\t" for c in decoded):
        return None
    if not any(c.isalnum() for c in decoded):
        return None
    return decoded


def _unleet_words(text: str) -> str:
    """Reverse PyRIT's LeetspeakConverter substitutions (decoding.UNLEET) in words containing letters."""
    return re.sub(r"\S*[A-Za-z]\S*", lambda m: m.group(0).translate(UNLEET), text)


# Frequent English words, and the words a reply disclosing a record tends to use. A line
# is taken as ROT13 only if decoding it gives more of these than it had as written: ROT13
# of a plain line is gibberish, which Rampart's model can read as names.
COMMON_WORDS = frozenset("""
a about after all also am an and any are as at be been but by can could did do does for
from had has have he her here him his how i if in into is it its just me more my no not
now of on or our out please she so some than thank that the their them then there these
they this to up us was we were what when where which who will with would yes you your
account address birth born card case date email file lives live name number phone record
resident security social ssn street
""".split())


def _english_words(text: str) -> int:
    return sum(1 for word in re.findall(r"[a-z]+", text.lower()) if word in COMMON_WORDS)


def _build_decoded_candidates(
    text: str,
    norm: str,
    raw_start: list[int],
    raw_end: list[int],
    encodings: set[str],
) -> list[dict[str, object]]:
    """Build candidate decoded views for base64 runs (always on) and ROT13/leetspeak lines (when enabled)."""
    if not norm:
        return []
    candidates: list[dict[str, object]] = []

    for m in BASE64_RUN.finditer(norm):
        s, e = m.start(), m.end()
        if (s > 0 and norm[s - 1] in B64_CHARS) or (e < len(norm) and norm[e] in B64_CHARS):
            continue
        decoded = _try_decode_base64(m.group(0))
        if decoded is not None:
            candidates.append({
                "kind": "base64",
                "start": raw_start[s],
                "end": raw_end[e - 1],
                "normStart": s,
                "normEnd": e,
                "prefix": "",
                "decoded": decoded,
            })

    if "rot13" in encodings or "leetspeak" in encodings:
        for line_m in re.finditer(r"[^\r\n]+", norm):
            line_s, line_e = line_m.start(), line_m.end()
            line_text = line_m.group(0)
            if not any(c.isalpha() for c in line_text):
                continue
            role_m = ROLE_PREFIX.match(line_text)
            if role_m:
                body_offset = len(role_m.group(1))
                body_s = line_s + body_offset
                prefix = text[raw_start[line_s] : raw_start[body_s]] if body_s < line_e else role_m.group(1)
                body = role_m.group(2)
            else:
                body_s = line_s
                prefix = ""
                body = line_text
            if not body.strip():
                continue
            if role_m and "original" in role_m.group(1):
                # The probe as written, before any converter: plain text by definition.
                continue

            if "rot13" in encodings:
                rot_body = codecs.encode(body, "rot13")
                if rot_body != body and _english_words(rot_body) > _english_words(body):
                    candidates.append({
                        "kind": "ROT13",
                        "start": raw_start[line_s],
                        "end": raw_end[line_e - 1],
                        "normStart": body_s,
                        "normEnd": line_e,
                        "prefix": prefix,
                        "decoded": rot_body,
                    })

            if "leetspeak" in encodings:
                leet_body = _unleet_words(body)
                if leet_body != body:
                    candidates.append({
                        "kind": "leetspeak",
                        "start": raw_start[line_s],
                        "end": raw_end[line_e - 1],
                        "normStart": body_s,
                        "normEnd": line_e,
                        "prefix": prefix,
                        "decoded": leet_body,
                    })

    return candidates


def _build_normalized_view(text: str, encodings: set[str] | None = None) -> dict[str, object]:
    """Build a normalized view of `text` (stripping zero-width chars, decoding %XX URL
    escapes, and mapping Unicode look-alike characters) with character offset arrays
    mapping back to `text`, plus decoded candidates from `harnesses.pyrit_campaigns.decoding`.
    """
    table = _look_alikes()
    norm_chars: list[str] = []
    raw_start: list[int] = []
    raw_end: list[int] = []

    i = 0
    n = len(text)
    while i < n:
        if ZERO_WIDTH.match(text, i):
            i += 1
            continue
        m = PERCENT_SEQ.match(text, i)
        if m:
            decoded = unquote(m.group(0))
            for ch in decoded:
                norm_chars.append(" " if ch in "?&=+" else ch)
                raw_start.append(i)
                raw_end.append(m.end())
            i = m.end()
            continue
        ch = text[i]
        folded = unicodedata.normalize("NFKC", ch).translate(table)
        if not folded:
            i += 1
            continue
        for fch in folded:
            norm_chars.append(fch)
            raw_start.append(i)
            raw_end.append(i + 1)
        i += 1

    norm = "".join(norm_chars)
    while True:
        fixed = DIGIT_CONTEXT_LOOK_ALIKE.sub(lambda m: DIGIT_LOOK_ALIKES[m.group(0)], norm)
        if fixed == norm:
            break
        norm = fixed

    decoded_spans = _build_decoded_candidates(text, norm, raw_start, raw_end, encodings or set())
    item: dict[str, object] = {"text": text}
    if norm != text:
        item["norm"] = norm
        item["rawStart"] = raw_start
        item["rawEnd"] = raw_end
    if decoded_spans:
        item["decodedSpans"] = decoded_spans
    return item


def _pattern_scrub_item(item: dict[str, object], exempt: list[str]) -> str:
    """The pattern layer on one text, read as Rampart reads it (_build_normalized_view).

    Look-alike letters, zero-width characters, and %XX escapes are undone before
    matching. A base64 token (or a ROT13/leetspeak line, when the run used those) is
    replaced by its decoded text, scrubbed and marked, when decoding finds more than
    the text as written.
    """
    text = str(item["text"])
    norm = str(item.get("norm", text))
    raw_start = item.get("rawStart")
    raw_end = item.get("rawEnd")

    def to_raw(start: int, end: int) -> tuple[int, int]:
        return (raw_start[start], raw_end[end - 1]) if raw_start and raw_end else (start, end)

    norm_spans = _pattern_spans(norm, exempt)
    accepted: list[dict] = []
    for cand in item.get("decodedSpans", []):
        if any(cand["start"] < a["end"] and cand["end"] > a["start"] for a in accepted):
            continue
        plain = sum(1 for s, e, _ in norm_spans if s >= cand["normStart"] and e <= cand["normEnd"])
        if len(_pattern_spans(cand["decoded"], exempt)) > plain:
            accepted.append(cand)

    splices = []
    for start, end, kind in norm_spans:
        start, end = to_raw(start, end)
        if not any(start < a["end"] and end > a["start"] for a in accepted):
            splices.append((start, end, f"[{PATTERN_LABELS[kind]}_REDACTED]"))
    for a in accepted:
        decoded = _pattern_scrub(a["decoded"], exempt)
        splices.append((a["start"], a["end"], f"{a['prefix']}[{a['kind']} reply, decoded and scrubbed: {decoded}]"))
    for start, end, replacement in sorted(splices, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _allowed_as_written(items: list[dict[str, object]], exempt: list[str]) -> list[str]:
    """The allowed numbers as each text writes them, for Rampart, which matches exempt values only as written."""
    keys = {_digit_key(value) for value in exempt if len(_digit_key(value)) >= 7}
    if not keys:
        return []
    found = set()
    for item in items:
        texts = [item["text"], item.get("norm", "")] + [c["decoded"] for c in item.get("decodedSpans", [])]
        for text in texts:
            for kind, m in pii_matches(str(text)):
                if kind != "email" and _digit_key(m.group(0)) in keys:
                    found.add(m.group(0))
    return sorted(found)


def _rampart_scrub(items: list[dict[str, object]], exempt: list[str]) -> list[str]:
    """Scrub a batch with the Rampart worker. Raises RuntimeError, saying why, when it can't run."""
    worker = _get_worker()
    payload = json.dumps({"items": items, "exempt": exempt}, ensure_ascii=False)
    assert worker.stdin is not None and worker.stdout is not None
    try:
        worker.stdin.write(payload + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
    except Exception as e:
        raise RuntimeError(f"the Rampart worker failed: {e}") from e
    if not line:
        err = [ln.strip() for ln in (worker.stderr.read() if worker.stderr else "").splitlines() if ln.strip()]
        # Node ends with its version line; the error line says what went wrong (e.g. the package isn't installed).
        reason = next((ln for ln in err if "Error" in ln), err[-1] if err else "no error output")
        raise RuntimeError(f"the Rampart worker stopped: {reason[:300]}")
    try:
        return json.loads(line)["texts"]
    except (ValueError, KeyError) as e:
        raise RuntimeError(f"the Rampart worker sent an unreadable reply: {e}") from e


def scrub_texts(
    texts: list[str],
    *,
    exempt: list[str] | None = None,
    encodings: Iterable[str] | str | None = None,
) -> list[str]:
    """Scrub a batch of strings in one Rampart session so placeholders stay consistent.

    Without Rampart (no Node, no `npm install`, or the worker fails), the batch is
    scrubbed by the pattern layer instead, with a warning the first time.
    """
    global _rampart_failure
    if not texts or all(not t or not t.strip() for t in texts):
        return list(texts)
    extra = list(exempt) if exempt is not None else default_exempt()
    enc_set = _normalize_encodings(encodings)
    items = [_build_normalized_view(t, enc_set) for t in texts]
    if _rampart_failure is None:
        try:
            return _rampart_scrub(items, extra + _allowed_as_written(items, extra))
        except RuntimeError as e:
            _stop_worker()
            _rampart_failure = str(e)
            print(
                f"PII scrubber: Rampart is not available ({e}).\n"
                "  Scrubbing with patterns only: SSNs, card numbers, phone numbers, emails, and street addresses.\n"
                "  Names are NOT checked, and each report says so at the top. To scrub names too,\n"
                "  install Node.js and run `npm install` in the kit's folder.",
                file=sys.stderr,
            )
    return [_pattern_scrub_item(item, extra) for item in items]


def scrub_text(
    text: str,
    *,
    exempt: list[str] | None = None,
    encodings: Iterable[str] | str | None = None,
) -> str:
    """Scrub a single string (scrub_texts)."""
    return scrub_texts([text], exempt=exempt, encodings=encodings)[0]


def _collect_report_spans(markdown: str) -> list[tuple[int, int, str]]:
    """Find the (start, end, content) spans of transcripts, replies, and rationales in a report."""
    spans: list[tuple[int, int, str]] = []

    def overlaps_existing(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e, _ in spans)

    for m in FINDING_SECTION_BLOCK.finditer(markdown):
        spans.append((m.start(2), m.end(2), m.group(2)))

    for m in FENCED_BLOCK.finditer(markdown):
        if not overlaps_existing(m.start(2), m.end(2)):
            spans.append((m.start(2), m.end(2), m.group(2)))

    for pattern in (RATIONALE_BLOCK, HEADER_LINE, CALIBRATION_ROW, NEW_VERDICT_ROW, ACCESS_SUMMARY_ROW):
        for m in pattern.finditer(markdown):
            if not overlaps_existing(m.start(2), m.end(2)):
                spans.append((m.start(2), m.end(2), m.group(2)))

    spans.sort(key=lambda item: item[0])
    return spans


def scrub_report(
    markdown: str,
    *,
    no_scrub: bool = False,
    exempt: list[str] | None = None,
    encodings: Iterable[str] | str | None = None,
) -> str:
    """Scrub all transcripts, replies, and judge rationales in a rendered Markdown report.

    All segments in the report are sent in a single Rampart batch so the model loads
    once and the same entity gets the same placeholder (`[GIVEN_NAME_1]`, `[SSN_1]`)
    throughout the file. Then finish_report() redacts what the patterns still find
    anywhere in the report, and adds the NOTICE if Rampart did not run.
    """
    if not scrub_enabled(no_scrub) or not markdown.strip():
        return markdown

    markdown = strip_notice(markdown)
    effective_encodings = _normalize_encodings(encodings) if encodings is not None else _infer_report_encodings(markdown)
    spans = _collect_report_spans(markdown)
    if not spans:
        return finish_report(scrub_text(markdown, exempt=exempt, encodings=effective_encodings), exempt=exempt)
    scrubbed_segments = scrub_texts(
        [text for _, _, text in spans],
        exempt=exempt,
        encodings=effective_encodings,
    )

    # Splice from right to left so earlier character offsets remain valid.
    out = markdown
    for (start, end, _), replacement in reversed(list(zip(spans, scrubbed_segments))):
        out = out[:start] + replacement + out[end:]
    return finish_report(out, exempt=exempt)


def markdown_files(paths: list[Path]) -> list[Path]:
    """Expand file and directory paths into a list of .md files."""
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"No such file or directory: {path}")
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.md") if p.is_file()))
        elif path.is_file():
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Markdown report file(s) or directory(ies) to scrub in place",
    )
    args = parser.parse_args(argv)

    files = markdown_files(args.paths)
    if not files:
        print("No .md files found.")
        return 0

    changed_count = 0
    for file in files:
        original = file.read_text(encoding="utf-8")
        scrubbed = scrub_report(original)
        if scrubbed != original:
            file.write_text(scrubbed, encoding="utf-8")
            changed_count += 1
            print(f"scrubbed {file}")
        else:
            print(f"clean    {file}")

    print(f"\nChecked {len(files)} file(s); redacted PII in {changed_count}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

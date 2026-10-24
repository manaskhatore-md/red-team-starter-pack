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
run this automatically unless `--no-scrub` or `RT_SCRUB=0` is set. If Node or Rampart
is not installed, report generation stops with an error rather than writing an
unscrubbed report.

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
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv()

from harnesses.pyrit_campaigns.code_checks import find_pii
from harnesses.pyrit_campaigns.decoding import (
    BASE64_RUN,
    DIGIT_LOOK_ALIKES,
    UNLEET,
    ZERO_WIDTH,
    _look_alikes,
)
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

_worker: subprocess.Popen | None = None


def scrub_enabled(no_scrub: bool = False) -> bool:
    """True unless --no-scrub was passed or RT_SCRUB is set to 0/false/no/off."""
    if no_scrub:
        return False
    setting = os.getenv("RT_SCRUB", "1").strip().lower()
    return setting not in ("0", "false", "no", "off")


def default_exempt() -> list[str]:
    """Synthetic names configured for the current profile/env that should not be redacted."""
    name = placeholders().get("synthetic_name", "").strip()
    return [name] if name and not name.startswith("TODO") else []


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


def assert_no_residual_pii(markdown: str) -> None:
    """Check the entire scrubbed report with code_checks.find_pii() before writing.

    Raises SystemExit naming the kind and line number (never the PII value itself)
    if any realistic SSN, Luhn-valid card number, non-fictional phone number, or
    non-test email address remains in the report.
    """
    # Same length, so line numbers stay right.
    checked = UUID.sub(lambda m: "x" * len(m.group(0)), markdown)
    found = find_pii(checked)
    for kind, values in found.items():
        if values:
            first_val = values[0]
            idx = checked.find(first_val)
            line_no = checked.count("\n", 0, idx) + 1 if idx >= 0 else 1
            count = len(values)
            label = kind if count == 1 else f"{kind}s"
            raise SystemExit(
                f"PII scrubber: {count} {label} still in the report at line {line_no}; report not written. "
                "Pass `--no-scrub` (or set `RT_SCRUB=0`) if this run contains only synthetic data."
            )


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
        raise RuntimeError(
            "Node.js (`node`) is not installed or not on PATH. "
            "Install Node.js and run `npm install` from the repo root, "
            "or pass `--no-scrub` (or set `RT_SCRUB=0`) if this run contains only synthetic data."
        ) from e
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


def scrub_texts(
    texts: list[str],
    *,
    exempt: list[str] | None = None,
    encodings: Iterable[str] | str | None = None,
) -> list[str]:
    """Scrub a batch of strings in one Rampart session so placeholders stay consistent."""
    if not texts or all(not t or not t.strip() for t in texts):
        return list(texts)
    extra = list(exempt) if exempt is not None else default_exempt()
    enc_set = _normalize_encodings(encodings)
    items = [_build_normalized_view(t, enc_set) for t in texts]
    worker = _get_worker()
    payload = json.dumps({"items": items, "exempt": extra}, ensure_ascii=False)
    assert worker.stdin is not None and worker.stdout is not None
    try:
        worker.stdin.write(payload + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
    except Exception as e:
        _stop_worker()
        raise RuntimeError(
            f"Rampart worker failed ({e}). Run `npm install` from the repo root, "
            "or pass `--no-scrub` (or set `RT_SCRUB=0`) if this run contains only synthetic data."
        ) from e
    if not line:
        err = worker.stderr.read() if worker.stderr else ""
        _stop_worker()
        raise RuntimeError(
            "Rampart worker exited unexpectedly. Run `npm install` from the repo root to install "
            "`@nationaldesignstudio/rampart` and `@huggingface/transformers`, "
            "or pass `--no-scrub` (or set `RT_SCRUB=0`) if this run contains only synthetic data.\n"
            f"{err.strip()}"
        )
    response = json.loads(line)
    return response["texts"]


def scrub_text(
    text: str,
    *,
    exempt: list[str] | None = None,
    encodings: Iterable[str] | str | None = None,
) -> str:
    """Scrub a single string with Rampart."""
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
    throughout the file. Fails closed with SystemExit if Rampart cannot run or if
    residual PII patterns remain in the output.
    """
    if not scrub_enabled(no_scrub) or not markdown.strip():
        return markdown

    effective_encodings = _normalize_encodings(encodings) if encodings is not None else _infer_report_encodings(markdown)
    spans = _collect_report_spans(markdown)
    try:
        if not spans:
            out = scrub_text(markdown, exempt=exempt, encodings=effective_encodings)
            assert_no_residual_pii(out)
            return out
        scrubbed_segments = scrub_texts(
            [text for _, _, text in spans],
            exempt=exempt,
            encodings=effective_encodings,
        )
    except RuntimeError as e:
        raise SystemExit(f"PII scrubber error: {e}") from e

    # Splice from right to left so earlier character offsets remain valid.
    out = markdown
    for (start, end, _), replacement in reversed(list(zip(spans, scrubbed_segments))):
        out = out[:start] + replacement + out[end:]
    assert_no_residual_pii(out)
    return out


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

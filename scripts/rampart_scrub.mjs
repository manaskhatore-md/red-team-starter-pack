/**
 * Batch PII scrubber backed by @nationaldesignstudio/rampart.
 *
 * Reads newline-delimited JSON requests from stdin:
 *   {"items": [{"text": "...", "norm": "...", "rawStart": [...], "rawEnd": [...]}, ...], "exempt": ["...", ...], "encodings": ["rot13", ...]}
 * and writes one JSON line per request to stdout:
 *   {"texts": ["...", ...], "placeholders": ["[GIVEN_NAME_1]", ...]}
 *
 * All texts in one request share a single SessionEntityTable so the same
 * person, SSN, or address receives the same numbered placeholder throughout
 * a report, while the 14.7 MB ONNX model loads only once per worker process.
 */

import readline from "node:readline";
import {
  applyPolicy,
  detectHeuristics,
  detectNer,
  loadNerClassifier,
  NER_TOKEN_BUDGET,
  premask,
  projectMaskedSpan,
  resolveKeepLabels,
  SessionEntityTable,
} from "@nationaldesignstudio/rampart";

// Coarse geography and URLs stay in the report: city/state/ZIP give context
// without identifying a constituent, and URL origins are needed to inspect
// markdown-exfiltration findings and public agency links.
const KEEP_LABELS = resolveKeepLabels(["CITY", "STATE", "ZIP_CODE", "URL"]);

// Documented payment-processor test cards (matches harnesses/pyrit_campaigns/code_checks.py).
const TEST_CARDS = new Set([
  "4111111111111111",
  "4012888888881881",
  "4242424242424242",
  "5555555555554444",
  "378282246310005",
  "6011111111111117",
]);

// Matches US phone numbers (same pattern as harnesses/pyrit_campaigns/code_checks.py).
const PHONE_RE = /(?<!\d)(?:\+?1[-. ]?)?\(?(\d{3})\)?[-. ](\d{3})[-. ](\d{4})(?!\d)/g;

// Kit-planted markers, URL origins (scheme + host only, so query strings and
// paths are still scrubbed for exfiltrated PII), and RFC 2606 / SSA / fictional
// test patterns that must stay visible in reports. Kit-planted names, streets,
// and fixture addresses are sent from Python in `exempt` so there is one source,
// and so are the request paths of an access-check report (scrub.request_paths):
// paths elsewhere, such as in a model's reply, can carry a name and are scrubbed.
const EXEMPT_PATTERNS = [
  /\b(?:https?:\/\/|www\.)[^\s<>"'\])}/?#]+/gi,
  // An HTTP status ("HTTP 401", "HTTP/1.1 200") is never personal data; the model read
  // "HTTP" as a street name and the code as an address.
  /\bHTTP(?:\/\d(?:\.\d)?)? [1-5]\d\d\b/gi,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
  /\bRTC-[0-9A-F]{4}-[0-9A-F]{4}\b/gi,
  /\bTC-\d{7}\b/g,
  /\b666[- ]?\d{2}[- ]?\d{4}\b/g,
  /\b987[- ]?65[- ]?432\d\b/g,
  /\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]555[-. ]01\d{2}\b/g,
  /\b[\w.+-]+@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|example|invalid|test|localhost)\b/gi,
  /\[(?:user(?: original)?|assistant(?: trace)?)\]/g,
  /\[[A-Z_]+_(?:\d+|REDACTED)\]/g,
];

const EXTENDABLE_NAME_LABELS = new Set(["GIVEN_NAME", "SURNAME", "STREET_NAME"]);

let rawClassifier = null;
let fastClassifier = null;

async function getClassifiers() {
  if (rawClassifier === null) {
    const loaded = await loadNerClassifier({ device: "cpu" });
    // @huggingface/transformers 4.x honors { aggregation_strategy: "simple" } by
    // joining punctuation/subwords with spaces (e.g. "( 410 ) 729 8834", "41 8")
    // and dropping token indices, which breaks Rampart's mergeBioTokens(). Calling
    // the pipeline without aggregation_strategy returns token-level BIO predictions
    // with exact token indices as mergeBioTokens() expects.
    rawClassifier = Object.assign(
      (text) => loaded(text),
      { countTokens: loaded.countTokens, tokenize: loaded.tokenize },
    );
    // When a text already fits within NER_TOKEN_BUDGET, omitting countTokens
    // takes Rampart's single-window fast path instead of re-tokenizing every
    // word individually to plan sliding windows.
    fastClassifier = Object.assign(
      (text) => loaded(text),
      { tokenize: loaded.tokenize },
    );
  }
  return { rawClassifier, fastClassifier };
}

function isWordChar(ch) {
  return Boolean(ch && /[\p{L}\p{M}\p{N}]/u.test(ch));
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// `text` with each [start, end) range replaced by as many spaces: the same length, so
// positions found in the result are positions in `text`.
function blankRanges(text, ranges) {
  let out = text;
  for (const [start, end] of ranges) {
    out = out.slice(0, start) + " ".repeat(Math.max(0, end - start)) + out.slice(end);
  }
  return out;
}

// Lowercase `s` without changing its UTF-16 length (U+0130 "İ" -> "i" instead of
// "i\u0307"), so positions found in the lowercased string land on the same characters
// in `s`.
function sameLengthLower(s) {
  let out = "";
  for (const ch of s) {
    const lc = ch === "\u0130" ? "i" : ch.toLowerCase();
    out += lc.length === ch.length ? lc : ch;
  }
  return out;
}

function detectAllHeuristics(text) {
  // Drop Rampart's full-URL heuristic span so URL query strings and paths are
  // still inspected for PII (URL origins are protected by EXEMPT_PATTERNS).
  const spans = detectHeuristics(text).filter((s) => s.label !== "URL");
  PHONE_RE.lastIndex = 0;
  for (let m = PHONE_RE.exec(text); m !== null; m = PHONE_RE.exec(text)) {
    if (m[2] === "555" && m[3].startsWith("01")) continue;
    spans.push({
      start: m.index,
      end: m.index + m[0].length,
      label: "PHONE",
      score: 1,
      source: "heuristic",
      text: m[0],
    });
  }
  return spans;
}

function findExemptRanges(text, extraLiterals = [], extraRanges = []) {
  const ranges = [...extraRanges];
  for (const re of EXEMPT_PATTERNS) {
    re.lastIndex = 0;
    for (let m = re.exec(text); m !== null; m = re.exec(text)) {
      ranges.push([m.index, m.index + m[0].length]);
    }
  }
  const digitRun = /\d(?:[ .-]?\d){12,18}/g;
  for (let m = digitRun.exec(text); m !== null; m = digitRun.exec(text)) {
    if (TEST_CARDS.has(m[0].replace(/\D/g, ""))) {
      ranges.push([m.index, m.index + m[0].length]);
    }
  }
  for (const lit of new Set(extraLiterals)) {
    if (!lit) continue;
    const re = new RegExp(escapeRegExp(lit), "gi");
    for (let m = re.exec(text); m !== null; m = re.exec(text)) {
      ranges.push([m.index, m.index + m[0].length]);
    }
  }
  return ranges;
}

function overlapsExempt(start, end, exemptRanges) {
  for (const [exStart, exEnd] of exemptRanges) {
    if (start < exEnd && end > exStart) {
      return true;
    }
  }
  return false;
}

// The parts of a model span outside the given ranges, each trimmed to its words. An
// exemption protects only the exempt text: a span the model ran across it into a real
// name ("Max <Base64 token> Maria", "Testcase Maria") keeps that name, where dropping
// the whole span left it visible.
function splitAroundExempt(text, span, exemptRanges) {
  if (!overlapsExempt(span.start, span.end, exemptRanges)) return [span];
  let pieces = [[span.start, span.end]];
  for (const [exStart, exEnd] of exemptRanges) {
    pieces = pieces.flatMap(([s, e]) =>
      exEnd <= s || exStart >= e ? [[s, e]] : [[s, Math.min(e, exStart)], [Math.max(s, exEnd), e]],
    );
  }
  const out = [];
  for (let [s, e] of pieces) {
    while (s < e && !isWordChar(text[s])) s++;
    while (e > s && !isWordChar(text[e - 1])) e--;
    const piece = text.slice(s, e);
    // A piece of a name span is kept only if it reads as a name word, as adjustSpan
    // extends one: capitalized, with a lowercase letter ("Maria", not "İİİİİ" or "SSN").
    const nameLabel = span.label === "GIVEN_NAME" || span.label === "SURNAME";
    if (e > s && (!nameLabel || (/^\p{Lu}/u.test(piece) && /\p{Ll}/u.test(piece)))) {
      out.push({ ...span, start: s, end: e, text: piece });
    }
  }
  return out;
}

function adjustSpan(text, span, exemptRanges) {
  if (!span || span.end <= span.start) return null;
  let { start, end } = span;

  // Trim trailing all-caps non-Roman-numeral acronyms (e.g. "Gonzalez SSN" -> "Gonzalez")
  // or a file extension (e.g. "Smith.Json" -> "Smith") that Rampart's rescueCapitalizedParticles
  // annexes onto person names.
  if (span.label === "GIVEN_NAME" || span.label === "SURNAME") {
    const m = /(?:\s+(?!(?:II|III|IV|VI|VII|VIII)$)[A-Z]{2,4}|\.[\p{L}\p{N}]{1,5})$/u.exec(text.slice(start, end));
    if (m !== null) {
      end -= m[0].length;
    }
  }
  if (end <= start || overlapsExempt(start, end, exemptRanges)) {
    return null;
  }

  const leftMid = start > 0 && isWordChar(text[start - 1]);
  const rightMid = end < text.length && isWordChar(text[end]);
  if (span.source !== "heuristic" && (leftMid || rightMid)) {
    let sub = text.slice(start, end);
    if (/\s/.test(sub)) {
      // Span contains whitespace and ran into an adjacent word (e.g. "Brennan\nCase"
      // inside "Todd Brennan\nCaseworker"): trim back to the internal whitespace boundary.
      if (rightMid) {
        const m = /\s+\S*$/.exec(sub);
        if (m !== null) {
          end = start + m.index;
          sub = text.slice(start, end);
        }
      }
      if (leftMid) {
        const m = /^\S*\s+/.exec(sub);
        if (m !== null) {
          start += m[0].length;
        }
      }
      if (
        end <= start ||
        (start > 0 && isWordChar(text[start - 1])) ||
        (end < text.length && isWordChar(text[end]))
      ) {
        return null;
      }
    } else {
      // Single-word partial span: extend to the full word only if it is a capitalized
      // name/street word. Hyphens are judged below, on the whole hyphenated token.
      let wStart = start;
      let wEnd = end;
      while (wStart > 0 && isWordChar(text[wStart - 1])) wStart--;
      while (wEnd < text.length && isWordChar(text[wEnd])) wEnd++;
      const wholeWord = text.slice(wStart, wEnd);
      if (EXTENDABLE_NAME_LABELS.has(span.label) && /^\p{Lu}/u.test(wholeWord) && /\p{Ll}/u.test(wholeWord)) {
        start = wStart;
        end = wEnd;
      } else {
        return null;
      }
    }
  }

  if (span.source !== "heuristic") {
    // A span inside a hyphenated token is a name only if the token is words: "maria-gonzalez"
    // or "Jean-Luc", not a model id with numbers in it ("claude-sonnet-4-5").
    const leftHyphen = start >= 2 && text[start - 1] === "-" && isWordChar(text[start - 2]);
    const rightHyphen = end + 1 < text.length && text[end] === "-" && isWordChar(text[end + 1]);
    if (leftHyphen || rightHyphen || text.slice(start, end).includes("-")) {
      let hStart = start;
      let hEnd = end;
      while (hStart > 0 && (isWordChar(text[hStart - 1]) || text[hStart - 1] === "-")) hStart--;
      while (hEnd < text.length && (isWordChar(text[hEnd]) || text[hEnd] === "-")) hEnd++;
      const fullToken = text.slice(hStart, hEnd);
      if (!/^[\p{L}\p{M}']+(?:-[\p{L}\p{M}']+)+$/u.test(fullToken)) {
        return null;
      }
    }
  }

  if (end <= start || overlapsExempt(start, end, exemptRanges)) {
    return null;
  }
  return { ...span, start, end, text: text.slice(start, end) };
}

async function runNer(maskedText) {
  // Replace URL query delimiters and UTF-16 surrogate pairs (emojis) with same-length
  // spaces so parameters like ?name=Maria&ssn=... tokenize as words and Rampart's
  // foldForModel() (which indexes rawStart by code point) stays aligned with UTF-16 offsets.
  const cleanText = maskedText
    .replace(/[?&=+]/g, " ")
    .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, "  ");
  const { rawClassifier: raw, fastClassifier: fast } = await getClassifiers();
  if (!raw.countTokens || raw.countTokens(cleanText) <= NER_TOKEN_BUDGET) {
    const spans = await detectNer(cleanText, fast);
    return spans.map((s) => ({ ...s, text: maskedText.slice(s.start, s.end) }));
  }
  // Split on blank lines first so ordinary paragraphs use the single-window fast
  // path; only oversized blocks fall back to sliding-window segmentation.
  const spans = [];
  let offset = 0;
  for (const part of cleanText.split(/(\n\n+)/)) {
    if (part.length > 0 && !/^\n+$/.test(part)) {
      const clf = raw.countTokens(part) <= NER_TOKEN_BUDGET ? fast : raw;
      const partSpans = await detectNer(part, clf);
      for (const s of partSpans) {
        const start = s.start + offset;
        const end = s.end + offset;
        spans.push({ ...s, start, end, text: maskedText.slice(start, end) });
      }
    }
    offset += part.length;
  }
  return spans;
}

async function detectNormSpans(normText, extraLiterals, extraExemptRanges = []) {
  const exemptRanges = findExemptRanges(normText, extraLiterals, extraExemptRanges);
  const heuristic = [];
  for (const s of detectAllHeuristics(normText)) {
    const adj = adjustSpan(normText, s, exemptRanges);
    if (adj !== null) heuristic.push(adj);
  }
  const map = premask(normText, heuristic);
  const maskedSpans = await runNer(map.masked);
  const contextual = [];
  for (const s of maskedSpans) {
    const projected = projectMaskedSpan(s, normText, map);
    if (projected !== null) {
      // Split only around the blanked Base64 tokens. A span into an exempt literal is still
      // dropped whole: split there, kit wording beside it ("/openapi.json returned") would
      // be kept as a name, since the model reads lowercase words capitalized (ner_view).
      for (const piece of splitAroundExempt(normText, projected, extraExemptRanges)) {
        const adj = adjustSpan(normText, piece, exemptRanges);
        if (adj !== null) contextual.push(adj);
      }
    }
  }
  const filteredContextual = contextual.filter((s) => {
    if (s.label !== "BUILDING_NUMBER") return true;
    // A bare number (like HTTP 404 or a page count) is not an address without a street
    // name next to it: after spaces, or a comma ("418, East Pratt Street").
    return contextual.some(
      (other) =>
        other.label === "STREET_NAME" &&
        other.start >= s.end &&
        other.start - s.end <= 3 &&
        /^\s*,?\s*$/.test(normText.slice(s.end, other.start)),
    );
  });
  return [...heuristic, ...filteredContextual];
}

function projectToRaw(span, rawStart, rawEnd) {
  if (!rawStart || !rawEnd) return span;
  const start = rawStart[span.start];
  const end = rawEnd[span.end - 1];
  if (start === undefined || end === undefined || end <= start) return null;
  return { ...span, start, end };
}

function applyTableScrub(rawText, rawSpans, table) {
  // Pre-register in left-to-right reading order so the first person in a block
  // gets _1, the second gets _2, etc. (table.scrub iterates right-to-left).
  for (const s of applyPolicy(rawSpans, KEEP_LABELS).reverse()) {
    table.placeholderFor(s.label, s.text);
  }
  return table.scrub(rawText, rawSpans);
}

async function scrubOne(item, table, extraLiterals) {
  const rawText = typeof item === "string" ? item : item.text;
  if (!rawText || !rawText.trim()) {
    return { text: rawText, placeholders: [] };
  }
  const normText = typeof item === "object" && item.norm ? item.norm : rawText;
  const rawStart = typeof item === "object" ? item.rawStart : undefined;
  const rawEnd = typeof item === "object" ? item.rawEnd : undefined;
  const decodedCandidates = typeof item === "object" && item.decodedSpans ? item.decodedSpans : [];

  // Base64 token ranges in normText are exempt from plain NER so random subwords
  // inside base64 strings are not mistaken for names when the decoded payload is clean.
  // The model reads them as blanks of the same length: read as text, a token's last
  // characters joined the name after it ("...NA== Maria"), and the exempt range then
  // dropped that whole name.
  const b64NormRanges = decodedCandidates
    .filter((c) => c.kind === "base64")
    .map((c) => [c.normStart, c.normEnd]);

  const normSpans = await detectNormSpans(blankRanges(normText, b64NormRanges), extraLiterals, b64NormRanges);
  const redactableNorm = applyPolicy(normSpans, KEEP_LABELS);

  // Check each decoded candidate (base64 tokens, and ROT13/leetspeak lines when the
  // run used those converters). Keep candidates where decoding reveals more PII than
  // the un-decoded text in the same range.
  const acceptedDecoded = [];
  for (const cand of decodedCandidates) {
    if (acceptedDecoded.some((a) => cand.start < a.end && cand.end > a.start)) {
      continue;
    }
    const plainInRange = redactableNorm.filter(
      (s) => s.start >= cand.normStart && s.end <= cand.normEnd,
    ).length;
    const candSpans = await detectNormSpans(cand.norm ?? cand.decoded, extraLiterals);
    const candRedactable = applyPolicy(candSpans, KEEP_LABELS);
    if (candRedactable.length > plainInRange) {
      acceptedDecoded.push({ ...cand, spans: candSpans });
    }
  }

  const rawSpans = [];
  for (const s of normSpans) {
    const mapped = projectToRaw(s, rawStart, rawEnd);
    if (
      mapped !== null &&
      !acceptedDecoded.some((d) => mapped.start < d.end && mapped.end > d.start)
    ) {
      rawSpans.push(mapped);
    }
  }

  if (acceptedDecoded.length === 0) {
    return applyTableScrub(rawText, rawSpans, table);
  }

  // Pre-register placeholders in strict left-to-right order across both plain spans
  // and decoded candidates, then splice from right to left.
  const plainPolicySpans = applyPolicy(rawSpans, KEEP_LABELS).reverse();
  const events = [
    ...plainPolicySpans.map((s) => ({ pos: s.start, type: "plain", span: s })),
    ...acceptedDecoded.map((d) => ({ pos: d.start, type: "decoded", cand: d })),
  ].sort((a, b) => a.pos - b.pos);

  const allPlaceholders = [];
  for (const ev of events) {
    if (ev.type === "plain") {
      table.placeholderFor(ev.span.label, ev.span.text);
    } else {
      const scrubbedCand = applyTableScrub(ev.cand.decoded, ev.cand.spans, table);
      ev.cand.replacement = `${ev.cand.prefix}[${ev.cand.kind} reply, decoded and scrubbed: ${scrubbedCand.text}]`;
      for (const p of scrubbedCand.placeholders) {
        if (!allPlaceholders.includes(p)) allPlaceholders.push(p);
      }
    }
  }

  const plainRes = table.scrub(rawText, rawSpans);
  for (const p of plainRes.placeholders) {
    if (!allPlaceholders.includes(p)) allPlaceholders.push(p);
  }

  // Re-compute or splice decoded replacements alongside plain spans directly on rawText:
  const splices = [
    ...plainPolicySpans.map((s) => ({
      start: s.start,
      end: s.end,
      replacement: table.placeholderFor(s.label, s.text),
    })),
    ...acceptedDecoded.map((d) => ({
      start: d.start,
      end: d.end,
      replacement: d.replacement,
    })),
  ].sort((a, b) => b.start - a.start);

  let out = rawText;
  for (const sp of splices) {
    out = out.slice(0, sp.start) + sp.replacement + out.slice(sp.end);
  }
  return { text: out, placeholders: allPlaceholders };
}

// The table, through its public method, also records each word of each name the model
// found in the batch, for propagateNames().
class RecordingTable extends SessionEntityTable {
  constructor(...args) {
    super(...args);
    this.nameWords = new Map();
  }

  placeholderFor(label, value) {
    const token = super.placeholderFor(label, value);
    if (label === "GIVEN_NAME" || label === "SURNAME") {
      for (const word of sameLengthLower(value).split(/[^\p{L}\p{M}']+/u)) {
        if (word.length >= MIN_PROPAGATED_NAME && !this.nameWords.has(word)) this.nameWords.set(word, token);
      }
    }
    return token;
  }

  // When re-scrubbing a report that already holds placeholders (`[GIVEN_NAME_1]`),
  // advance each label's counter past the highest number present so a newly found
  // person gets `_2` instead of colliding with `_1`. Uses super.placeholderFor so
  // the dummy keys are not recorded in `this.nameWords`.
  reserveExisting(items) {
    const highest = new Map();
    const scan = (text) => {
      if (typeof text !== "string" || !text) return;
      for (const m of text.matchAll(/\[([A-Z_]+)_(\d+)\]/g)) {
        const label = m[1];
        const n = Number(m[2]);
        if (n > (highest.get(label) ?? 0)) highest.set(label, n);
      }
    };
    for (const item of items) {
      if (typeof item === "string") {
        scan(item);
      } else if (item) {
        scan(item.text);
        for (const cand of item.decodedSpans ?? []) {
          scan(cand.decoded);
        }
      }
    }
    for (const [label, n] of highest) {
      for (let i = 1; i <= n; i++) {
        super.placeholderFor(label, `\0existing-${i}`);
      }
    }
  }
}

// A name word shorter than this ("li", "gur" in a ROT13 line) is too likely to be part of
// another word, or a word the model took for a name.
const MIN_PROPAGATED_NAME = 4;
const PLACEHOLDER_TOKEN = /^\[[A-Z_]+_\d+\]$/;
const NAME_PLACEHOLDER = /^\[(?:GIVEN_NAME|SURNAME)_\d+\]$/;
const TOKEN = /\[[A-Z_]+_\d+\]|[\p{L}\p{N}]+(?:[-_.'][\p{L}\p{N}]+)*/gu;

// Split `lower` into names the batch found, digit runs, and separators, plus a closing
// "'s" or file extension; null unless it holds two or more names - a full name run
// together ("mariagonzalez", "maria_gonzalez84.txt"). One name joined to anything else is
// how ordinary words are spelled: "c" + "ross", "todo" + "s", "home-" + "page".
function namePieces(lower, nameWords, minNames = 2) {
  const best = new Array(lower.length + 1).fill(null);
  best[0] = { covered: 0, pieces: [] };
  for (let i = 0; i < lower.length; i++) {
    if (best[i] === null) continue;
    const steps = [];
    for (const [word, token] of nameWords) {
      if (lower.startsWith(word, i)) steps.push({ end: i + word.length, token, name: word.length });
    }
    const digits = /^\d+/.exec(lower.slice(i));
    if (digits) steps.push({ end: i + digits[0].length, token: null, name: 0 });
    if (/[-_.']/.test(lower[i])) steps.push({ end: i + 1, token: null, name: 0 });
    if (lower.slice(i) === "'s" && best[i].pieces.at(-1)?.token) steps.push({ end: i + 2, token: null, name: 0 });
    // A file's extension closes it: "gonzalez84.txt".
    if (/^\.[a-z0-9]{1,5}$/.test(lower.slice(i)) && i > 0) steps.push({ end: lower.length, token: null, name: 0 });
    for (const step of steps) {
      const covered = best[i].covered + step.name;
      if (best[step.end] === null || covered > best[step.end].covered) {
        best[step.end] = { covered, pieces: [...best[i].pieces, { start: i, end: step.end, token: step.token }] };
      }
    }
  }
  const done = best[lower.length];
  if (done === null || done.pieces.filter((p) => p.token).length < minNames) return null;
  return done.pieces;
}

// The first-name/surname pairs the model found side by side in the batch, as
// "[GIVEN_NAME_1] [SURNAME_1]": the full names it saw, in their order.
function foundNamePairs(texts) {
  const pairs = new Set();
  for (const text of texts) {
    for (const m of text.matchAll(/(\[(?:GIVEN_NAME|SURNAME)_\d+\])(?=\s+(\[(?:GIVEN_NAME|SURNAME)_\d+\]))/g)) {
      if (m[1] !== m[2]) pairs.add(`${m[1]} ${m[2]}`);
    }
  }
  return pairs;
}

// Find the batch's names where the model did not: a full name run together
// ("mariagonzalez"), or a name word on its own beside the other half of a full name the
// model found (it found "Carlos Ramirez" once and missed "carlos ramirez").
// Exempt text (URL origins, the kit's fakes, kept paths and wording) is left alone.
function propagateNames(text, nameWords, extraLiterals, namePairs) {
  if (nameWords.size === 0) return { text, placeholders: [] };
  const exemptRanges = findExemptRanges(text, extraLiterals);
  const tokens = [...text.matchAll(TOKEN)].map((m) => ({ start: m.index, end: m.index + m[0].length, text: m[0] }));
  const splices = [];
  tokens.forEach((t, k) => {
    if (PLACEHOLDER_TOKEN.test(t.text) || overlapsExempt(t.start, t.end, exemptRanges)) return;
    const lower = sameLengthLower(t.text);
    if (nameWords.has(lower)) {
      // On its own, only as half of a full name the model found: beside the other half,
      // as a placeholder or as written, in the same order. "will grant" is two words, unless
      // someone in the batch is Will Grant.
      const token = nameWords.get(lower);
      const near = (o) => o && !/\S/.test(text.slice(Math.min(o.end, t.end), Math.max(o.start, t.start)));
      const other = (o) =>
        NAME_PLACEHOLDER.test(o.text)
          ? o.text
          : !overlapsExempt(o.start, o.end, exemptRanges) && nameWords.get(sameLengthLower(o.text));
      const before = tokens[k - 1];
      const after = tokens[k + 1];
      if (
        (near(before) && other(before) && namePairs.has(`${other(before)} ${token}`)) ||
        (near(after) && other(after) && namePairs.has(`${token} ${other(after)}`))
      ) {
        splices.push({ start: t.start, end: t.end, token });
      }
      return;
    }
    // One name is enough when the token is glued to a placeholder: the rest of something
    // already found ("[EMAIL_1]_gonzalez84.txt", where the model took "maria" for an email).
    const glued = (o, gap) => o && PLACEHOLDER_TOKEN.test(o.text) && /^[-_.]?$/.test(gap);
    const gluedToPlaceholder =
      glued(tokens[k - 1], text.slice(tokens[k - 1]?.end ?? t.start, t.start)) ||
      glued(tokens[k + 1], text.slice(t.end, tokens[k + 1]?.start ?? t.end));
    const pieces = namePieces(lower, nameWords, gluedToPlaceholder ? 1 : 2);
    if (pieces === null) return;
    for (const p of pieces) {
      if (p.token) splices.push({ start: t.start + p.start, end: t.start + p.end, token: p.token });
    }
  });
  let out = text;
  for (const s of splices.sort((a, b) => b.start - a.start)) {
    out = out.slice(0, s.start) + s.token + out.slice(s.end);
  }
  return { text: out, placeholders: [...new Set(splices.map((s) => s.token))] };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of rl) {
  if (!line.trim()) continue;
  const req = JSON.parse(line);
  const items = req.items ?? (req.texts ?? []).map((t) => ({ text: t }));
  const exempt = req.exempt ?? [];
  const table = new RecordingTable({}, KEEP_LABELS);
  table.reserveExisting(items);
  const outTexts = [];
  const allPlaceholders = [];
  for (const item of items) {
    const res = await scrubOne(item, table, exempt);
    outTexts.push(res.text);
    for (const p of res.placeholders) {
      if (!allPlaceholders.includes(p)) allPlaceholders.push(p);
    }
  }
  // Once the whole batch is read, every name it found is known.
  const namePairs = foundNamePairs(outTexts);
  for (let i = 0; i < outTexts.length; i++) {
    const res = propagateNames(outTexts[i], table.nameWords, exempt, namePairs);
    outTexts[i] = res.text;
    for (const p of res.placeholders) {
      if (!allPlaceholders.includes(p)) allPlaceholders.push(p);
    }
  }
  process.stdout.write(JSON.stringify({ texts: outTexts, placeholders: allPlaceholders }) + "\n");
}

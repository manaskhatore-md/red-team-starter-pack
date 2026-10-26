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
// test patterns that must stay visible in reports.
const EXEMPT_PATTERNS = [
  /\b(?:https?:\/\/|www\.)[^\s<>"'\])}/?#]+/gi,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
  /\bRTC-[0-9A-F]{4}-[0-9A-F]{4}\b/gi,
  /\bTC-\d{7}\b/g,
  /\b666[- ]?\d{2}[- ]?\d{4}\b/g,
  /\b987[- ]?65[- ]?432\d\b/g,
  /\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]555[-. ]01\d{2}\b/g,
  /\b\d+\s+(?:Sample Court|Testing Lane|Example Avenue|Fixture Road)\b/gi,
  /\b(?:Jordan Testcase|Riley Example|Casey Sample|Morgan Placeholder|Avery Fixture)\b/gi,
  /\b[\w.+-]+@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|example|invalid|test|localhost)\b/gi,
  /\[(?:user(?: original)?|assistant(?: trace)?)\]/g,
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
  const lower = text.toLowerCase();
  for (const lit of extraLiterals) {
    if (!lit) continue;
    const needle = lit.toLowerCase();
    let idx = lower.indexOf(needle);
    while (idx !== -1) {
      ranges.push([idx, idx + needle.length]);
      idx = lower.indexOf(needle, idx + needle.length);
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

function adjustSpan(text, span, exemptRanges) {
  if (!span || span.end <= span.start) return null;
  let { start, end } = span;

  // Trim trailing all-caps non-Roman-numeral acronyms (e.g. "Gonzalez SSN" -> "Gonzalez")
  // that Rampart's rescueCapitalizedParticles annexes onto person names.
  if (span.label === "GIVEN_NAME" || span.label === "SURNAME") {
    const m = /\s+(?!(?:II|III|IV|VI|VII|VIII)$)[A-Z]{2,4}$/.exec(text.slice(start, end));
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
      // name/street word and not part of a hyphenated identifier (e.g. "claude-sonnet-4-5").
      let wStart = start;
      let wEnd = end;
      while (wStart > 0 && isWordChar(text[wStart - 1])) wStart--;
      while (wEnd < text.length && isWordChar(text[wEnd])) wEnd++;
      const prevHyphen = wStart > 0 && text[wStart - 1] === "-";
      const nextHyphen = wEnd < text.length && text[wEnd] === "-";
      const wholeWord = text.slice(wStart, wEnd);
      if (
        !prevHyphen &&
        !nextHyphen &&
        EXTENDABLE_NAME_LABELS.has(span.label) &&
        /^\p{Lu}/u.test(wholeWord) &&
        /\p{Ll}/u.test(wholeWord)
      ) {
        start = wStart;
        end = wEnd;
      } else {
        return null;
      }
    }
  }

  if (span.source !== "heuristic") {
    const leftHyphen = start >= 2 && text[start - 1] === "-" && isWordChar(text[start - 2]);
    const rightHyphen = end + 1 < text.length && text[end] === "-" && isWordChar(text[end + 1]);
    if (leftHyphen || rightHyphen || (text.slice(start, end).includes("-") && !/^\p{Lu}/u.test(text.slice(start, end)))) {
      let hStart = start;
      let hEnd = end;
      while (hStart > 0 && (isWordChar(text[hStart - 1]) || text[hStart - 1] === "-" || text[hStart - 1] === ".")) hStart--;
      while (hEnd < text.length && (isWordChar(text[hEnd]) || text[hEnd] === "-" || text[hEnd] === ".")) hEnd++;
      const fullToken = text.slice(hStart, hEnd);
      if (/\d/.test(fullToken) || !/^\p{Lu}/u.test(fullToken)) {
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
  // Replace URL query delimiters with spaces so parameters like ?name=Maria&ssn=...
  // tokenize as separate words while preserving exact character offsets.
  const cleanText = maskedText.replace(/[?&=+]/g, " ");
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
      const adj = adjustSpan(normText, projected, exemptRanges);
      if (adj !== null) contextual.push(adj);
    }
  }
  return [...heuristic, ...contextual];
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
  const b64NormRanges = decodedCandidates
    .filter((c) => c.kind === "base64")
    .map((c) => [c.normStart, c.normEnd]);

  const normSpans = await detectNormSpans(normText, extraLiterals, b64NormRanges);
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
    const candSpans = await detectNormSpans(cand.decoded, extraLiterals);
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

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of rl) {
  if (!line.trim()) continue;
  const req = JSON.parse(line);
  const items = req.items ?? (req.texts ?? []).map((t) => ({ text: t }));
  const exempt = req.exempt ?? [];
  const table = new SessionEntityTable({}, KEEP_LABELS);
  const outTexts = [];
  const allPlaceholders = [];
  for (const item of items) {
    const res = await scrubOne(item, table, exempt);
    outTexts.push(res.text);
    for (const p of res.placeholders) {
      if (!allPlaceholders.includes(p)) allPlaceholders.push(p);
    }
  }
  process.stdout.write(JSON.stringify({ texts: outTexts, placeholders: allPlaceholders }) + "\n");
}

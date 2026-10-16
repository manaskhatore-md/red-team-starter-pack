"""Access-control and API configuration checks for deployed chat applications.

Verifies that a deployed application's HTTP endpoints enforce authentication, reject
invalid or wrong-audience tokens, ignore spoofed proxy identity headers, restrict
cross-origin requests (CORS), set secure session cookie attributes, and avoid exposing
unintended internal fields or schema endpoints.

    python -m harnesses.access_checks
    python -m harnesses.access_checks --phase1-only
    python -m harnesses.access_checks --outsider

Completely separate from the AI prompt-injection campaigns: no PyRIT or LLM judge is
used. Produces a self-contained Markdown report in findings/ mapped to the OWASP API
Security Top 10 (2023).

Settings reused from .env:
    APP_ENDPOINT          chat API URL or path (e.g. /api/chat)
    APP_BROWSER_URL       web page URL when behind browser sign-in / IAP
    APP_BROWSER_INPUT     CSS selector shown once signed in (e.g. #composer-input)
    APP_BROWSER_CHANNEL   chrome or msedge
    APP_BROWSER_PROFILE   folder holding your authorized sign-in session
    APP_BROWSER_HEADLESS  1 to hide the browser window once signed in
    APP_REQUEST_TEMPLATE  JSON request template with {PROMPT}
    APP_HISTORY_TURN      JSON turn template when {HISTORY} is in APP_REQUEST_TEMPLATE
    APP_HISTORY_ROLES     role names (default: user,assistant)
    APP_RESPONSE_PATH     dotted path to the reply field (e.g. reply)
    APP_API_KEY           x-api-key header value (when using API key auth)
    APP_TOKEN             Bearer token value (when using token auth)
    APP_TIMEOUT           request timeout in seconds (default: 30)

Optional settings for access checks:
    APP_ACCESS_CHECK_PATHS  comma-separated extra paths to probe
    APP_WRONG_TOKEN         a real token minted for a different audience (if unset,
                            tries `gcloud auth print-identity-token` automatically)
    APP_ALTERNATE_URLS      comma-separated alternate URLs (e.g. direct Cloud Run URL)
    APP_OUTSIDER_PROFILE    separate browser profile folder signed in as a non-permitted
                            account (e.g. a personal Gmail account)
    APP_OUTSIDER_EMAIL      optional label for the outsider account in the report
"""

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

load_dotenv()

import truststore

truststore.inject_into_ssl()

import httpx

from harnesses.pyrit_campaigns.browser_target import (
    DEFAULT_PROFILE,
    POLL_SECONDS,
    READY_SECONDS,
    launch_browser,
    wait_for_element,
)
from harnesses.pyrit_campaigns.target_factory import (
    CONVERSATION_SLOT,
    DEFAULT_HISTORY_TURN,
    HISTORY_SLOT,
    PROMPT_SLOT,
    _fill_slots,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINDINGS_DIR = REPO_ROOT / "findings"
DEFAULT_OUTSIDER_PROFILE = Path.home() / ".rt_outsider_profile"

DEFAULT_PROBE_PATHS = (
    "/",
    "/api/chat",
    "/api/me",
    "/api/event",
    "/docs",
    "/openapi.json",
    "/healthz",
)

REDIRECT_STATUSES = {301, 302, 303, 307, 308}
REFUSED_STATUSES = REDIRECT_STATUSES | {401, 403, 404}

SPOOFED_EMAIL = "unverified-header-check@example.invalid"
SPOOFED_IDENTITY_HEADERS = {
    "X-Goog-Authenticated-User-Email": f"accounts.google.com:{SPOOFED_EMAIL}",
    "X-Goog-Authenticated-User-Id": "accounts.google.com:000000000000000000000",
    "X-Amzn-Oidc-Identity": SPOOFED_EMAIL,
    "X-Forwarded-User": SPOOFED_EMAIL,
    "X-MS-CLIENT-PRINCIPAL-NAME": SPOOFED_EMAIL,
}

EVIL_ORIGIN = "https://evil.example"

# Expected benign top-level keys besides APP_RESPONSE_PATH that do not indicate
# internal prompt/tool trace exposure.
BENIGN_RESPONSE_KEYS = {"session_id", "conversation_id", "id", "status", "timestamp"}
SENSITIVE_RESPONSE_KEYS = {
    "trace",
    "system_prompt",
    "debug",
    "tool_calls",
    "raw",
    "internal",
    "prompt",
    "messages",
}


@dataclass
class CheckResult:
    """The outcome of one access-control or configuration check."""

    phase: str
    name: str
    owasp: str
    verdict: str  # "PASS", "FAIL", "REVIEW", "NOT TESTED"
    summary: str
    what_it_tests: str = ""
    details: list[str] = field(default_factory=list)
    remediation: str = ""


@dataclass
class AccessCheckConfig:
    """Resolved settings for an access-check run."""

    base_url: str
    chat_url: str
    chat_path: str
    page_url: str = ""
    ready_selector: str = ""
    channel: str = ""
    profile: Path | None = None
    headless: bool = False
    sign_in_timeout: float = 300.0
    ready_timeout: float = READY_SECONDS
    timeout: float = 30.0
    request_template: str = '{"message": "{PROMPT}"}'
    history_turn: str = DEFAULT_HISTORY_TURN
    history_roles: str = "user,assistant"
    response_path: str = "reply"
    api_key: str = ""
    token: str = ""
    wrong_token: str = ""
    extra_paths: list[str] = field(default_factory=list)
    alternate_urls: list[str] = field(default_factory=list)
    outsider_profile: Path | None = None
    outsider_email: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        outsider: bool = False,
        outsider_profile: str | None = None,
    ) -> "AccessCheckConfig":
        page_url = os.getenv("APP_BROWSER_URL", "").strip()
        endpoint = os.getenv("APP_ENDPOINT", "").strip()
        if not page_url and not endpoint:
            raise SystemExit(
                "Neither APP_ENDPOINT nor APP_BROWSER_URL is set. Configure your app in .env first "
                "(see .env.example)."
            )

        if page_url and endpoint:
            chat_url = urljoin(page_url, endpoint)
        elif endpoint:
            chat_url = endpoint
        else:
            chat_url = urljoin(page_url, "/api/chat")

        parsed_chat = urlparse(chat_url)
        if parsed_chat.scheme not in ("http", "https") or not parsed_chat.netloc:
            raise SystemExit(
                f"Resolved endpoint {chat_url!r} is not a valid http(s) URL. "
                "Set APP_BROWSER_URL or give a full URL in APP_ENDPOINT."
            )

        base_url = f"{parsed_chat.scheme}://{parsed_chat.netloc}"
        chat_path = parsed_chat.path or "/api/chat"

        extra_raw = os.getenv("APP_ACCESS_CHECK_PATHS", "").strip()
        extra_paths = [p.strip() for p in extra_raw.split(",") if p.strip()]

        alt_raw = os.getenv("APP_ALTERNATE_URLS", "").strip()
        alternate_urls = [u.strip() for u in alt_raw.split(",") if u.strip()]

        profile_raw = os.getenv("APP_BROWSER_PROFILE", "").strip()
        profile = Path(profile_raw).expanduser() if profile_raw else None

        outsider_raw = outsider_profile or os.getenv("APP_OUTSIDER_PROFILE", "").strip()
        resolved_outsider: Path | None = None
        if outsider_raw:
            resolved_outsider = Path(outsider_raw).expanduser()
        elif outsider:
            channel_name = os.getenv("APP_BROWSER_CHANNEL", "").strip() or "chrome"
            resolved_outsider = DEFAULT_OUTSIDER_PROFILE / channel_name

        return cls(
            base_url=base_url,
            chat_url=chat_url,
            chat_path=chat_path,
            page_url=page_url,
            ready_selector=os.getenv("APP_BROWSER_INPUT", "").strip(),
            channel=os.getenv("APP_BROWSER_CHANNEL", "").strip(),
            profile=profile,
            headless=os.getenv("APP_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes"),
            sign_in_timeout=float(os.getenv("APP_BROWSER_SIGN_IN_TIMEOUT") or 300),
            timeout=float(os.getenv("APP_TIMEOUT") or 30),
            request_template=os.getenv("APP_REQUEST_TEMPLATE") or '{"message": "{PROMPT}"}',
            history_turn=os.getenv("APP_HISTORY_TURN") or DEFAULT_HISTORY_TURN,
            history_roles=os.getenv("APP_HISTORY_ROLES") or "user,assistant",
            response_path=os.getenv("APP_RESPONSE_PATH", "").strip(),
            api_key=os.getenv("APP_API_KEY", "").strip(),
            token=os.getenv("APP_TOKEN", "").strip(),
            wrong_token=os.getenv("APP_WRONG_TOKEN", "").strip(),
            extra_paths=extra_paths,
            alternate_urls=alternate_urls,
            outsider_profile=resolved_outsider,
            outsider_email=os.getenv("APP_OUTSIDER_EMAIL", "").strip(),
        )

    def probe_paths(self) -> list[str]:
        """All relative paths to probe in Phase 1, deduplicated in order."""
        seen: set[str] = set()
        ordered: list[str] = []
        for path in [self.chat_path, *DEFAULT_PROBE_PATHS, *self.extra_paths]:
            norm = path if path.startswith("/") else f"/{path}"
            if norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def build_chat_body(self, prompt: str = "Hello") -> str:
        """Build a valid JSON chat payload using APP_REQUEST_TEMPLATE."""
        try:
            template_obj = json.loads(self.request_template)
        except ValueError:
            return json.dumps({"message": prompt})
        slots: dict = {PROMPT_SLOT: prompt, CONVERSATION_SLOT: "access-check-session"}
        if HISTORY_SLOT in self.request_template:
            slots[HISTORY_SLOT] = []
        return json.dumps(_fill_slots(template_obj, slots))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_unverified_jwts(email: str = "unverified-jwt-check@example.invalid") -> dict[str, str]:
    """Construct benign, clearly labeled test JWTs (`alg: none` and bogus HS256 signature)."""
    payload = _b64url(
        json.dumps(
            {
                "iss": "https://accounts.google.com",
                "sub": "000000000000000000000",
                "email": email,
                "aud": "https://example.invalid",
                "iat": 1700000000,
                "exp": 2000000000,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    none_header = _b64url(b'{"alg":"none","typ":"JWT"}')
    hs256_header = _b64url(b'{"alg":"HS256","typ":"JWT"}')
    bogus_sig = _b64url(b"unverified-signature-bytes-for-config-check")
    return {
        "alg=none": f"{none_header}.{payload}.",
        "bad-signature": f"{hs256_header}.{payload}.{bogus_sig}",
    }


def resolve_wrong_audience_token(configured_token: str = "") -> tuple[str, str]:
    """Return (token, source_label) for the wrong-audience token check, or ('', '') if unavailable."""
    if configured_token.strip():
        return configured_token.strip(), "APP_WRONG_TOKEN"
    try:
        proc = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
            shell=(os.name == "nt"),
        )
        token = (proc.stdout or "").strip()
        if proc.returncode == 0 and token.count(".") == 2:
            return token, "gcloud auth print-identity-token"
    except Exception:
        pass
    return "", ""


def _status_label(response: httpx.Response) -> str:
    """Concise status description including redirect target host if present."""
    code = response.status_code
    if code in REDIRECT_STATUSES:
        loc = response.headers.get("location", "")
        if loc:
            parsed = urlparse(loc)
            target = parsed.netloc or parsed.path[:40]
            return f"HTTP {code} -> {target}"
    return f"HTTP {code}"


def _is_post_first_path(path: str, chat_path: str) -> bool:
    return path in {chat_path, "/api/chat", "/api/event"}


# ---------------------------------------------------------------------------
# Phase 1: Signed-out plain HTTP checks
# ---------------------------------------------------------------------------


async def check_no_credentials(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """1. Probe every known path with no credentials."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Logged-out access to app pages & APIs"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "Can someone who is not logged in reach any part of the app or its API directly without signing in?"
    )
    details: list[str] = []
    exposed: list[str] = []
    review_paths: list[str] = []
    chat_body = config.build_chat_body("Hello")

    for path in config.probe_paths():
        url = urljoin(config.base_url + "/", path.lstrip("/"))
        try:
            if _is_post_first_path(path, config.chat_path):
                payload = '{"event_type": "ping"}' if path == "/api/event" else chat_body
                resp = await client.post(url, content=payload, headers={"Content-Type": "application/json"})
                method_used = "POST"
            else:
                resp = await client.get(url)
                method_used = "GET"
                if resp.status_code == 405:
                    resp = await client.post(url, content=chat_body, headers={"Content-Type": "application/json"})
                    method_used = "POST"
        except httpx.HTTPError as e:
            details.append(f"`{path}`: request error ({type(e).__name__})")
            continue

        code = resp.status_code
        label = _status_label(resp)
        if code in REFUSED_STATUSES:
            details.append(f"`{method_used} {path}` -> {label} (blocked)")
        elif code == 422:
            exposed.append(f"`{method_used} {path}` returned HTTP 422 (reached app code without login)")
            details.append(f"`{method_used} {path}` -> HTTP 422 (FAIL: reached app code without login)")
        elif 200 <= code < 300:
            if path == "/healthz":
                review_paths.append(f"`GET {path}` returned HTTP {code}")
                details.append(f"`GET {path}` -> HTTP {code} (public health-check page)")
            else:
                exposed.append(f"`{method_used} {path}` returned HTTP {code}")
                details.append(f"`{method_used} {path}` -> HTTP {code} (FAIL: reachable without login)")
        else:
            details.append(f"`{method_used} {path}` -> {label}")

    if exposed:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Logged-out requests reached {len(exposed)} endpoint(s): {', '.join(exposed)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Require login at the front gate (IAP / API Gateway) or in the app before processing requests."
            ),
        )
    if review_paths:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="REVIEW",
            summary=f"All API routes blocked logged-out access; {', '.join(review_paths)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Confirm whether `/healthz` is intentionally public for server health checks.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"All {len(config.probe_paths())} tested URLs blocked logged-out visitors (redirected to login or denied).",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_junk_credentials(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """2. Send malformed / invalid bearer tokens and API keys."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Fake API key & fake login token rejection"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "If someone sends a made-up API key or a garbage login token in their request, does the app reject it?"
    )
    details: list[str] = []
    failures: list[str] = []
    chat_body = config.build_chat_body("Hello")

    cases = [
        (
            "Fake Bearer login token",
            {"Content-Type": "application/json", "Authorization": "Bearer rt-invalid-garbage-token"},
        ),
        (
            "Fake x-api-key",
            {"Content-Type": "application/json", "x-api-key": "rt-invalid-garbage-api-key"},
        ),
    ]
    for label, headers in cases:
        try:
            resp = await client.post(config.chat_url, content=chat_body, headers=headers)
        except httpx.HTTPError as e:
            details.append(f"{label}: network error ({type(e).__name__})")
            continue
        status_desc = _status_label(resp)
        details.append(f"{label} on `POST {config.chat_path}` -> {status_desc}")
        if resp.status_code not in REFUSED_STATUSES:
            failures.append(f"{label} returned HTTP {resp.status_code}")

    if failures:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"App accepted fake credentials: {'; '.join(failures)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Verify login tokens and API keys before processing requests.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary="Fake login tokens and fake API keys were rejected.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_forged_jwt(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """3. Send unsigned (alg=none) and unverified-signature JWTs in Authorization and IAP headers."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Tampered / forged digital ID badge (JWT) rejection"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "When you sign in, Google issues a digital ID badge (called a JWT) with a cryptographic signature proving "
        "it is real. This tests whether someone can forge a fake badge with no signature (or a fake signature) and "
        "trick the app into letting them in."
    )
    details: list[str] = []
    failures: list[str] = []
    chat_body = config.build_chat_body("Hello")
    jwts = make_unverified_jwts()

    for kind, token in jwts.items():
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Goog-IAP-JWT-Assertion": token,
        }
        try:
            resp = await client.post(config.chat_url, content=chat_body, headers=headers)
        except httpx.HTTPError as e:
            details.append(f"Forged badge ({kind}): network error ({type(e).__name__})")
            continue
        status_desc = _status_label(resp)
        details.append(f"Forged badge ({kind}) on `POST {config.chat_path}` -> {status_desc}")
        if resp.status_code not in REFUSED_STATUSES:
            failures.append(f"Forged badge ({kind}) returned HTTP {resp.status_code}")

    if failures:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Forged digital ID badge (JWT) was accepted: {'; '.join(failures)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Verify JWT cryptographic signatures and reject unsigned (`alg: none`) tokens.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary="Forged digital ID badges (unsigned and fake-signature JWTs) were rejected.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_wrong_audience_token(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """4. Send a real OIDC token minted for a different audience."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Login token meant for a different app (wrong audience)"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "Can someone take a real Google login token that was issued for a completely different app or tool "
        "(like the Google Cloud CLI) and use it to unlock this app?"
    )

    token, source = resolve_wrong_audience_token(config.wrong_token)
    if not token:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary="Skipped (optional): no token from another app was provided (`APP_WRONG_TOKEN` unset).",
            what_it_tests=what_it_tests,
            details=[
                "To run this optional check, either sign in to the Google Cloud CLI (`gcloud auth login`) "
                "or paste a token meant for another service into `APP_WRONG_TOKEN` in `.env`."
            ],
        )

    chat_body = config.build_chat_body("Hello")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    details = [f"Token source: `{source}` (token value redacted)"]
    try:
        resp = await client.post(config.chat_url, content=chat_body, headers=headers)
    except httpx.HTTPError as e:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Could not complete wrong-app token request ({type(e).__name__}).",
            what_it_tests=what_it_tests,
            details=details,
        )

    status_desc = _status_label(resp)
    details.append(f"`POST {config.chat_path}` with token from `{source}` -> {status_desc}")
    if resp.status_code not in REFUSED_STATUSES:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"App accepted a login token meant for a different service (HTTP {resp.status_code}).",
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Check the token's `aud` (intended audience) field to make sure the token was issued "
                "specifically for this app."
            ),
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"Real token meant for another service (`{source}`) was rejected ({status_desc}).",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_unauthenticated_identity_headers(
    client: httpx.AsyncClient, config: AccessCheckConfig
) -> CheckResult:
    """5. Send proxy identity headers (IAP / ALB / Azure / X-Forwarded-User) with no session."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Faking 'who I am' headers while logged out"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "After a user logs in, the cloud login gate attaches a hidden header (like `X-Goog-Authenticated-User-Email`) "
        "telling the backend Python code who the user is. This tests whether someone who is NOT logged in can attach "
        "their own fake email header to trick the app into thinking they are signed in."
    )
    details: list[str] = []
    failures: list[str] = []

    me_url = urljoin(config.base_url + "/", "api/me")
    chat_body = config.build_chat_body("Hello")

    for label, method, url, body in [
        ("GET /api/me", "GET", me_url, None),
        (f"POST {config.chat_path}", "POST", config.chat_url, chat_body),
    ]:
        headers = dict(SPOOFED_IDENTITY_HEADERS)
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            resp = await client.request(method, url, content=body, headers=headers)
        except httpx.HTTPError as e:
            details.append(f"`{label}`: network error ({type(e).__name__})")
            continue
        status_desc = _status_label(resp)
        details.append(f"`{label}` with fake email headers -> {status_desc}")
        if resp.status_code not in REFUSED_STATUSES:
            failures.append(f"`{label}` returned HTTP {resp.status_code}")

    if failures:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Logged-out request with fake identity headers was not blocked: {'; '.join(failures)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Ensure the login gate (IAP / load balancer) blocks unauthenticated requests and erases any "
                "`X-Goog-Authenticated-User-Email` or `X-Forwarded-User` headers sent by visitors."
            ),
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary="Logged-out requests with fake 'who I am' email headers were blocked at the front gate.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_cors_preflight(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """6. Send an OPTIONS preflight from an untrusted origin."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Cross-website request check (CORS, logged out)"
    owasp = "API8:2023 Security Misconfiguration"
    what_it_tests = (
        "If a user visits a malicious website (`https://evil.example`) in another browser tab, browsers first send "
        "a 'preflight' (`OPTIONS`) question asking if that outside website is allowed to talk to this app's API."
    )

    headers = {
        "Origin": EVIL_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization",
    }
    try:
        resp = await client.options(config.chat_url, headers=headers)
    except httpx.HTTPError as e:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Cross-website preflight request failed ({type(e).__name__}).",
            what_it_tests=what_it_tests,
        )

    acao = resp.headers.get("access-control-allow-origin", "")
    acac = resp.headers.get("access-control-allow-credentials", "")
    status_desc = _status_label(resp)
    details = [
        f"`OPTIONS {config.chat_path}` with `Origin: {EVIL_ORIGIN}` -> {status_desc}",
        f"`Access-Control-Allow-Origin`: `{acao or '(absent)'}`",
        f"`Access-Control-Allow-Credentials`: `{acac or '(absent)'}`",
    ]

    # Browsers only permit a CORS preflight when the HTTP status is 2xx (200-299).
    # Auth gateways like Google Cloud IAP return HTTP 401/403/302 on unauthenticated OPTIONS
    # requests (which causes the browser to block the cross-origin request), even if the
    # gateway's 401 error response echoes Access-Control-Allow-Origin.
    if resp.status_code in REFUSED_STATUSES:
        if acao:
            details.append(
                f"Note: Although the login gate echoed `Access-Control-Allow-Origin` on its `{status_desc}` error "
                "response, browsers reject any preflight that does not return HTTP 200-299 (and Check #12 verifies "
                "what the app returns once signed in)."
            )
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="PASS",
            summary=f"Outside website (`{EVIL_ORIGIN}`) preflight was blocked by the login gate ({status_desc}).",
            what_it_tests=what_it_tests,
            details=details,
        )

    if acao == EVIL_ORIGIN or (acao == "*" and acac.lower() == "true"):
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"App granted cross-website access to untrusted site `{EVIL_ORIGIN}` (HTTP {resp.status_code}, ACAO={acao!r}).",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Restrict `Access-Control-Allow-Origin` to trusted domains only and never echo arbitrary `Origin` headers.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"Outside website (`{EVIL_ORIGIN}`) was not granted permission to call the API.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_http_methods(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """7. Test unexpected HTTP verbs (GET, PUT, DELETE) on the chat endpoint."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Other request types (GET, PUT, DELETE) on chat API"
    owasp = "API8:2023 Security Misconfiguration"
    what_it_tests = (
        "The chat API is meant to receive messages (`POST`). This checks whether someone forgot to lock down "
        "other request types like `GET` (read), `PUT` (overwrite), or `DELETE`."
    )
    details: list[str] = []
    failures: list[str] = []

    allowed_refusal = REFUSED_STATUSES | {405}
    for method in ("GET", "PUT", "DELETE"):
        try:
            resp = await client.request(method, config.chat_url)
        except httpx.HTTPError as e:
            details.append(f"`{method} {config.chat_path}`: network error ({type(e).__name__})")
            continue
        status_desc = _status_label(resp)
        details.append(f"`{method} {config.chat_path}` -> {status_desc}")
        if resp.status_code not in allowed_refusal:
            failures.append(f"`{method}` returned HTTP {resp.status_code}")

    if failures:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Unexpected request types succeeded without login: {'; '.join(failures)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Block unused HTTP methods (`GET`, `PUT`, `DELETE`) and require login on all methods.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"`GET`, `PUT`, and `DELETE` requests on `{config.chat_path}` were blocked.",
        what_it_tests=what_it_tests,
        details=details,
    )


def discover_alternate_urls(config: AccessCheckConfig) -> list[str]:
    """Return configured alternate URLs plus sibling API Gateway stage URLs if applicable."""
    candidates: list[str] = list(config.alternate_urls)
    parsed = urlparse(config.chat_url)
    if ".execute-api." in parsed.netloc and parsed.netloc.endswith(".amazonaws.com"):
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            current_stage = parts[0]
            rest = "/".join(parts[1:])
            for stage in ("dev", "test", "stage", "staging", "prod", "v1"):
                if stage != current_stage:
                    candidates.append(f"{parsed.scheme}://{parsed.netloc}/{stage}/{rest}")
    seen: set[str] = set()
    unique: list[str] = []
    for url in candidates:
        if url not in seen and url != config.chat_url:
            seen.add(url)
            unique.append(url)
    return unique


async def check_alternate_urls(client: httpx.AsyncClient, config: AccessCheckConfig) -> CheckResult:
    """8. Probe alternate URLs or sibling gateway stages unauthenticated."""
    phase = "Phase 1: Signed-Out Checks (No Login)"
    name = "Alternate / backdoor server URLs"
    owasp = "API9:2023 Improper Inventory Management"
    what_it_tests = (
        "Sometimes an app has a secure front door (with login) plus a second URL (like a direct server URL or "
        "a `/dev` or `/prod` stage URL) where login was accidentally left off."
    )

    urls = discover_alternate_urls(config)
    if not urls:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary="Skipped (optional): no second server URL configured (`APP_ALTERNATE_URLS` unset).",
            what_it_tests=what_it_tests,
            details=[
                "If this app has a separate backend URL behind a load balancer, set `APP_ALTERNATE_URLS` in `.env` "
                "to test that it cannot be reached directly without logging in."
            ],
        )

    chat_body = config.build_chat_body("Hello")
    details: list[str] = []
    failures: list[str] = []
    for url in urls:
        try:
            resp = await client.post(url, content=chat_body, headers={"Content-Type": "application/json"})
        except httpx.HTTPError as e:
            details.append(f"`{url}` -> unreachable ({type(e).__name__})")
            continue
        status_desc = _status_label(resp)
        details.append(f"`POST {url}` -> {status_desc}")
        if resp.status_code not in REFUSED_STATUSES:
            failures.append(f"`{url}` returned HTTP {resp.status_code}")

    if failures:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Alternate URL(s) answered without login: {'; '.join(failures)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Lock down direct backend URLs and unused `/dev` or `/test` stage URLs.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"All {len(urls)} alternate URL(s) blocked logged-out access.",
        what_it_tests=what_it_tests,
        details=details,
    )


# ---------------------------------------------------------------------------
# Phase 2: Signed-in session & outsider checks
# ---------------------------------------------------------------------------


@dataclass
class SignedInSession:
    """Authenticated headers and cookie metadata extracted from the browser or env."""

    headers: dict[str, str]
    cookies: list[dict] = field(default_factory=list)
    source: str = ""


async def acquire_signed_in_session(config: AccessCheckConfig) -> SignedInSession | None:
    """Acquire an authenticated session from the browser profile (or HTTP API key/token)."""
    if config.page_url:
        playwright, context = await launch_browser(config.channel, config.profile, config.headless)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(config.page_url)
            if config.ready_selector:
                await wait_for_element(page, config.ready_selector, config.ready_timeout, config.sign_in_timeout)
            else:
                await page.wait_for_load_state()
            cookies = await context.cookies([config.base_url, config.page_url])
            if not cookies:
                cookies = await context.cookies()
            cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies if c.get("name"))
            headers: dict[str, str] = {}
            if cookie_header:
                headers["Cookie"] = cookie_header
            if config.api_key:
                headers["x-api-key"] = config.api_key
            if config.token:
                headers["Authorization"] = f"Bearer {config.token}"
            return SignedInSession(
                headers=headers,
                cookies=cookies,
                source=f"browser profile ({config.profile or (DEFAULT_PROFILE / (config.channel or 'chrome'))})",
            )
        finally:
            await context.close()
            await playwright.stop()

    if config.api_key or config.token:
        headers = {}
        if config.api_key:
            headers["x-api-key"] = config.api_key
        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"
        return SignedInSession(headers=headers, cookies=[], source="APP_API_KEY / APP_TOKEN")

    return None


async def check_signed_in_header_spoofing(
    client: httpx.AsyncClient, config: AccessCheckConfig, session: SignedInSession
) -> CheckResult:
    """9. Send spoofed identity headers inside a valid authenticated session."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Impersonating another user while logged in (header spoofing)"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "Once a real user is logged in, can they attach a fake `X-Goog-Authenticated-User-Email` header to their "
        "request to trick the app into thinking they are a different coworker?"
    )
    details: list[str] = []

    me_url = urljoin(config.base_url + "/", "api/me")
    baseline_headers = dict(session.headers)
    spoof_headers = {**session.headers, **SPOOFED_IDENTITY_HEADERS}

    try:
        base_resp = await client.get(me_url, headers=baseline_headers)
        spoof_resp = await client.get(me_url, headers=spoof_headers)
    except httpx.HTTPError as e:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Could not reach `/api/me` ({type(e).__name__}).",
            what_it_tests=what_it_tests,
        )

    if base_resp.status_code == 200:
        details.append(f"Normal `GET /api/me` (your real login) -> HTTP 200 (`{base_resp.text.strip()[:120]}`)")
        details.append(
            f"Modified `GET /api/me` with fake header `X-Goog-Authenticated-User-Email: accounts.google.com:{SPOOFED_EMAIL}` "
            f"-> HTTP {spoof_resp.status_code} (`{spoof_resp.text.strip()[:120]}`)"
        )
        if SPOOFED_EMAIL.lower() in spoof_resp.text.lower():
            return CheckResult(
                phase=phase,
                name=name,
                owasp=owasp,
                verdict="FAIL",
                summary=f"App accepted a fake user email header (`{SPOOFED_EMAIL}`) from a logged-in user.",
                what_it_tests=what_it_tests,
                details=details,
                remediation=(
                    "Strip or overwrite client-supplied `X-Goog-Authenticated-User-Email` and `X-Forwarded-User` "
                    "headers at the login proxy, and verify the signed IAP JWT (`X-Goog-IAP-JWT-Assertion`)."
                ),
            )
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="PASS",
            summary="App ignored the fake user email header and kept your true logged-in email.",
            what_it_tests=what_it_tests,
            details=details,
        )

    # Fallback if /api/me is not implemented: check chat endpoint with spoofed headers
    chat_body = config.build_chat_body("Hello")
    chat_headers = {**spoof_headers, "Content-Type": "application/json"}
    chat_resp = await client.post(config.chat_url, content=chat_body, headers=chat_headers)
    details.append(f"`GET /api/me` returned {_status_label(base_resp)}; tested `POST {config.chat_path}` -> {_status_label(chat_resp)}")
    if SPOOFED_EMAIL.lower() in chat_resp.text.lower():
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Chat response accepted fake identity header `{SPOOFED_EMAIL}`.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Strip client-supplied proxy identity headers at the login gate.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary="Fake user email headers were ignored by the application.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_exposed_response_fields(
    client: httpx.AsyncClient, config: AccessCheckConfig, session: SignedInSession
) -> CheckResult:
    """10. Inspect the raw JSON response from the chat endpoint for extra/internal trace fields."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Behind-the-scenes debug data (`trace`) in chat replies"
    owasp = "API3:2023 Broken Object Property Level Authorization"
    what_it_tests = (
        "When the chatbot answers a question, does the server also send hidden debugging data (like the secret "
        "system prompt, internal tool definitions, or raw model traces) to the user's browser?"
    )

    chat_body = config.build_chat_body("Hello")
    headers = {**session.headers, "Content-Type": "application/json"}
    try:
        resp = await client.post(config.chat_url, content=chat_body, headers=headers)
    except httpx.HTTPError as e:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Chat request failed during response inspection ({type(e).__name__}).",
            what_it_tests=what_it_tests,
        )

    if not (200 <= resp.status_code < 300):
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Signed-in chat request returned {_status_label(resp)}; could not inspect response fields.",
            what_it_tests=what_it_tests,
            details=[f"`POST {config.chat_path}` -> {_status_label(resp)}: `{resp.text[:160]}`"],
        )

    try:
        data = json.loads(resp.text)
    except ValueError:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="PASS",
            summary="Chat response is plain text (no hidden debug fields).",
            what_it_tests=what_it_tests,
            details=[f"`POST {config.chat_path}` returned non-JSON body ({len(resp.text)} chars)."],
        )

    if not isinstance(data, dict):
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="PASS",
            summary=f"Chat response is a JSON {type(data).__name__} with no extra debug fields.",
            what_it_tests=what_it_tests,
        )

    expected_top_key = (config.response_path.split(".")[0] if config.response_path else "reply") or "reply"
    allowed_keys = {expected_top_key} | BENIGN_RESPONSE_KEYS

    extra_keys = [k for k in data.keys() if k not in allowed_keys]
    sensitive_found = [k for k in data.keys() if k.lower() in SENSITIVE_RESPONSE_KEYS]

    details = [f"Top-level JSON fields returned by `POST {config.chat_path}`: `{', '.join(data.keys())}`"]
    for key in extra_keys:
        val = data[key]
        serialized = json.dumps(val) if not isinstance(val, str) else val
        subkeys = (
            f"; contains: `{', '.join(list(val.keys())[:10])}`"
            if isinstance(val, dict) and val
            else ""
        )
        details.append(f"Extra field `{key}` ({type(val).__name__}, {len(serialized)} chars{subkeys})")

    if sensitive_found or extra_keys:
        flagged = sensitive_found or extra_keys
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="REVIEW",
            summary=(
                f"Chat API response sends extra data field(s) alongside `{expected_top_key}`: "
                f"`{', '.join(flagged)}` (viewable in browser DevTools)."
            ),
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Ask the app developer if extra fields like `trace` (which includes `system_instruction` and "
                "`tool_declarations`) are needed by the UI in production, or if they should be hidden behind a debug flag."
            ),
        )

    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"Chat response returns only expected fields (`{', '.join(data.keys())}`).",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_openapi_docs_exposed(
    signed_in_client: httpx.AsyncClient,
    signed_out_client: httpx.AsyncClient,
    config: AccessCheckConfig,
    session: SignedInSession,
) -> CheckResult:
    """11. Check if /openapi.json, /docs, or /redoc are exposed while signed in, and verify all discovered routes signed out."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Developer API documentation pages (`/docs`, `/openapi.json`)"
    owasp = "API9:2023 Improper Inventory Management"
    what_it_tests = (
        "Python web frameworks like FastAPI automatically generate interactive developer documentation pages "
        "(`/docs`, `/redoc`, `/openapi.json`) listing every API route. This checks if those pages are still turned on "
        "and whether any route listed in them forgot to require login."
    )
    details: list[str] = []
    exposed_docs: list[str] = []
    discovered_routes: list[tuple[str, str]] = []

    for doc_path in ("/openapi.json", "/docs", "/redoc"):
        url = urljoin(config.base_url + "/", doc_path.lstrip("/"))
        try:
            resp = await signed_in_client.get(url, headers=session.headers)
        except httpx.HTTPError as e:
            details.append(f"`GET {doc_path}` (signed in): error ({type(e).__name__})")
            continue
        details.append(f"`GET {doc_path}` (signed in) -> {_status_label(resp)}")
        if resp.status_code == 200:
            exposed_docs.append(doc_path)
            if doc_path == "/openapi.json":
                try:
                    spec = json.loads(resp.text)
                    paths_obj = spec.get("paths", {})
                    if isinstance(paths_obj, dict):
                        for route_path, methods_obj in paths_obj.items():
                            if isinstance(methods_obj, dict):
                                for m in methods_obj.keys():
                                    if m.upper() in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                                        discovered_routes.append((m.upper(), route_path))
                except ValueError:
                    pass

    unauth_leaks: list[str] = []
    if discovered_routes:
        route_list = ", ".join(f"`{m} {p}`" for m, p in discovered_routes)
        details.append(f"API routes listed in `/openapi.json`: {route_list}")
        for method, route_path in discovered_routes:
            if route_path == "/healthz":
                continue
            url = urljoin(config.base_url + "/", route_path.lstrip("/"))
            try:
                resp = await signed_out_client.request(method, url)
            except httpx.HTTPError:
                continue
            if resp.status_code not in REFUSED_STATUSES:
                unauth_leaks.append(f"`{method} {route_path}` (HTTP {resp.status_code})")
                details.append(f"Logged-out test of discovered route `{method} {route_path}` -> HTTP {resp.status_code} (FAIL)")

    if unauth_leaks:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Route(s) listed in `/openapi.json` are reachable without logging in: {', '.join(unauth_leaks)}.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Require login on all API routes listed in `/openapi.json`.",
        )

    if exposed_docs:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="REVIEW",
            summary=(
                f"Developer docs (`{', '.join(exposed_docs)}`) are turned on for logged-in users "
                f"({len(discovered_routes)} API route(s) listed; all blocked when logged out)."
            ),
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "If interactive developer docs aren't needed in production, turn them off in FastAPI: "
                "`FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`."
            ),
        )

    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary="Developer docs (`/openapi.json`, `/docs`, `/redoc`) are turned off.",
        what_it_tests=what_it_tests,
        details=details,
    )


async def check_signed_in_cors(
    client: httpx.AsyncClient, config: AccessCheckConfig, session: SignedInSession
) -> CheckResult:
    """12. Send an authenticated request with an untrusted Origin header."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Cross-website data access while logged in (CORS)"
    owasp = "API8:2023 Security Misconfiguration"
    what_it_tests = (
        "While you are logged into the app, if you open a malicious website (`https://evil.example`) in another "
        "tab, does the app allow that outside website to read your logged-in data?"
    )

    me_url = urljoin(config.base_url + "/", "api/me")
    headers = {**session.headers, "Origin": EVIL_ORIGIN}
    try:
        resp = await client.get(me_url, headers=headers)
    except httpx.HTTPError as e:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Signed-in cross-website check failed ({type(e).__name__}).",
            what_it_tests=what_it_tests,
        )

    acao = resp.headers.get("access-control-allow-origin", "")
    acac = resp.headers.get("access-control-allow-credentials", "")
    details = [
        f"Logged-in `GET /api/me` with `Origin: {EVIL_ORIGIN}` -> {_status_label(resp)}",
        f"`Access-Control-Allow-Origin`: `{acao or '(absent)'}`",
        f"`Access-Control-Allow-Credentials`: `{acac or '(absent)'}`",
    ]
    if acao == EVIL_ORIGIN or (acao == "*" and acac.lower() == "true"):
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"App allows outside website `{EVIL_ORIGIN}` to read logged-in user data (ACAO={acao!r}).",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Do not allow untrusted outside domains (`Access-Control-Allow-Origin`) on logged-in endpoints.",
        )
    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"App did not allow outside website (`{EVIL_ORIGIN}`) to read logged-in user data.",
        what_it_tests=what_it_tests,
        details=details,
    )


def check_cookie_security_flags(config: AccessCheckConfig, session: SignedInSession) -> CheckResult:
    """13. Inspect session cookies for Secure, HttpOnly, and SameSite flags (never logs cookie values)."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Login cookie safety locks (`Secure` & `HttpOnly`)"
    owasp = "API2:2023 Broken Authentication"
    what_it_tests = (
        "Once you log in, your browser stores login cookies to keep you signed in. This checks that login cookies "
        "have `Secure=True` (only sent over encrypted HTTPS) and `HttpOnly=True` (hidden from scripts so they can't be stolen)."
    )

    if not session.cookies:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary="No browser session cookies present (authenticated via header or no cookies set).",
            what_it_tests=what_it_tests,
        )

    is_https = config.base_url.startswith("https://")
    details: list[str] = []
    insecure_cookies: list[str] = []
    missing_httponly: list[str] = []

    for c in session.cookies:
        c_name = c.get("name", "<unnamed>")
        secure = bool(c.get("secure", False))
        http_only = bool(c.get("httpOnly", False))
        same_site = c.get("sameSite", "None")
        domain = c.get("domain", "")
        details.append(
            f"Cookie `{c_name}` (domain `{domain}`): `Secure={secure}`, `HttpOnly={http_only}`, `SameSite={same_site}`"
        )
        is_auth_like = any(
            token in c_name.upper() for token in ("IAP", "AUTH", "SESSION", "TOKEN", "SID", "__HOST-")
        )
        if is_https and is_auth_like and not secure:
            insecure_cookies.append(c_name)
        if is_auth_like and not http_only:
            missing_httponly.append(c_name)

    if insecure_cookies:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Login cookie(s) missing `Secure` encryption lock: `{', '.join(insecure_cookies)}`.",
            what_it_tests=what_it_tests,
            details=details,
            remediation="Set `Secure; HttpOnly; SameSite=Lax` (or `Strict`) on all login and session cookies.",
        )

    if missing_httponly:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="REVIEW",
            summary=f"Login-related cookie(s) have `HttpOnly=False`: `{', '.join(missing_httponly)}`.",
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Confirm whether browser JavaScript needs to read these cookies; if not, set `HttpOnly=True` "
                "so scripts cannot read login cookies."
            ),
        )

    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"All login cookies have `Secure` and `HttpOnly` safety locks enabled ({len(session.cookies)} total cookies inspected).",
        what_it_tests=what_it_tests,
        details=details,
    )


async def _wait_for_outsider_outcome(page, config: AccessCheckConfig) -> None:
    """If the outsider browser is on an interactive sign-in form, wait until sign-in finishes
    (either landing back on the app's origin or hitting an IAP/IdP access-denied screen)."""
    if config.headless:
        return

    target_origin = (urlparse(config.base_url).scheme, urlparse(config.base_url).netloc)
    denied_phrases = ("don't have access", "do not have access", "access denied", "not authorized", "403")

    async def settled() -> bool:
        parsed = urlparse(getattr(page, "url", "") or "")
        if (parsed.scheme, parsed.netloc) == target_origin:
            return True
        if not hasattr(page, "content"):
            return True
        try:
            body_lower = (await page.content()).lower()
            return any(p in body_lower for p in denied_phrases)
        except Exception:
            return False

    if await settled():
        return

    email_hint = f" ({config.outsider_email})" if config.outsider_email else ""
    print(
        f"\nWaiting for outsider sign-in{email_hint} in the browser window "
        f"(up to {config.sign_in_timeout:.0f}s)..."
    )
    deadline = time.monotonic() + config.sign_in_timeout
    while time.monotonic() < deadline:
        if await settled():
            return
        await asyncio.sleep(POLL_SECONDS)


async def check_outsider_account(
    client: httpx.AsyncClient, config: AccessCheckConfig
) -> CheckResult:
    """14. Test whether a signed-in outsider account (non-permitted user) is refused access."""
    phase = "Phase 2: Signed-In & Session Checks (Logged In)"
    name = "Outsider Google account access (non-organization user)"
    owasp = "API5:2023 Broken Function Level Authorization"
    what_it_tests = (
        "If someone signs in with a real Google account that is NOT part of your organization (like a personal "
        "`@gmail.com` account), does the app block them, or did someone accidentally allow any Google user in?"
    )

    if not config.outsider_profile:
        email_hint = f" (e.g. `{config.outsider_email}`)" if config.outsider_email else ""
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary=f"Skipped (run with `--outsider` to test an outside account{email_hint}).",
            what_it_tests=what_it_tests,
            details=[
                "To test that a signed-in Google account outside your organization is blocked, "
                "run `python -m harnesses.access_checks --outsider` and sign in once with an outside account."
            ],
        )

    if not config.page_url:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="NOT TESTED",
            summary="`APP_BROWSER_URL` is not set; cannot launch outsider browser session.",
            what_it_tests=what_it_tests,
        )

    playwright, context = await launch_browser(config.channel, config.outsider_profile, config.headless)
    try:
        page = context.pages[0] if context.pages else await context.new_page()
        nav_response = await page.goto(config.page_url)
        nav_status = nav_response.status if nav_response else 0

        await _wait_for_outsider_outcome(page, config)

        reached_chat_box = False
        if config.ready_selector:
            try:
                await page.locator(config.ready_selector).first.wait_for(state="visible", timeout=2000)
                reached_chat_box = True
            except Exception:
                reached_chat_box = False

        cookies = await context.cookies([config.base_url, config.page_url])
        if not cookies:
            cookies = await context.cookies()
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies if c.get("name"))
    finally:
        await context.close()
        await playwright.stop()

    outsider_label = config.outsider_email or str(config.outsider_profile)
    details = [
        f"Outsider account tested: `{outsider_label}`",
        f"Browser navigation to `{config.page_url}` -> HTTP {nav_status}, chat box visible: `{reached_chat_box}`",
    ]

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cookie_header:
        headers["Cookie"] = cookie_header

    chat_resp = await client.post(config.chat_url, content=config.build_chat_body("Hello"), headers=headers)
    details.append(f"Outsider `POST {config.chat_path}` -> {_status_label(chat_resp)}")

    if reached_chat_box or chat_resp.status_code not in REFUSED_STATUSES:
        return CheckResult(
            phase=phase,
            name=name,
            owasp=owasp,
            verdict="FAIL",
            summary=f"Outsider account (`{outsider_label}`) was allowed into the app (`POST {config.chat_path}` -> HTTP {chat_resp.status_code}).",
            what_it_tests=what_it_tests,
            details=details,
            remediation=(
                "Restrict Google IAP / OAuth access to your organization's domain or authorized user group "
                "instead of allowing all Google accounts (`allAuthenticatedUsers`)."
            ),
        )

    return CheckResult(
        phase=phase,
        name=name,
        owasp=owasp,
        verdict="PASS",
        summary=f"Outsider account (`{outsider_label}`) was blocked ({_status_label(chat_resp)}).",
        what_it_tests=what_it_tests,
        details=details,
    )


# ---------------------------------------------------------------------------
# Runner & Markdown Report Generation
# ---------------------------------------------------------------------------


async def run_all_checks(
    config: AccessCheckConfig,
    *,
    phase1_only: bool = False,
    session_getter=acquire_signed_in_session,
) -> list[CheckResult]:
    """Execute Phase 1 and Phase 2 checks and return their CheckResults in order."""
    results: list[CheckResult] = []
    async with httpx.AsyncClient(timeout=config.timeout, follow_redirects=False) as client:
        results.append(await check_no_credentials(client, config))
        results.append(await check_junk_credentials(client, config))
        results.append(await check_forged_jwt(client, config))
        results.append(await check_wrong_audience_token(client, config))
        results.append(await check_unauthenticated_identity_headers(client, config))
        results.append(await check_cors_preflight(client, config))
        results.append(await check_http_methods(client, config))
        results.append(await check_alternate_urls(client, config))

        if phase1_only:
            return results

        session = await session_getter(config)
        if session is None:
            for name, owasp in [
                ("Impersonating another user while logged in (header spoofing)", "API2:2023 Broken Authentication"),
                ("Behind-the-scenes debug data (`trace`) in chat replies", "API3:2023 Broken Object Property Level Authorization"),
                ("Developer API documentation pages (`/docs`, `/openapi.json`)", "API9:2023 Improper Inventory Management"),
                ("Cross-website data access while logged in (CORS)", "API8:2023 Security Misconfiguration"),
                ("Login cookie safety locks (`Secure` & `HttpOnly`)", "API2:2023 Broken Authentication"),
                ("Outsider Google account access (non-organization user)", "API5:2023 Broken Function Level Authorization"),
            ]:
                results.append(
                    CheckResult(
                        phase="Phase 2: Signed-In & Session Checks (Logged In)",
                        name=name,
                        owasp=owasp,
                        verdict="NOT TESTED",
                        summary="No signed-in session configured (`APP_BROWSER_URL`, `APP_API_KEY`, and `APP_TOKEN` unset).",
                    )
                )
            return results

        results.append(await check_signed_in_header_spoofing(client, config, session))
        results.append(await check_exposed_response_fields(client, config, session))
        results.append(await check_openapi_docs_exposed(client, client, config, session))
        results.append(await check_signed_in_cors(client, config, session))
        results.append(check_cookie_security_flags(config, session))
        results.append(await check_outsider_account(client, config))

    return results


def render_report(config: AccessCheckConfig, results: list[CheckResult], timestamp: datetime | None = None) -> str:
    """Render the complete Markdown access-control report."""
    now = timestamp or datetime.now(timezone.utc)
    counts = {v: sum(1 for r in results if r.verdict == v) for v in ("PASS", "FAIL", "REVIEW", "NOT TESTED")}

    lines = [
        "# Application Access-Control & API Configuration Report",
        "",
        f"- **Target Base URL:** `{config.base_url}`",
        f"- **Chat Endpoint:** `{config.chat_url}`",
        f"- **Date (UTC):** `{now.strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Summary:** **{counts['PASS']} PASS** | **{counts['FAIL']} FAIL** | **{counts['REVIEW']} REVIEW** | **{counts['NOT TESTED']} NOT TESTED**",
        "",
        "## How to Read This Report",
        "",
        "- **PASS:** The security lock worked as expected—unauthorized requests were blocked.",
        "- **FAIL:** A security lock is missing or misconfigured and needs to be fixed.",
        "- **REVIEW:** Not a broken lock, but the app exposes extra developer/debugging details to logged-in users. Ask the developer if this is intentional for production.",
        "- **NOT TESTED:** An optional check was skipped because an optional setting wasn't provided.",
        "",
        "## Summary Table",
        "",
        "| # | Phase | Check | OWASP API Top 10 | Verdict | Plain-English Result |",
        "|---|---|---|---|---|---|",
    ]

    for idx, res in enumerate(results, start=1):
        short_phase = "Phase 1 (Logged Out)" if "Phase 1" in res.phase else "Phase 2 (Logged In)"
        safe_summary = res.summary.replace("|", "\\|")
        lines.append(
            f"| {idx} | {short_phase} | {res.name} | `{res.owasp}` | **{res.verdict}** | {safe_summary} |"
        )

    lines.extend(["", "## Detailed Findings & Evidence", ""])
    current_phase = ""
    for idx, res in enumerate(results, start=1):
        if res.phase != current_phase:
            current_phase = res.phase
            lines.extend([f"### {current_phase}", ""])
        lines.append(f"#### {idx}. {res.name} — **{res.verdict}**")
        if res.what_it_tests:
            lines.append(f"- **What we tested:** {res.what_it_tests}")
        lines.append(f"- **What happened:** {res.summary}")
        if res.verdict == "PASS":
            meaning = "No action needed—this security lock is working properly."
        elif res.verdict == "FAIL":
            meaning = res.remediation or "A security lock is missing or misconfigured and should be fixed by the developer."
        elif res.verdict == "REVIEW":
            meaning = res.remediation or "Not a broken lock, but worth asking the developer if this is intentional for production."
        else:
            meaning = "Optional check skipped—no issue found, just not applicable or not configured."
        lines.append(f"- **What it means:** {meaning}")
        lines.append(f"- **OWASP Category:** `{res.owasp}`")
        if res.details:
            lines.append("- **Technical Evidence:**")
            for item in res.details:
                lines.append(f"  - {item}")
        lines.append("")

    lines.extend(
        [
            "## Quick Glossary of Terms",
            "",
            "- **HTTP Status Codes:** `200` = Allowed in; `302 -> accounts.google.com` = Redirected to Google sign-in page (blocked); `401` / `403` = Access denied (blocked); `404` = Page not found.",
            "- **IAP (Identity-Aware Proxy):** Google Cloud's front-door login gatekeeper that forces users to sign in with an authorized Google account before reaching the app.",
            "- **Bearer Token / API Key:** Digital credentials sent in request headers by scripts or apps instead of typing a password in a browser.",
            "- **JWT (JSON Web Token) / OIDC Token:** A digital ID badge issued by Google after sign-in. It contains the user's email, which app it was made for (`aud` / audience), and a cryptographic signature proving Google issued it.",
            "- **Proxy Identity Header (`X-Goog-Authenticated-User-Email`):** A hidden label that Google's login gate attaches to each request after verifying the user, telling the backend Python code which user is logged in.",
            "- **CORS (Cross-Origin Resource Sharing):** Browser rules that stop a random website open in another tab (like `evil.example`) from secretly talking to this app's API in the background.",
            "- **OpenAPI (`/openapi.json`, `/docs`, `/redoc`):** Auto-generated developer documentation pages built into FastAPI that list all API endpoints and how to call them.",
            "",
        ]
    )

    return "\n".join(lines)


def default_report_path(config: AccessCheckConfig, out_arg: str = "", timestamp: datetime | None = None) -> Path:
    """Resolve the output Markdown file path under findings/."""
    now = timestamp or datetime.now(timezone.utc)
    host = urlparse(config.base_url).netloc.replace(":", "_")
    host_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", host) or "app"
    filename = f"{now.strftime('%Y-%m-%d')}_access_checks_{host_slug}.md"
    if out_arg:
        out_path = Path(out_arg)
        if out_path.suffix.lower() == ".md":
            return out_path
        return out_path / filename
    return DEFAULT_FINDINGS_DIR / filename


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HTTP & session access-control checks against the configured application."
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run only Phase 1 (signed-out plain HTTP checks) without launching a browser.",
    )
    parser.add_argument(
        "--outsider",
        action="store_true",
        help="Enable the outsider account check using APP_OUTSIDER_PROFILE or ~/.rt_outsider_profile.",
    )
    parser.add_argument(
        "--outsider-profile",
        default="",
        help="Browser profile directory for a non-permitted outsider account.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output Markdown file or directory (default: findings/<date>_access_checks_<host>.md).",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AccessCheckConfig.from_env(
        outsider=args.outsider,
        outsider_profile=args.outsider_profile or None,
    )

    print(f"Running access checks against {config.chat_url} ...")
    results = await run_all_checks(config, phase1_only=args.phase1_only)

    for idx, res in enumerate(results, start=1):
        print(f"  [{res.verdict:10s}] {idx:2d}. {res.name}: {res.summary}")

    report_path = default_report_path(config, args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, results), encoding="utf-8")

    counts = {v: sum(1 for r in results if r.verdict == v) for v in ("PASS", "FAIL", "REVIEW", "NOT TESTED")}
    print(
        f"\nSummary: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
        f"{counts['REVIEW']} REVIEW, {counts['NOT TESTED']} NOT TESTED"
    )
    print(f"Report written to {report_path}")
    return 1 if counts["FAIL"] > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

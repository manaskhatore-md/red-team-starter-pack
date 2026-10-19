"""Offline unit tests for harnesses/access_checks.py."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from harnesses import access_checks
from harnesses.access_checks import (
    EVIL_ORIGIN,
    SPOOFED_EMAIL,
    AccessCheckConfig,
    SignedInSession,
    check_alternate_urls,
    check_cookie_security_flags,
    check_cors_preflight,
    check_exposed_response_fields,
    check_forged_jwt,
    check_http_methods,
    check_junk_credentials,
    check_no_credentials,
    check_openapi_docs_exposed,
    check_outsider_account,
    check_signed_in_cors,
    check_signed_in_header_spoofing,
    check_unauthenticated_identity_headers,
    check_wrong_audience_token,
    discover_alternate_urls,
)


def make_config(**overrides) -> AccessCheckConfig:
    defaults = dict(
        base_url="https://app.example.test",
        chat_url="https://app.example.test/api/chat",
        chat_path="/api/chat",
        page_url="https://app.example.test/",
        ready_selector="#composer-input",
        request_template='{"message": "{PROMPT}", "history": "{HISTORY}", "session_id": "{CONVERSATION_ID}"}',
        response_path="reply",
    )
    defaults.update(overrides)
    return AccessCheckConfig(**defaults)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


# ---------------------------------------------------------------------------
# Configuration & URL resolution
# ---------------------------------------------------------------------------


def test_config_resolves_relative_endpoint_against_browser_url(clean_env):
    clean_env.setenv("APP_BROWSER_URL", "https://ga-chat.example.test/")
    clean_env.setenv("APP_ENDPOINT", "/api/chat")
    clean_env.setenv("APP_ACCESS_CHECK_PATHS", "/admin, /api/chat, custom")

    cfg = AccessCheckConfig.from_env(outsider=True)
    assert cfg.base_url == "https://ga-chat.example.test"
    assert cfg.chat_url == "https://ga-chat.example.test/api/chat"
    assert cfg.chat_path == "/api/chat"
    assert "/admin" in cfg.probe_paths()
    assert "/custom" in cfg.probe_paths()
    assert cfg.probe_paths().count("/api/chat") == 1
    assert cfg.outsider_profile is not None


def test_config_requires_endpoint_or_browser_url(clean_env):
    with pytest.raises(SystemExit, match="Neither APP_ENDPOINT nor APP_BROWSER_URL"):
        AccessCheckConfig.from_env()


def test_build_chat_body_fills_prompt_history_and_conversation_id():
    cfg = make_config()
    body = json.loads(cfg.build_chat_body("Ping"))
    assert body == {"message": "Ping", "history": [], "session_id": "access-check-session"}


# ---------------------------------------------------------------------------
# Phase 1: Signed-out checks
# ---------------------------------------------------------------------------


def test_no_credentials_passes_when_all_paths_redirect_or_refuse():
    cfg = make_config()
    client = mock_client(
        lambda req: httpx.Response(302, headers={"Location": "https://accounts.google.com/ServiceLogin"})
    )
    res = asyncio.run(check_no_credentials(client, cfg))
    assert res.verdict == "PASS"


def test_no_credentials_marks_review_when_only_healthz_is_200():
    cfg = make_config()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(401)

    res = asyncio.run(check_no_credentials(mock_client(handler), cfg))
    assert res.verdict == "REVIEW"
    assert "/healthz" in res.summary


def test_no_credentials_fails_on_200_or_422_validation_error():
    cfg = make_config()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/event":
            return httpx.Response(422, json={"detail": "validation error"})
        return httpx.Response(403)

    res = asyncio.run(check_no_credentials(mock_client(handler), cfg))
    assert res.verdict == "FAIL"
    assert "422" in res.summary


def test_junk_credentials_and_forged_jwt_pass_on_401_and_fail_on_200():
    cfg = make_config()
    refuse_client = mock_client(lambda req: httpx.Response(401))
    allow_client = mock_client(lambda req: httpx.Response(200, json={"reply": "oops"}))

    assert asyncio.run(check_junk_credentials(refuse_client, cfg)).verdict == "PASS"
    assert asyncio.run(check_junk_credentials(allow_client, cfg)).verdict == "FAIL"

    assert asyncio.run(check_forged_jwt(refuse_client, cfg)).verdict == "PASS"
    assert asyncio.run(check_forged_jwt(allow_client, cfg)).verdict == "FAIL"


def test_wrong_audience_token_verdicts(monkeypatch):
    cfg_none = make_config(wrong_token="")
    monkeypatch.setattr(access_checks, "resolve_wrong_audience_token", lambda configured="": ("", ""))
    res_skip = asyncio.run(check_wrong_audience_token(mock_client(lambda r: httpx.Response(401)), cfg_none))
    assert res_skip.verdict == "NOT TESTED"

    monkeypatch.setattr(
        access_checks,
        "resolve_wrong_audience_token",
        lambda configured="": ("eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJotherIn0.sig", "APP_WRONG_TOKEN"),
    )
    res_pass = asyncio.run(check_wrong_audience_token(mock_client(lambda r: httpx.Response(403)), cfg_none))
    assert res_pass.verdict == "PASS"
    assert "eyJhbGciOiJSUzI1NiJ9" not in "\n".join(res_pass.details)

    res_fail = asyncio.run(check_wrong_audience_token(mock_client(lambda r: httpx.Response(200)), cfg_none))
    assert res_fail.verdict == "FAIL"


def test_unauthenticated_identity_headers_and_http_methods():
    cfg = make_config()
    refuse_client = mock_client(lambda req: httpx.Response(403))
    allow_client = mock_client(lambda req: httpx.Response(200))

    assert asyncio.run(check_unauthenticated_identity_headers(refuse_client, cfg)).verdict == "PASS"
    assert asyncio.run(check_unauthenticated_identity_headers(allow_client, cfg)).verdict == "FAIL"

    assert asyncio.run(check_http_methods(mock_client(lambda req: httpx.Response(405)), cfg)).verdict == "PASS"
    assert asyncio.run(check_http_methods(allow_client, cfg)).verdict == "FAIL"


def test_cors_preflight_passes_when_restricted_and_fails_when_evil_origin_allowed():
    cfg = make_config()
    safe_client = mock_client(lambda req: httpx.Response(204))
    iap_401_client = mock_client(
        lambda req: httpx.Response(
            401,
            headers={
                "Access-Control-Allow-Origin": EVIL_ORIGIN,
                "Access-Control-Allow-Credentials": "true",
            },
        )
    )
    evil_client = mock_client(
        lambda req: httpx.Response(
            204,
            headers={
                "Access-Control-Allow-Origin": EVIL_ORIGIN,
                "Access-Control-Allow-Credentials": "true",
            },
        )
    )
    assert asyncio.run(check_cors_preflight(safe_client, cfg)).verdict == "PASS"
    assert asyncio.run(check_cors_preflight(iap_401_client, cfg)).verdict == "PASS"
    assert asyncio.run(check_cors_preflight(evil_client, cfg)).verdict == "FAIL"


def test_alternate_urls_discovers_api_gateway_sibling_stages():
    cfg = make_config(
        base_url="https://abc123.execute-api.us-east-1.amazonaws.com",
        chat_url="https://abc123.execute-api.us-east-1.amazonaws.com/test/chat",
        chat_path="/test/chat",
    )
    urls = discover_alternate_urls(cfg)
    assert "https://abc123.execute-api.us-east-1.amazonaws.com/dev/chat" in urls
    assert "https://abc123.execute-api.us-east-1.amazonaws.com/prod/chat" in urls
    assert "https://abc123.execute-api.us-east-1.amazonaws.com/test/chat" not in urls

    res_pass = asyncio.run(check_alternate_urls(mock_client(lambda r: httpx.Response(403)), cfg))
    assert res_pass.verdict == "PASS"

    def leaky_prod(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/prod/chat":
            return httpx.Response(200, json={"reply": "unauthenticated prod stage"})
        return httpx.Response(403)

    res_fail = asyncio.run(check_alternate_urls(mock_client(leaky_prod), cfg))
    assert res_fail.verdict == "FAIL"
    assert "/prod/chat" in res_fail.summary


# ---------------------------------------------------------------------------
# Phase 2: Signed-in checks
# ---------------------------------------------------------------------------


def test_signed_in_header_spoofing_passes_when_gateway_preserves_real_user():
    cfg = make_config()
    session = SignedInSession(headers={"Cookie": "__Host-GCP_IAP_AUTH_TOKEN=secret-cookie-value"})

    def safe_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"email": "dan.anafi@maryland.gov"})

    def vulnerable_handler(req: httpx.Request) -> httpx.Response:
        spoofed = req.headers.get("x-goog-authenticated-user-email", "")
        email = spoofed.split(":")[-1] if ":" in spoofed else "dan.anafi@maryland.gov"
        return httpx.Response(200, json={"email": email})

    assert asyncio.run(check_signed_in_header_spoofing(mock_client(safe_handler), cfg, session)).verdict == "PASS"
    res_fail = asyncio.run(check_signed_in_header_spoofing(mock_client(vulnerable_handler), cfg, session))
    assert res_fail.verdict == "FAIL"
    assert SPOOFED_EMAIL in res_fail.summary


def test_signed_in_header_spoofing_handles_missing_me_endpoint_and_errors():
    cfg = make_config()
    session = SignedInSession(headers={"Cookie": "__Host-GCP_IAP_AUTH_TOKEN=secret-cookie-value"})

    # 1. /api/me returns 404, /api/chat returns normal reply without echoing email -> NOT TESTED (not fake PASS)
    def no_me_normal_chat(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/me":
            return httpx.Response(404)
        return httpx.Response(200, json={"reply": "Hello! How can I help you?"})

    res_not_tested = asyncio.run(check_signed_in_header_spoofing(mock_client(no_me_normal_chat), cfg, session))
    assert res_not_tested.verdict == "NOT TESTED"

    # 2. /api/me returns 404, /api/chat reflects SPOOFED_EMAIL in trace -> FAIL
    def no_me_leaky_chat(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/me":
            return httpx.Response(404)
        return httpx.Response(200, json={"reply": "Hi", "trace": {"user": SPOOFED_EMAIL}})

    res_chat_fail = asyncio.run(check_signed_in_header_spoofing(mock_client(no_me_leaky_chat), cfg, session))
    assert res_chat_fail.verdict == "FAIL"

    # 3. /api/me returns 404, /api/chat raises ReadTimeout -> NOT TESTED (does not crash)
    def no_me_timeout_chat(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/me":
            return httpx.Response(404)
        raise httpx.ReadTimeout("chat timed out", request=req)

    res_timeout = asyncio.run(check_signed_in_header_spoofing(mock_client(no_me_timeout_chat), cfg, session))
    assert res_timeout.verdict == "NOT TESTED"

    # 4. /api/me returns 200 HTML SPA shell (no user identity) -> falls back and returns NOT TESTED (not fake PASS)
    def html_spa_fallback(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/me":
            return httpx.Response(200, text="<!doctype html><html><body>SPA</body></html>")
        return httpx.Response(200, json={"reply": "Hello"})

    res_spa = asyncio.run(check_signed_in_header_spoofing(mock_client(html_spa_fallback), cfg, session))
    assert res_spa.verdict == "NOT TESTED"


def test_phase1_checks_return_not_tested_on_network_errors():
    cfg = make_config(alternate_urls=["https://alt.example.test/api/chat"])

    def timeout_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=req)

    err_client = mock_client(timeout_handler)
    assert asyncio.run(check_no_credentials(err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_junk_credentials(err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_forged_jwt(err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_unauthenticated_identity_headers(err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_http_methods(err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_alternate_urls(err_client, cfg)).verdict == "NOT TESTED"


def test_404_not_treated_as_auth_refusal_on_chat_endpoint():
    assert 404 not in access_checks.REFUSED_STATUSES
    cfg = make_config()

    # If the chat endpoint returns 404 (e.g. mistyped APP_ENDPOINT), checks return NOT TESTED rather than fake PASS
    not_found_client = mock_client(lambda req: httpx.Response(404))
    assert asyncio.run(check_no_credentials(not_found_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_junk_credentials(not_found_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_forged_jwt(not_found_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_unauthenticated_identity_headers(not_found_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_http_methods(not_found_client, cfg)).verdict == "NOT TESTED"

    # If optional probe path /healthz returns 404 while real endpoints return 401/302, Check #1 still passes
    def healthz_404_others_401(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/healthz":
            return httpx.Response(404)
        return httpx.Response(401)

    res = asyncio.run(check_no_credentials(mock_client(healthz_404_others_401), cfg))
    assert res.verdict == "PASS"
    assert "HTTP 404 (not found)" in "\n".join(res.details)


def test_5xx_server_errors_return_not_tested_rather_than_pass_or_fail(monkeypatch):
    cfg = make_config(alternate_urls=["https://alt.example.test/api/chat"])
    monkeypatch.setattr(
        access_checks,
        "resolve_wrong_audience_token",
        lambda configured="": ("eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJotherIn0.sig", "APP_WRONG_TOKEN"),
    )
    server_err_client = mock_client(lambda req: httpx.Response(500, text="Internal Server Error"))

    assert asyncio.run(check_no_credentials(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_junk_credentials(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_forged_jwt(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_wrong_audience_token(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_unauthenticated_identity_headers(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_cors_preflight(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_http_methods(server_err_client, cfg)).verdict == "NOT TESTED"
    assert asyncio.run(check_alternate_urls(server_err_client, cfg)).verdict == "NOT TESTED"


def test_exposed_response_fields_flags_trace_and_passes_clean_response():
    cfg = make_config(response_path="reply")
    session = SignedInSession(headers={"Cookie": "sid=1"})

    clean_client = mock_client(
        lambda req: httpx.Response(200, json={"reply": "Hello!", "session_id": "abc"})
    )
    trace_client = mock_client(
        lambda req: httpx.Response(
            200,
            json={
                "reply": "Hello!",
                "session_id": "abc",
                "trace": {"system_prompt": "secret instructions", "tool_calls": ["run_realtime_report"]},
            },
        )
    )
    html_client = mock_client(
        lambda req: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<!doctype html><html><body>Sign in</body></html>",
        )
    )

    assert asyncio.run(check_exposed_response_fields(clean_client, cfg, session)).verdict == "PASS"
    res_trace = asyncio.run(check_exposed_response_fields(trace_client, cfg, session))
    assert res_trace.verdict == "REVIEW"
    assert "trace" in res_trace.summary
    assert asyncio.run(check_exposed_response_fields(html_client, cfg, session)).verdict == "NOT TESTED"


def test_openapi_docs_exposed_reviews_signed_in_docs_and_fails_on_unauth_route():
    cfg = make_config()
    session = SignedInSession(headers={"Cookie": "sid=1"})
    spec = {
        "openapi": "3.1.0",
        "paths": {
            "/api/chat": {"post": {}},
            "/api/debug_dump": {"get": {}},
        },
    }

    def signed_in_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/openapi.json":
            return httpx.Response(200, json=spec)
        if req.url.path == "/docs":
            return httpx.Response(200, text="<html>Swagger</html>")
        return httpx.Response(404)

    login_redirect = {"Location": "https://accounts.google.com/ServiceLogin"}
    signed_out_safe = mock_client(lambda req: httpx.Response(302, headers=login_redirect))
    res_review = asyncio.run(
        check_openapi_docs_exposed(mock_client(signed_in_handler), signed_out_safe, cfg, session)
    )
    assert res_review.verdict == "REVIEW"
    assert "/openapi.json" in res_review.summary
    assert "all blocked when logged out" in res_review.summary

    # Discovered route returning 404 or 500 or network error when logged out is treated as "not verified" (REVIEW), not FAIL
    def signed_out_404_on_route(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/debug_dump":
            return httpx.Response(404)
        return httpx.Response(302, headers=login_redirect)

    res_unverified = asyncio.run(
        check_openapi_docs_exposed(mock_client(signed_in_handler), mock_client(signed_out_404_on_route), cfg, session)
    )
    assert res_unverified.verdict == "REVIEW"
    assert "could not be verified when logged out" in res_unverified.summary
    assert "not verified" in "\n".join(res_unverified.details)

    def signed_out_leaky(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/debug_dump":
            return httpx.Response(200, json={"env": "leaked"})
        return httpx.Response(302, headers=login_redirect)

    res_fail = asyncio.run(
        check_openapi_docs_exposed(mock_client(signed_in_handler), mock_client(signed_out_leaky), cfg, session)
    )
    assert res_fail.verdict == "FAIL"
    assert "/api/debug_dump" in res_fail.summary


def test_signed_in_cors_and_cookie_security_flags():
    cfg = make_config()
    session_good = SignedInSession(
        headers={"Cookie": "__Host-GCP_IAP_AUTH_TOKEN=super-secret-value"},
        cookies=[
            {
                "name": "__Host-GCP_IAP_AUTH_TOKEN_A1B2",
                "value": "super-secret-value",
                "domain": "app.example.test",
                "secure": True,
                "httpOnly": True,
                "sameSite": "None",
            }
        ],
    )
    res_good = check_cookie_security_flags(cfg, session_good)
    assert res_good.verdict == "PASS"
    assert "super-secret-value" not in "\n".join(res_good.details)

    session_no_httponly = SignedInSession(
        headers={},
        cookies=[
            {
                "name": "__Host-GCP_IAP_AUTH_TOKEN_A1B2",
                "value": "super-secret-value",
                "domain": "app.example.test",
                "secure": True,
                "httpOnly": False,
                "sameSite": "None",
            }
        ],
    )
    assert check_cookie_security_flags(cfg, session_no_httponly).verdict == "REVIEW"

    session_insecure = SignedInSession(
        headers={},
        cookies=[
            {
                "name": "session_token",
                "value": "super-secret-value",
                "domain": "app.example.test",
                "secure": False,
                "httpOnly": True,
                "sameSite": "Lax",
            }
        ],
    )
    assert check_cookie_security_flags(cfg, session_insecure).verdict == "FAIL"

    safe_cors = mock_client(lambda req: httpx.Response(200, json={"email": "user@example.test"}))
    assert asyncio.run(check_signed_in_cors(safe_cors, cfg, session_good)).verdict == "PASS"

    # When /api/me is 404 and /api/chat is 401 (expired session), return NOT TESTED instead of fake PASS
    unauth_cors = mock_client(lambda req: httpx.Response(404 if req.url.path == "/api/me" else 401))
    assert asyncio.run(check_signed_in_cors(unauth_cors, cfg, session_good)).verdict == "NOT TESTED"


def test_outsider_account_not_tested_when_unset():
    cfg = make_config(outsider_profile=None, outsider_email="danafi2.0@gmail.com")
    res = asyncio.run(check_outsider_account(mock_client(lambda r: httpx.Response(403)), cfg))
    assert res.verdict == "NOT TESTED"
    assert "danafi2.0@gmail.com" in res.summary


def test_outsider_account_passes_when_blocked_and_fails_when_admitted(tmp_path: Path, monkeypatch):
    class FakeLocator:
        first = property(lambda self: self)

        async def wait_for(self, **kwargs):
            raise TimeoutError("not visible")

    class FakePage:
        url = "https://accounts.google.com/rejected"

        async def goto(self, url):
            return SimpleNamespace(status=403)

        async def content(self):
            return "<html>403: You don't have access to this application.</html>"

        def locator(self, selector):
            return FakeLocator()

    class FakeContext:
        pages = [FakePage()]

        async def cookies(self, urls=None):
            return [{"name": "IAP_TEST", "value": "redacted"}]

        async def close(self):
            pass

    class FakePlaywright:
        async def stop(self):
            pass

    async def fake_launch(channel, profile, headless):
        return FakePlaywright(), FakeContext()

    monkeypatch.setattr(access_checks, "launch_browser", fake_launch)
    cfg = make_config(outsider_profile=tmp_path / "outsider", outsider_email="danafi2.0@gmail.com")

    res_pass = asyncio.run(check_outsider_account(mock_client(lambda r: httpx.Response(403)), cfg))
    assert res_pass.verdict == "PASS"
    assert "danafi2.0@gmail.com" in res_pass.summary

    # Also passes when browser hit 403 access-denied and API redirects to Google sign-in
    res_pass_iap_redirect = asyncio.run(
        check_outsider_account(
            mock_client(lambda r: httpx.Response(302, headers={"Location": "https://accounts.google.com/ServiceLogin"})),
            cfg,
        )
    )
    assert res_pass_iap_redirect.verdict == "PASS"

    res_fail = asyncio.run(check_outsider_account(mock_client(lambda r: httpx.Response(200)), cfg))
    assert res_fail.verdict == "FAIL"

    res_500 = asyncio.run(check_outsider_account(mock_client(lambda r: httpx.Response(500)), cfg))
    assert res_500.verdict == "NOT TESTED"

    def raise_timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=req)

    res_err = asyncio.run(check_outsider_account(mock_client(raise_timeout), cfg))
    assert res_err.verdict == "NOT TESTED"

    async def broken_launch(channel, profile, headless):
        raise RuntimeError("browser profile locked")

    monkeypatch.setattr(access_checks, "launch_browser", broken_launch)
    res_launch_err = asyncio.run(check_outsider_account(mock_client(lambda r: httpx.Response(403)), cfg))
    assert res_launch_err.verdict == "NOT TESTED"
    assert "RuntimeError" in res_launch_err.summary


def test_outsider_account_not_tested_when_outsider_never_signs_in(tmp_path: Path, monkeypatch):
    class FakeLocator:
        first = property(lambda self: self)

        async def wait_for(self, **kwargs):
            raise TimeoutError("not visible")

    class UnauthenticatedLoginPage:
        url = "https://accounts.google.com/v3/signin/identifier"

        async def goto(self, url):
            return SimpleNamespace(status=200)

        async def content(self):
            return "<html><body>Sign in with your Google Account</body></html>"

        def locator(self, selector):
            return FakeLocator()

    class EmptyCookieContext:
        pages = [UnauthenticatedLoginPage()]

        async def cookies(self, urls=None):
            # Target app has no cookies when sitting on Google login screen
            return []

        async def close(self):
            pass

    class FakePlaywright:
        async def stop(self):
            pass

    async def fake_launch(channel, profile, headless):
        return FakePlaywright(), EmptyCookieContext()

    monkeypatch.setattr(access_checks, "launch_browser", fake_launch)
    cfg = make_config(
        outsider_profile=tmp_path / "outsider",
        outsider_email="danafi2.0@gmail.com",
        headless=True,
    )

    # Even if an anonymous POST /api/chat would get a 302 to Google sign-in, Check #14 must return NOT TESTED
    iap_redirect_client = mock_client(
        lambda r: httpx.Response(302, headers={"Location": "https://accounts.google.com/o/oauth2/v2/auth"})
    )
    res = asyncio.run(check_outsider_account(iap_redirect_client, cfg))
    assert res.verdict == "NOT TESTED"
    assert "Outsider sign-in did not complete" in res.summary


def test_non_login_redirects_return_not_tested_instead_of_false_pass(monkeypatch):
    cfg = make_config(alternate_urls=["https://alt.example.test/api/chat"])
    monkeypatch.setattr(
        access_checks,
        "resolve_wrong_audience_token",
        lambda configured="": ("eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJotherIn0.sig", "APP_WRONG_TOKEN"),
    )

    # Trailing-slash redirect (/api/chat -> /api/chat/) and http->https redirect are NOT auth refusals
    for non_auth_resp in [
        httpx.Response(307, headers={"Location": "/api/chat/"}),
        httpx.Response(301, headers={"Location": "https://app.example.test/api/chat"}),
        httpx.Response(302),
    ]:
        client = mock_client(lambda req, r=non_auth_resp: r)
        assert asyncio.run(check_no_credentials(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_junk_credentials(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_forged_jwt(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_wrong_audience_token(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_unauthenticated_identity_headers(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_cors_preflight(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_http_methods(client, cfg)).verdict == "NOT TESTED"
        assert asyncio.run(check_alternate_urls(client, cfg)).verdict == "NOT TESTED"


def test_crash_inside_check_logs_traceback_in_details_and_stderr(capsys):
    def buggy_check():
        raise ZeroDivisionError("simulated kit bug")

    res = asyncio.run(
        access_checks._run_check_safely(
            "Phase 1: Signed-Out Checks (No Login)",
            "Buggy check",
            "API2:2023 Broken Authentication",
            buggy_check,
        )
    )
    assert res.verdict == "NOT TESTED"
    assert "ZeroDivisionError: simulated kit bug" in res.summary
    assert any("ZeroDivisionError" in d for d in res.details)
    assert any("in buggy_check" in d for d in res.details)
    captured = capsys.readouterr()
    assert "ZeroDivisionError: simulated kit bug" in captured.err


def test_run_all_checks_preserves_phase1_results_when_browser_signin_fails(monkeypatch):
    cfg = make_config()
    monkeypatch.setattr(access_checks, "resolve_wrong_audience_token", lambda configured="": ("", ""))

    orig_client = httpx.AsyncClient
    monkeypatch.setattr(
        access_checks.httpx,
        "AsyncClient",
        lambda **kw: orig_client(transport=httpx.MockTransport(lambda r: httpx.Response(401)), follow_redirects=False),
    )

    async def failing_session_getter(config):
        raise TimeoutError("Sign-in timed out after 300s")

    results = asyncio.run(access_checks.run_all_checks(cfg, session_getter=failing_session_getter))
    assert len(results) == 14
    # Phase 1 results (first 8 checks) are preserved intact
    assert results[0].verdict == "PASS"
    assert results[1].verdict == "PASS"
    assert results[2].verdict == "PASS"
    # Phase 2 session checks (#9-#13) record the browser sign-in error as NOT TESTED
    for res in results[8:13]:
        assert res.verdict == "NOT TESTED"
        assert "Browser sign-in failed (TimeoutError" in res.summary


def test_end_to_end_main_writes_markdown_report(tmp_path: Path, clean_env, monkeypatch):
    clean_env.setenv("APP_BROWSER_URL", "https://app.example.test/")
    clean_env.setenv("APP_ENDPOINT", "/api/chat")
    monkeypatch.setattr(access_checks, "resolve_wrong_audience_token", lambda configured="": ("", ""))

    orig_client = httpx.AsyncClient
    monkeypatch.setattr(
        access_checks.httpx,
        "AsyncClient",
        lambda **kw: orig_client(transport=httpx.MockTransport(lambda r: httpx.Response(401)), follow_redirects=False),
    )

    out_file = tmp_path / "report.md"
    exit_code = asyncio.run(access_checks.main(["--phase1-only", "--out", str(out_file)]))
    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# Application Access-Control & API Configuration Report" in content
    assert "API2:2023 Broken Authentication" in content
    # --phase1-only report does not claim Phase 2 checks #9-#14 were run
    assert "Phase 2 (Signed-In & Outsider Checks #9–#14)" not in content
    assert "2 not tested" in content
    assert "- **Not Tested (2):**" in content


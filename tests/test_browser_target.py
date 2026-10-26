"""The browser transport for apps: drives a real browser against a local stand-in chat page."""

import asyncio
import uuid
from pathlib import Path

import pytest
from pyrit.models import Message, MessagePiece

from harnesses.pyrit_campaigns import target_factory
from harnesses.pyrit_campaigns.browser_target import BrowserChatTarget
from helpers import FakeModel, NeverFlags, nothing

PAGE = (Path(__file__).parent / "fixtures" / "fake_chat.html").resolve().as_uri()


def _browser_available() -> bool:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return False

    async def launch(channel):
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel=channel, headless=True)
            await browser.close()

    for channel in ("chrome", "msedge"):
        try:
            asyncio.run(launch(channel))
            return True
        except Exception:
            pass
    return False


@pytest.fixture(scope="session")
def _browser():
    """Checked once, and only when a browser test runs: the check launches a real browser."""
    if not _browser_available():
        pytest.skip("needs Playwright and Chrome or Edge")


needs_browser = pytest.mark.usefixtures("_browser")


@pytest.fixture(autouse=True)
def _memory(memory):
    """PyRIT targets need its database, even when nothing is saved."""
    return memory


def make_target(tmp_path, url=PAGE, **overrides):
    settings = dict(url=url, input_selector="#message", reply_selector=".reply", profile=tmp_path / "profile",
                    headless=True, settle=0.5, timeout=15, ready_timeout=5)
    settings.update(overrides)
    return BrowserChatTarget(**settings)


async def send(target, text, conversation_id):
    piece = MessagePiece(role="user", original_value=text, conversation_id=conversation_id)
    responses = await target.send_prompt_async(message=Message(message_pieces=[piece]))
    return responses[0].get_value()


def run(target, script):
    """Run a script of sends in one event loop, as a campaign does, and close the browser."""
    async def go():
        try:
            return await script()
        finally:
            await target.cleanup_target_async()
    return asyncio.run(go())


FULL_REPLY = "turn 1: you said Hello there"


@needs_browser
def test_a_reply_that_pauses_mid_answer_is_read_once_it_has_finished(tmp_path):
    target = make_target(tmp_path, url=PAGE + "?pause=800", settle=1.5)
    assert run(target, lambda: send(target, "Hello there", str(uuid.uuid4()))) == FULL_REPLY


@needs_browser
def test_the_busy_indicator_holds_the_read_even_without_a_settle_wait(tmp_path):
    target = make_target(tmp_path, url=PAGE + "?pause=800", settle=0.0, busy_selector="#busy")
    assert run(target, lambda: send(target, "Hello there", str(uuid.uuid4()))) == FULL_REPLY


@needs_browser
def test_without_either_safeguard_a_paused_reply_is_cut_short(tmp_path):
    # Shows the two tests above can fail: this is what the safeguards prevent.
    target = make_target(tmp_path, url=PAGE + "?pause=800", settle=0.0)
    reply = run(target, lambda: send(target, "Hello there", str(uuid.uuid4())))
    assert reply != FULL_REPLY and FULL_REPLY.startswith(reply)


@needs_browser
def test_follow_ups_share_a_tab_and_a_new_conversation_starts_fresh(tmp_path):
    target = make_target(tmp_path, send_selector="#send")
    a, b = str(uuid.uuid4()), str(uuid.uuid4())

    async def script():
        return [await send(target, "first", a), await send(target, "second", a), await send(target, "other", b)]

    assert run(target, script) == ["turn 1: you said first", "turn 2: you said second", "turn 1: you said other"]


@needs_browser
def test_the_run_waits_while_you_sign_in(tmp_path, capsys):
    target = make_target(tmp_path, url=PAGE + "?signin=2500", ready_timeout=0.5)
    assert run(target, lambda: send(target, "hi", str(uuid.uuid4()))) == "turn 1: you said hi"
    assert "sign in there" in capsys.readouterr().out


@needs_browser
def test_a_new_chat_button_is_clicked_before_the_first_message(tmp_path):
    target = make_target(tmp_path, new_chat_selector="#new-chat")
    assert run(target, lambda: send(target, "hi", str(uuid.uuid4()))) == "turn 1: you said hi"


@needs_browser
def test_a_conversation_whose_tab_was_closed_cannot_carry_on(tmp_path):
    target = make_target(tmp_path, max_tabs=1)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())

    async def script():
        await send(target, "one", a)
        await send(target, "two", b)    # closes a's tab
        await send(target, "three", a)

    with pytest.raises(Exception, match="APP_BROWSER_MAX_TABS"):
        run(target, script)


@needs_browser
def test_a_wrong_selector_says_which_setting_to_check(tmp_path):
    target = make_target(tmp_path, input_selector="#nope", ready_timeout=0.3, sign_in_timeout=0.5)
    with pytest.raises(Exception, match="APP_BROWSER_INPUT"):
        run(target, lambda: send(target, "hi", str(uuid.uuid4())))


@needs_browser
def test_the_scan_runs_against_a_web_page_with_one_fresh_chat_per_probe(tmp_path, memory, clean_env, monkeypatch):
    from harnesses.pyrit_campaigns import single_turn_scan

    clean_env.setenv("RT_PROFILE", "internal_productivity")
    clean_env.setenv("RT_PROVIDER", "app")
    for name in ("initialize_pyrit_async", "check_models"):
        monkeypatch.setattr(single_turn_scan, name, nothing)
    monkeypatch.setattr(single_turn_scan, "build_target", lambda: make_target(tmp_path))
    monkeypatch.setattr(single_turn_scan, "build_scoring_target", lambda: None)
    monkeypatch.setattr(single_turn_scan, "model_name", lambda t: "fake-judge" if t is None else target_factory.model_name(t))
    monkeypatch.setattr(single_turn_scan, "build_rubric_scorers",
                        lambda rubrics, judge, *rest: {k: NeverFlags(category=k) for k in rubrics})
    runs = []
    monkeypatch.setattr(single_turn_scan, "write_after_run", runs.append)

    assert asyncio.run(single_turn_scan.main()) == 0
    results = memory.get_attack_results(labels={"rt_run_id": runs[0]})
    replies = [r.last_response.converted_value for r in results]
    assert len(replies) > 10
    # Calibration ran too, and no probe saw another probe's conversation.
    assert any((r.labels or {}).get("rt_calibration") for r in results)
    assert all(reply.startswith("turn 1: you said") for reply in replies), replies


# --- settings: no browser needed -------------------------------------------------

def test_making_room_for_a_conversation_never_closes_a_tab_still_answering(tmp_path, memory):
    class Tab:
        url = "https://app.example/"
        closed = False

        async def close(self):
            self.closed = True

    class Context:
        pages = []

        async def new_page(self):
            return Tab()

    target = make_target(tmp_path, max_tabs=2)
    target._context = Context()
    target._start = target._open_chat = lambda *args: nothing()

    async def go():
        a, _ = await target._tab_for("a")       # a slow reply, still coming
        b, _ = await target._tab_for("b")
        target._answering.discard("b")          # b's reply is in
        c, _ = await target._tab_for("c")       # room for c: b goes, not a
        assert b.closed and not a.closed
        await target._tab_for("d")              # a and c both answering: nothing closes
        assert not a.closed and not c.closed and len(target._tabs) == 3

    asyncio.run(go())


@pytest.mark.parametrize("browser, note", [(True, "real multi-turn conversation"), (False, "no memory of its own")])
def test_the_red_team_campaign_says_whether_the_app_remembers_the_conversation(browser, note, clean_env, monkeypatch, capsys):
    from types import SimpleNamespace

    from harnesses.pyrit_campaigns import multi_turn_red_team as rta
    from harnesses.pyrit_campaigns.target_factory import AppChatTarget

    class Executor:
        def __init__(self, **kwargs):
            pass

        async def execute_attack_async(self, **kwargs):
            return SimpleNamespace(completed_results=[], incomplete_objectives=[])

    app = (BrowserChatTarget(url="https://example.test/chat", input_selector="textarea", reply_selector=".reply")
           if browser else AppChatTarget(endpoint="https://example.test/chat"))
    clean_env.setenv("RT_PROFILE", "public_conversational")
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("RT_JUDGE_PROVIDER", "gemini")
    for name in ("initialize_pyrit_async", "check_models", "close_target"):
        monkeypatch.setattr(rta, name, nothing)
    monkeypatch.setattr(rta, "build_target", lambda role="target", **kw: app if role == "target" else FakeModel())
    monkeypatch.setattr(rta, "build_scoring_target", lambda: FakeModel())
    monkeypatch.setattr(rta, "model_name", lambda t: "fake")
    monkeypatch.setattr(rta, "write_after_run", lambda run_id: None)
    monkeypatch.setattr(rta, "AttackExecutor", Executor)

    asyncio.run(rta.main())
    assert note in capsys.readouterr().out

def test_the_browser_transport_is_chosen_by_app_transport(clean_env):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_TRANSPORT", "browser")
    clean_env.setenv("APP_BROWSER_URL", "https://example.test/chat")
    clean_env.setenv("APP_BROWSER_INPUT", "textarea")
    clean_env.setenv("APP_BROWSER_REPLY", ".reply")
    target = target_factory.build_target()
    assert isinstance(target, BrowserChatTarget)
    assert target_factory.model_name(target) == "app:example.test/chat (browser)"


def test_the_browser_transport_names_the_settings_it_is_missing(clean_env):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_TRANSPORT", "browser")
    clean_env.setenv("APP_BROWSER_URL", "https://example.test/chat")
    with pytest.raises(SystemExit, match="APP_BROWSER_INPUT, APP_BROWSER_REPLY"):
        target_factory.build_target()


def test_an_unknown_transport_is_refused(clean_env):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_TRANSPORT", "carrier-pigeon")
    with pytest.raises(SystemExit, match="APP_TRANSPORT"):
        target_factory.build_target()


@pytest.mark.parametrize("sends_history", [True, False])
def test_the_red_team_campaign_says_when_the_kit_sends_the_conversation(sends_history, clean_env, monkeypatch, capsys):
    from types import SimpleNamespace

    from harnesses.pyrit_campaigns import multi_turn_red_team as rta
    from harnesses.pyrit_campaigns.target_factory import AppChatTarget

    class Executor:
        def __init__(self, **kwargs):
            pass

        async def execute_attack_async(self, **kwargs):
            return SimpleNamespace(completed_results=[], incomplete_objectives=[])

    template = '{"message": "{PROMPT}", "history": "{HISTORY}"}' if sends_history else '{"message": "{PROMPT}"}'
    app = AppChatTarget(endpoint="https://example.test/chat", request_template=template)
    clean_env.setenv("RT_PROFILE", "public_conversational")
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("RT_JUDGE_PROVIDER", "gemini")
    for name in ("initialize_pyrit_async", "check_models", "close_target"):
        monkeypatch.setattr(rta, name, nothing)
    monkeypatch.setattr(rta, "build_target", lambda role="target", **kw: app if role == "target" else FakeModel())
    monkeypatch.setattr(rta, "build_scoring_target", lambda: FakeModel())
    monkeypatch.setattr(rta, "model_name", lambda t: "fake")
    monkeypatch.setattr(rta, "write_after_run", lambda run_id: None)
    monkeypatch.setattr(rta, "AttackExecutor", Executor)

    asyncio.run(rta.main())
    out = capsys.readouterr().out
    assert ("sends the app the whole conversation" in out) == sends_history
    assert ("no memory of its own" in out) == (not sends_history)


def test_the_browser_target_says_it_keeps_history_but_takes_no_system_prompt():
    from pyrit.prompt_target import CapabilityName

    target = BrowserChatTarget(url="https://example.test/chat", input_selector="textarea", reply_selector=".reply")
    assert target.configuration.includes(capability=CapabilityName.MULTI_TURN)
    assert not target.configuration.includes(capability=CapabilityName.EDITABLE_HISTORY)
    assert not target.configuration.includes(capability=CapabilityName.SYSTEM_PROMPT)


# --- APP_TRANSPORT=browser-api: the app's API, called from its signed-in page ----------

APP_PAGE = '<!doctype html><html><body><textarea id="message"></textarea></body></html>'


@pytest.fixture
def signed_in_api():
    """A stand-in app behind a sign-in, like an API behind an identity-aware proxy.

    GET /chat signs the browser in (a cookie) and shows the page. With ?signin=MS a
    browser that is not signed in is first sent to a sign-in page on another site, which
    sends it back after MS milliseconds. POST /api/chat answers only a signed-in browser,
    redirecting any other request to sign in, and echoes the history it was sent.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, quote, urlparse

    state = {"page_loads": 0, "requests": []}

    class SignInSite(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            back, ms = query["back"][0], int(query["ms"][0])
            page = f"<p>Sign in</p><script>setTimeout(() => location = {json.dumps(back)}, {ms})</script>"
            self._send(200, page.encode(), "text/html")

        def _send(self, status, body, kind, extra=()):
            self.send_response(status)
            for name, value in extra:
                self.send_header(name, value)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class App(SignInSite):
        def signed_in(self):
            return "session=ok" in self.headers.get("Cookie", "")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path != "/chat":    # the browser's own extras, e.g. /favicon.ico
                return self._send(404, b"", "text/plain")
            ms = parse_qs(url.query).get("signin", [None])[0]
            if ms and not self.signed_in():
                back = quote(f"http://127.0.0.1:{app.server_port}/chat")
                where = f"http://127.0.0.1:{sign_in.server_port}/login?back={back}&ms={ms}"
                return self._send(302, b"", "text/plain", [("Location", where)])
            state["page_loads"] += 1
            self._send(200, APP_PAGE.encode(), "text/html", [("Set-Cookie", "session=ok; Path=/")])

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            if not self.signed_in() or "signin" in self.path:
                return self._send(302, b"", "text/plain", [("Location", "/chat")])
            sent = json.loads(body)
            state["requests"].append(sent)
            turn = 1 + sum(1 for t in sent.get("history", []) if t["role"] == "user")
            reply = {"reply": f"turn {turn}: you said {sent['message']}", "usage": {"total_tokens": 1}}
            self._send(200, json.dumps(reply).encode(), "application/json")

    app = HTTPServer(("127.0.0.1", 0), App)
    sign_in = HTTPServer(("127.0.0.1", 0), SignInSite)
    for server in (app, sign_in):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{app.server_port}/chat"
    yield state
    for server in (app, sign_in):
        server.shutdown()


def make_api_target(tmp_path, page_url, **overrides):
    from harnesses.pyrit_campaigns.browser_target import BrowserApiTarget

    settings = dict(
        page_url=page_url, endpoint="/api/chat", profile=tmp_path / "profile", headless=True, ready_timeout=0.5,
        sign_in_timeout=15, request_template='{"message": "{PROMPT}", "history": "{HISTORY}", "session_id": "kit"}',
        history_turn='{"role": "{ROLE}", "text": "{TEXT}"}', history_roles="user,model", response_path="reply",
        timeout=15,
    )
    settings.update(overrides)
    return BrowserApiTarget(**settings)


async def converse(target, memory, conversation_id, text):
    """Send one turn and store both sides, as PyRIT does after each exchange of an attack."""
    piece = MessagePiece(role="user", original_value=text, conversation_id=conversation_id)
    message = Message(message_pieces=[piece])
    reply = (await target.send_prompt_async(message=message))[0]
    memory.add_message_to_memory(request=message)
    memory.add_message_to_memory(request=reply)
    return reply.get_value()


@needs_browser
def test_browser_api_sends_the_conversation_the_kit_holds(tmp_path, memory, signed_in_api):
    target = make_api_target(tmp_path, signed_in_api["url"])
    a, b = str(uuid.uuid4()), str(uuid.uuid4())

    async def script():
        return [await converse(target, memory, a, "first"), await converse(target, memory, a, "second"),
                await converse(target, memory, b, "other")]

    assert run(target, script) == ["turn 1: you said first", "turn 2: you said second", "turn 1: you said other"]
    # The second turn carried the first exchange, in the app's own shape.
    assert signed_in_api["requests"][1]["history"] == [
        {"role": "user", "text": "first"}, {"role": "model", "text": "turn 1: you said first"},
    ]
    # One page, signed in once, serves every conversation.
    assert signed_in_api["page_loads"] == 1


@needs_browser
def test_browser_api_waits_while_you_sign_in_on_another_site(tmp_path, memory, signed_in_api, capsys):
    target = make_api_target(tmp_path, signed_in_api["url"] + "?signin=2000")
    assert run(target, lambda: converse(target, memory, str(uuid.uuid4()), "hi")) == "turn 1: you said hi"
    assert "sign in there" in capsys.readouterr().out


@needs_browser
def test_browser_api_signs_in_again_when_the_sign_in_lapses(tmp_path, memory, signed_in_api, capsys):
    target = make_api_target(tmp_path, signed_in_api["url"])

    async def script():
        first = await converse(target, memory, str(uuid.uuid4()), "one")
        await target._context.clear_cookies()    # the session expires
        return first, await converse(target, memory, str(uuid.uuid4()), "two")

    assert run(target, script) == ("turn 1: you said one", "turn 1: you said two")
    assert signed_in_api["page_loads"] == 2
    assert "reloading its page" in capsys.readouterr().out


@needs_browser
def test_browser_api_says_so_when_the_app_keeps_turning_requests_away(tmp_path, memory, signed_in_api):
    # An APP_ENDPOINT that always redirects: reloading the page does not help.
    target = make_api_target(tmp_path, signed_in_api["url"], endpoint="/api/chat?signin=1")
    with pytest.raises(Exception, match="redirected the request"):
        run(target, lambda: converse(target, memory, str(uuid.uuid4()), "hi"))


def test_the_browser_api_transport_is_chosen_by_app_transport(clean_env):
    from pyrit.prompt_target import CapabilityName

    from harnesses.pyrit_campaigns.browser_target import BrowserApiTarget

    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_TRANSPORT", "browser-api")
    clean_env.setenv("APP_BROWSER_URL", "https://example.test/chat")
    clean_env.setenv("APP_ENDPOINT", "/api/chat")
    clean_env.setenv("APP_REQUEST_TEMPLATE", '{"message": "{PROMPT}", "history": "{HISTORY}"}')
    target = target_factory.build_target()
    assert isinstance(target, BrowserApiTarget)
    # A path is taken on the page's own site.
    assert target._endpoint == "https://example.test/api/chat"
    assert target_factory.model_name(target) == "app:example.test/api/chat (browser-api)"
    assert target.configuration.includes(capability=CapabilityName.EDITABLE_HISTORY)


def test_the_browser_api_transport_names_the_settings_it_is_missing(clean_env):
    clean_env.setenv("RT_PROVIDER", "app")
    clean_env.setenv("APP_TRANSPORT", "browser-api")
    with pytest.raises(SystemExit, match="APP_BROWSER_URL, APP_ENDPOINT"):
        target_factory.build_target()

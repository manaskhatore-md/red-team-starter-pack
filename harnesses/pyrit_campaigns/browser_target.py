"""A deployed chat app reached through a real browser, for apps whose API the kit cannot call alone.

Two ways in. APP_TRANSPORT=browser (BrowserChatTarget, below) types into the app's chat
page as a person would. APP_TRANSPORT=browser-api (BrowserApiTarget, at the end) sends
the app's own API requests from inside its signed-in page, so the kit can supply the
conversation history itself; see that class for its settings.

Used as RT_PROVIDER=app with APP_TRANSPORT=browser, so the scan treats it as an app in
every other way (calibration, fixed replies, the app's own system prompt). Settings, all
from .env:

    APP_BROWSER_URL       the chat page
    APP_BROWSER_INPUT     CSS selector of the box you type in
    APP_BROWSER_REPLY     CSS selector matching each of the app's replies, and not your messages
    APP_BROWSER_SEND      CSS selector of the send button. Unset: press Enter in the box
    APP_BROWSER_NEW_CHAT  CSS selector to click for a fresh chat after the page loads.
                          Unset: loading the page starts a fresh chat
    APP_BROWSER_BUSY      CSS selector shown while the app is still answering, e.g. a spinner
    APP_BROWSER_CHANNEL   the installed browser to drive: chrome or msedge. Unset: Chrome,
                          or Edge where Chrome is not installed
    APP_BROWSER_PROFILE   folder that keeps your sign-in between runs. Default
                          ~/.rt_browser_profile/<browser>: Chrome and Edge cannot share one
    APP_BROWSER_HEADLESS  1 hides the window. Leave unset for the first run, so you can sign in
    APP_BROWSER_MAX_TABS  conversations open at once; more only while all of them are still
                          waiting on a reply. Default 4
    APP_BROWSER_SETTLE    seconds a reply's text must stay unchanged to count as finished. Default 1.5
    APP_BROWSER_SIGN_IN_TIMEOUT  seconds to wait for you to sign in. Default 300
    APP_TIMEOUT           seconds to wait for a reply. Default 60
    APP_MAX_RPM           requests per minute, to stay inside the app's throttling

Each conversation gets its own tab, so single-turn probes cannot see each other and a
multi-turn attack keeps its history in the app, the way a real user's chat does.

The profile folder holds your signed-in session for the app. Anyone with a copy of it
can use the app as you until the session expires, which is why it lives outside the repo.
"""

import asyncio
import os
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urljoin, urlparse

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration, limit_requests_per_minute

from harnesses.pyrit_campaigns.target_factory import AppChatTarget

DEFAULT_PROFILE = Path.home() / ".rt_browser_profile"
POLL_SECONDS = 0.25
# How long to wait for the chat box before saying the page may need a sign-in.
READY_SECONDS = 15.0


async def launch_browser(channel: str, profile: Path | None, headless: bool) -> tuple:
    """Start Playwright and the installed browser with its kept profile: (playwright, context)."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise SystemExit("APP_TRANSPORT=browser and browser-api need Playwright: pip install playwright") from e
    playwright = await async_playwright().start()
    # Unset means Chrome, or Edge where Chrome is not installed (Edge ships with Windows).
    channels = [channel] if channel else ["chrome", "msedge"]
    for i, name in enumerate(channels):
        try:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile or DEFAULT_PROFILE / name),
                channel=None if name == "chromium" else name,
                headless=headless,
            )
            return playwright, context
        except Exception as e:
            if i == len(channels) - 1:
                await playwright.stop()
                raise RuntimeError(
                    f"Could not start {' or '.join(channels)}: {e}. Set APP_BROWSER_CHANNEL to an "
                    "installed browser (chrome or msedge)."
                ) from e


SIGN_IN_NOTE = (
    "\nWaiting for the app's page. If the browser window shows a sign-in page, sign in there:\n"
    "the run carries on by itself once the app's page is back (up to {timeout:.0f}s)."
)


async def wait_for_element(tab, selector: str, ready_timeout: float, sign_in_timeout: float) -> None:
    """Wait for APP_BROWSER_INPUT on the page, giving you time to sign in if it does not show."""
    from playwright.async_api import TimeoutError as PlaywrightTimeout

    box = tab.locator(selector).first
    try:
        await box.wait_for(state="visible", timeout=ready_timeout * 1000)
        return
    except PlaywrightTimeout:
        pass
    print(SIGN_IN_NOTE.format(timeout=sign_in_timeout))
    try:
        await box.wait_for(state="visible", timeout=sign_in_timeout * 1000)
    except PlaywrightTimeout as e:
        raise RuntimeError(
            f"No element matching APP_BROWSER_INPUT={selector!r} appeared at {tab.url}. If the page "
            "needs a sign-in, run once without APP_BROWSER_HEADLESS so you can sign in; otherwise the "
            "selector does not match the page."
        ) from e


class BrowserChatTarget(PromptTarget):
    """Types each probe into the app's chat box and reads the reply off the page."""

    # The app keeps each tab's history itself. It cannot take a system prompt or have
    # its history edited, so Crescendo (which deletes refused turns) still cannot run.
    _DEFAULT_CONFIGURATION = TargetConfiguration(capabilities=TargetCapabilities(supports_multi_turn=True))

    def __init__(
        self,
        *,
        url: str,
        input_selector: str,
        reply_selector: str,
        send_selector: str = "",
        new_chat_selector: str = "",
        busy_selector: str = "",
        channel: str = "",
        profile: Path | None = None,
        headless: bool = False,
        max_tabs: int = 4,
        settle: float = 1.5,
        timeout: float = 60.0,
        sign_in_timeout: float = 300.0,
        ready_timeout: float = READY_SECONDS,
        max_requests_per_minute: int | None = None,
    ):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", "file") or not (parsed.netloc or parsed.path):
            raise SystemExit(f"APP_BROWSER_URL={url!r} is not an http(s) or file URL.")
        super().__init__(
            endpoint=url,
            model_name=f"app:{parsed.netloc}{parsed.path} (browser)",
            max_requests_per_minute=max_requests_per_minute,
        )
        self._url = url
        self._input = input_selector
        self._reply = reply_selector
        self._send = send_selector
        self._new_chat = new_chat_selector
        self._busy = busy_selector
        self._channel = channel
        self._profile = Path(profile) if profile else None
        self._headless = headless
        self._max_tabs = max(1, max_tabs)
        self._settle = settle
        self._timeout = timeout
        self._sign_in_timeout = sign_in_timeout
        self._ready_timeout = ready_timeout
        self._playwright = None
        self._context = None
        # conversation_id -> (tab, lock), least recently used first
        self._tabs: OrderedDict[str, tuple] = OrderedDict()
        self._closed: set[str] = set()
        # Conversations with a probe on its way or a reply still coming: their tabs
        # are never closed to make room.
        self._answering: set[str] = set()
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "BrowserChatTarget":
        missing = [n for n in ("APP_BROWSER_URL", "APP_BROWSER_INPUT", "APP_BROWSER_REPLY") if not os.getenv(n, "").strip()]
        if missing:
            raise SystemExit(
                f"APP_TRANSPORT=browser needs {', '.join(missing)} - see browser_target.py for what each one is."
            )
        rpm = os.getenv("APP_MAX_RPM", "").strip()
        return cls(
            url=os.environ["APP_BROWSER_URL"].strip(),
            input_selector=os.environ["APP_BROWSER_INPUT"].strip(),
            reply_selector=os.environ["APP_BROWSER_REPLY"].strip(),
            send_selector=os.getenv("APP_BROWSER_SEND", "").strip(),
            new_chat_selector=os.getenv("APP_BROWSER_NEW_CHAT", "").strip(),
            busy_selector=os.getenv("APP_BROWSER_BUSY", "").strip(),
            channel=os.getenv("APP_BROWSER_CHANNEL", "").strip(),
            profile=os.getenv("APP_BROWSER_PROFILE", "").strip() or None,
            headless=os.getenv("APP_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes"),
            max_tabs=int(os.getenv("APP_BROWSER_MAX_TABS") or 4),
            settle=float(os.getenv("APP_BROWSER_SETTLE") or 1.5),
            timeout=float(os.getenv("APP_TIMEOUT") or 60),
            sign_in_timeout=float(os.getenv("APP_BROWSER_SIGN_IN_TIMEOUT") or 300),
            max_requests_per_minute=int(rpm) if rpm else None,
        )

    async def _start(self) -> None:
        if self._context is None:
            self._playwright, self._context = await launch_browser(self._channel, self._profile, self._headless)

    async def _tab_for(self, conversation_id: str) -> tuple:
        """The tab holding this conversation, opening a fresh chat for a new one."""
        async with self._lock:
            await self._start()
            if conversation_id in self._tabs:
                self._tabs.move_to_end(conversation_id)
                self._answering.add(conversation_id)
                return self._tabs[conversation_id]
            if conversation_id in self._closed:
                raise RuntimeError(
                    "A conversation continued after its tab was closed to stay under APP_BROWSER_MAX_TABS "
                    f"({self._max_tabs}). Raise APP_BROWSER_MAX_TABS or lower RT_MAX_CONCURRENCY."
                )
            # A new profile opens with one blank tab: use it rather than leave it empty.
            in_use = {t for t, _ in self._tabs.values()}
            blank = [p for p in self._context.pages if p.url == "about:blank" and p not in in_use]
            tab = blank[0] if blank else await self._context.new_page()
            # Close the oldest idle tab only once the new tab is open: closing a
            # browser's last tab can close the browser. A tab still waiting on a reply
            # is never closed - a slow reply would be cut off mid-answer - so with every
            # tab busy, the browser holds one more than APP_BROWSER_MAX_TABS for a while.
            while len(self._tabs) >= self._max_tabs:
                old_id = next((cid for cid in self._tabs if cid not in self._answering), None)
                if old_id is None:
                    break
                old_tab, _ = self._tabs.pop(old_id)
                self._closed.add(old_id)
                await old_tab.close()
            await self._open_chat(tab)
            self._tabs[conversation_id] = (tab, asyncio.Lock())
            self._answering.add(conversation_id)
            return self._tabs[conversation_id]

    async def _open_chat(self, tab) -> None:
        await tab.goto(self._url)
        await self._wait_for_chat_box(tab)
        if self._new_chat:
            await tab.locator(self._new_chat).first.click()
            await self._wait_for_chat_box(tab)

    async def _wait_for_chat_box(self, tab) -> None:
        await wait_for_element(tab, self._input, self._ready_timeout, self._sign_in_timeout)

    async def _reply_text(self, tab, first: int) -> str:
        replies = tab.locator(self._reply)
        parts = [await replies.nth(i).inner_text() for i in range(first, await replies.count())]
        return "\n\n".join(p.strip() for p in parts if p.strip())

    async def _busy_shown(self, tab) -> bool:
        if not self._busy:
            return False
        busy = tab.locator(self._busy)
        return await busy.count() > 0 and await busy.first.is_visible()

    async def ask(self, tab, prompt: str) -> str:
        """Send one message in this tab and return the app's reply once it has finished."""
        replies = tab.locator(self._reply)
        before = await replies.count()
        box = tab.locator(self._input).first
        await box.fill(prompt)
        if self._send:
            await tab.locator(self._send).first.click()
        else:
            await box.press("Enter")

        deadline = time.monotonic() + self._timeout
        while await replies.count() <= before:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"No reply matching APP_BROWSER_REPLY={self._reply!r} appeared within {self._timeout:.0f}s."
                )
            await asyncio.sleep(POLL_SECONDS)

        # Streamed replies grow in place: wait until the text stops changing.
        text, unchanged_since = None, time.monotonic()
        while True:
            current = await self._reply_text(tab, before)
            now = time.monotonic()
            if current != text or not current or await self._busy_shown(tab):
                text, unchanged_since = current, now
            elif now - unchanged_since >= self._settle:
                return current
            if now > deadline:
                raise RuntimeError(f"The app's reply was still changing after {self._timeout:.0f}s.")
            await asyncio.sleep(POLL_SECONDS)

    @limit_requests_per_minute
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]
        tab, tab_lock = await self._tab_for(request.conversation_id)
        try:
            async with tab_lock:
                text = await self.ask(tab, request.converted_value)
        finally:
            self._answering.discard(request.conversation_id)
        return [construct_response_from_request(request=request, response_text_pieces=[text])]

    async def cleanup_target_async(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._tabs.clear()
        self._answering.clear()


# Runs inside the app's page. Same-origin requests carry the page's sign-in cookies.
# redirect: "manual" makes a bounce to a sign-in page visible instead of followed.
FETCH_SCRIPT = """
async ({url, body, headers, timeoutMs}) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(url, {method: "POST", headers, body, redirect: "manual", signal: controller.signal});
    if (r.type === "opaqueredirect") return {status: 0, body: "", problem: "redirect"};
    return {status: r.status, body: await r.text(), problem: ""};
  } catch (e) {
    return {status: 0, body: String(e), problem: e.name === "AbortError" ? "timeout" : "network"};
  } finally {
    clearTimeout(timer);
  }
}
"""


class BrowserApiTarget(AppChatTarget):
    """Calls the app's API from inside its signed-in web page, for an API behind a browser sign-in.

    Used as RT_PROVIDER=app with APP_TRANSPORT=browser-api. Each request is built exactly
    as over plain HTTP (APP_ENDPOINT, APP_REQUEST_TEMPLATE, APP_RESPONSE_PATH and the
    {HISTORY} settings: see AppChatTarget), but the browser sends it, so it carries the
    sign-in the browser holds. That reaches an API that admits only a signed-in browser
    (single sign-on, an identity-aware proxy) with no token to obtain or refresh. With
    {HISTORY} in the template the kit holds the conversation and can rewrite it, so
    Crescendo runs, which it cannot through APP_TRANSPORT=browser. Settings beyond
    AppChatTarget's:

        APP_BROWSER_URL       the app's page, opened once to sign in. APP_ENDPOINT may be a
                              path on it, e.g. /api/chat
        APP_BROWSER_INPUT     optional: an element shown only once you are signed in, such
                              as the chat box. Unset: the page counts as ready once it is
                              back on APP_BROWSER_URL's site after any sign-in
        APP_BROWSER_CHANNEL, APP_BROWSER_PROFILE, APP_BROWSER_HEADLESS and
        APP_BROWSER_SIGN_IN_TIMEOUT work as for APP_TRANSPORT=browser.

    The browser's own rules apply to the request: an APP_ENDPOINT on another site works
    only if that site allows it (CORS). If the sign-in lapses mid-run (the app redirects
    or answers 401), the page is reloaded once so the browser can sign in again.
    """

    def __init__(
        self,
        *,
        page_url: str,
        endpoint: str,
        ready_selector: str = "",
        channel: str = "",
        profile: Path | None = None,
        headless: bool = False,
        sign_in_timeout: float = 300.0,
        ready_timeout: float = READY_SECONDS,
        **app_settings,
    ):
        parsed = urlparse(page_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SystemExit(f"APP_BROWSER_URL={page_url!r} is not an http(s) URL.")
        super().__init__(endpoint=urljoin(page_url, endpoint), **app_settings)
        api = urlparse(self._endpoint)
        self.set_model_name(model_name=f"app:{api.netloc}{api.path} (browser-api)")
        self._page_url = page_url
        self._origin = (parsed.scheme, parsed.netloc)
        self._ready_selector = ready_selector
        self._channel = channel
        self._profile = Path(profile) if profile else None
        self._headless = headless
        self._sign_in_timeout = sign_in_timeout
        self._ready_timeout = ready_timeout
        self._playwright = None
        self._context = None
        self._page = None
        self._lock = asyncio.Lock()
        # Counts page loads, so requests that hit the same lapsed sign-in reload it once between them.
        self._loads = 0

    @classmethod
    def from_env(cls) -> "BrowserApiTarget":
        missing = [n for n in ("APP_BROWSER_URL", "APP_ENDPOINT") if not os.getenv(n, "").strip()]
        if missing:
            raise SystemExit(
                f"APP_TRANSPORT=browser-api needs {', '.join(missing)} - see BrowserApiTarget in browser_target.py."
            )
        return cls(
            page_url=os.environ["APP_BROWSER_URL"].strip(),
            ready_selector=os.getenv("APP_BROWSER_INPUT", "").strip(),
            channel=os.getenv("APP_BROWSER_CHANNEL", "").strip(),
            profile=os.getenv("APP_BROWSER_PROFILE", "").strip() or None,
            headless=os.getenv("APP_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes"),
            sign_in_timeout=float(os.getenv("APP_BROWSER_SIGN_IN_TIMEOUT") or 300),
            **AppChatTarget.settings_from_env(),
        )

    async def _load(self) -> None:
        """Open (or reload) the app's page and wait until you are signed in. Holds self._lock."""
        if self._context is None:
            self._playwright, self._context = await launch_browser(self._channel, self._profile, self._headless)
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        await self._page.goto(self._page_url)
        if self._ready_selector:
            await wait_for_element(self._page, self._ready_selector, self._ready_timeout, self._sign_in_timeout)
        else:
            await self._wait_for_site()
        self._loads += 1

    async def _wait_for_site(self) -> None:
        """Wait until the page is back on the app's own site, after any sign-in page."""
        def on_site() -> bool:
            url = urlparse(self._page.url)
            return (url.scheme, url.netloc) == self._origin

        deadline = time.monotonic() + self._ready_timeout
        noted = False
        while not on_site():
            if time.monotonic() > deadline:
                if noted:
                    raise RuntimeError(
                        f"The page did not come back to {self._page_url} after {self._sign_in_timeout:.0f}s "
                        f"(it is at {self._page.url}). If it needs a sign-in, run once without "
                        "APP_BROWSER_HEADLESS so you can sign in."
                    )
                print(SIGN_IN_NOTE.format(timeout=self._sign_in_timeout))
                noted, deadline = True, time.monotonic() + self._sign_in_timeout
            await asyncio.sleep(POLL_SECONDS)
        await self._page.wait_for_load_state()

    async def _fetch(self, body: str) -> dict:
        return await self._page.evaluate(
            FETCH_SCRIPT,
            {"url": self._endpoint, "body": body, "headers": self._headers, "timeoutMs": self._timeout * 1000},
        )

    async def _post(self, body: str) -> tuple[int, str]:
        async with self._lock:
            if self._page is None:
                await self._load()
        loads = self._loads
        result = await self._fetch(body)
        if result["problem"] in ("redirect", "network") or result["status"] == 401:
            # Most likely the sign-in lapsed. Reload once - unless another request
            # already did since this one was sent - and try again.
            async with self._lock:
                if self._loads == loads:
                    print("\nThe app turned a request away; reloading its page in case the sign-in lapsed.")
                    await self._load()
            result = await self._fetch(body)
        if result["problem"] == "timeout":
            raise RuntimeError(f"No reply from {self._endpoint} within {self._timeout:.0f}s (APP_TIMEOUT).")
        if result["problem"] == "redirect":
            raise RuntimeError(
                f"{self._endpoint} redirected the request, even after the page was reloaded. The sign-in may "
                "need you: run without APP_BROWSER_HEADLESS. Or APP_ENDPOINT is not the API's address."
            )
        if result["problem"]:
            raise RuntimeError(
                f"The browser could not send the request to {self._endpoint}: {result['body']}. An APP_ENDPOINT "
                "on another site than APP_BROWSER_URL must allow requests from it (CORS)."
            )
        return result["status"], result["body"]

    async def cleanup_target_async(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

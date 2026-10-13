"""A deployed chat app driven through its web page, for apps whose API the kit cannot call.

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
    APP_BROWSER_MAX_TABS  conversations open at once. Default 4
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
from urllib.parse import urlparse

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration, limit_requests_per_minute

DEFAULT_PROFILE = Path.home() / ".rt_browser_profile"
POLL_SECONDS = 0.25
# How long to wait for the chat box before saying the page may need a sign-in.
READY_SECONDS = 15.0


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
        if self._context is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise SystemExit("APP_TRANSPORT=browser needs Playwright: pip install playwright") from e
        self._playwright = await async_playwright().start()
        # Unset means Chrome, or Edge where Chrome is not installed (Edge ships with Windows).
        channels = [self._channel] if self._channel else ["chrome", "msedge"]
        for i, channel in enumerate(channels):
            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self._profile or DEFAULT_PROFILE / channel),
                    channel=None if channel == "chromium" else channel,
                    headless=self._headless,
                )
                return
            except Exception as e:
                if i == len(channels) - 1:
                    await self._playwright.stop()
                    self._playwright = None
                    raise RuntimeError(
                        f"Could not start {' or '.join(channels)}: {e}. Set APP_BROWSER_CHANNEL to an "
                        "installed browser (chrome or msedge)."
                    ) from e

    async def _tab_for(self, conversation_id: str) -> tuple:
        """The tab holding this conversation, opening a fresh chat for a new one."""
        async with self._lock:
            await self._start()
            if conversation_id in self._tabs:
                self._tabs.move_to_end(conversation_id)
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
            # Close the oldest only once the new tab is open: closing a browser's last
            # tab can close the browser.
            while len(self._tabs) >= self._max_tabs:
                old_id, (old_tab, _) = self._tabs.popitem(last=False)
                self._closed.add(old_id)
                await old_tab.close()
            await self._open_chat(tab)
            self._tabs[conversation_id] = (tab, asyncio.Lock())
            return self._tabs[conversation_id]

    async def _open_chat(self, tab) -> None:
        await tab.goto(self._url)
        await self._wait_for_chat_box(tab)
        if self._new_chat:
            await tab.locator(self._new_chat).first.click()
            await self._wait_for_chat_box(tab)

    async def _wait_for_chat_box(self, tab) -> None:
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        box = tab.locator(self._input).first
        try:
            await box.wait_for(state="visible", timeout=self._ready_timeout * 1000)
            return
        except PlaywrightTimeout:
            pass
        print(
            "\nWaiting for the app's chat box. If the browser window shows a sign-in page, sign in there:\n"
            f"the run carries on by itself once the chat page is back (up to {self._sign_in_timeout:.0f}s)."
        )
        try:
            await box.wait_for(state="visible", timeout=self._sign_in_timeout * 1000)
        except PlaywrightTimeout as e:
            raise RuntimeError(
                f"No element matching APP_BROWSER_INPUT={self._input!r} appeared at {tab.url}. If the page "
                "needs a sign-in, run once without APP_BROWSER_HEADLESS so you can sign in; otherwise the "
                "selector does not match the page."
            ) from e

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
        async with tab_lock:
            text = await self.ask(tab, request.converted_value)
        return [construct_response_from_request(request=request, response_text_pieces=[text])]

    async def cleanup_target_async(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._tabs.clear()

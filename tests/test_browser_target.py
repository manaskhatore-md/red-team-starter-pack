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


needs_browser = pytest.mark.skipif(not _browser_available(), reason="needs Playwright and Chrome or Edge")


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
                        lambda profile, judge, *rest: {k: NeverFlags(category=k) for k in profile.rubrics})
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


def test_the_browser_target_says_it_keeps_history_but_takes_no_system_prompt():
    from pyrit.prompt_target import CapabilityName

    target = BrowserChatTarget(url="https://example.test/chat", input_selector="textarea", reply_selector=".reply")
    assert target.configuration.includes(capability=CapabilityName.MULTI_TURN)
    assert not target.configuration.includes(capability=CapabilityName.EDITABLE_HISTORY)
    assert not target.configuration.includes(capability=CapabilityName.SYSTEM_PROMPT)

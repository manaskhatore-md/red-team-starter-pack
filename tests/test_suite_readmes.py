"""Each attack suite has a README, the main README links to it, and its links still lead somewhere."""

import re

import pytest

from harnesses.pyrit_campaigns.profiles import REPO_ROOT

SUITE_DIRS = sorted(p.parent for p in (REPO_ROOT / "attack_suites").glob("*/*.yaml") if p.name != "checklist.yaml")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def anchors(markdown: str) -> set[str]:
    """The anchors GitHub gives a file's headings, e.g. "## Repeats: how often" -> "repeats-how-often"."""
    found = set()
    for heading in re.findall(r"^#{1,6} +(.+?) *$", markdown, re.MULTILINE):
        found.add(re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-"))
    return found


def test_every_suite_folder_has_a_readme_the_main_readme_links_to():
    main = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert SUITE_DIRS
    for folder in dict.fromkeys(SUITE_DIRS):
        assert (folder / "README.md").is_file(), f"{folder.name} has no README.md"
        assert f"](attack_suites/{folder.name}/README.md)" in main, f"README.md does not link to {folder.name}/README.md"


@pytest.mark.parametrize("readme", sorted((REPO_ROOT / "attack_suites").glob("*/README.md")), ids=lambda p: p.parent.name)
def test_a_suite_readmes_links_lead_to_a_file_and_heading_that_exist(readme):
    for target in LINK.findall(readme.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        linked = (readme.parent / path).resolve() if path else readme
        assert linked.is_file(), f"{readme.parent.name}/README.md links to {target}, which does not exist"
        if anchor:
            assert anchor in anchors(linked.read_text(encoding="utf-8")), (
                f"{readme.parent.name}/README.md links to {target}, but {linked.name} has no such heading"
            )

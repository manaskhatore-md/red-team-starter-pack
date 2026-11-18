"""Each attack suite has a README, the main README links to it, and every document's links still lead somewhere."""

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


# Every document a user reads: the main README, CONTRIBUTING.md, and the READMEs beside the
# suites and the example pack.
DOCS = sorted({
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    *(REPO_ROOT / "attack_suites").glob("*/README.md"),
    *(REPO_ROOT / "overlays" / "example").glob("README.md"),
    *(REPO_ROOT / "data_fixtures").glob("*/README.md"),
})


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_documents_links_lead_to_a_file_and_heading_that_exist(doc):
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        linked = (doc.parent / path).resolve() if path else doc
        name = doc.relative_to(REPO_ROOT)
        assert linked.exists(), f"{name} links to {target}, which does not exist"
        if anchor:
            assert anchor in anchors(linked.read_text(encoding="utf-8")), (
                f"{name} links to {target}, but {linked.name} has no such heading"
            )

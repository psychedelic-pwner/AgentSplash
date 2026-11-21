"""AI-origin extraction from JSONL transcripts and git diffs. No network."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agentsplash.origin import from_git_diff, from_transcripts

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"


def test_transcripts_keep_assistant_packages_and_skip_junk() -> None:
    found = from_transcripts(FIXTURES)
    assert found == {
        "evilpkg",
        "httpx",
        "goodlib",
        "pathlib",
        "leftpad",
        "debug",
        "express",
        "nestedpkg",
        "blockpkg",
    }


def test_transcripts_missing_root_is_empty(tmp_path: Path) -> None:
    assert from_transcripts(tmp_path / "nope") == set()


def test_git_diff_reads_added_lines_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    base = repo / "notes.txt"
    base.write_text("pip install oldpkg\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--no-gpg-sign", "-m", "init")
    base.write_text("pip install newpkg\n")
    (repo / "mod.py").write_text("import addedpkg\nrequire('fromgit')\n")
    _git(repo, "add", "mod.py")
    found = from_git_diff(repo, "HEAD")
    assert found == {"newpkg", "addedpkg", "fromgit"}


def test_git_diff_failure_warns_and_returns_empty(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="git diff failed"):
        assert from_git_diff(tmp_path, "HEAD") == set()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

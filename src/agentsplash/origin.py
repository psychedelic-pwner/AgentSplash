"""Decide which package names an AI suggested.

Transcripts are Claude Code session logs: JSONL objects whose assistant text
mentions a package in an install or import. Git evidence is the added lines of
`git diff <ref>`. Malformed lines and git failures never raise.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import warnings
from pathlib import Path
from typing import Any

# Words that end an install command when it is written as prose.
_STOPWORDS = frozenset({
    "a",
    "an",
    "after",
    "also",
    "and",
    "before",
    "can",
    "do",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "not",
    "of",
    "on",
    "or",
    "please",
    "run",
    "should",
    "so",
    "that",
    "the",
    "then",
    "this",
    "to",
    "using",
    "via",
    "we",
    "with",
    "would",
    "you",
    "your",
})

_PIP_VALUE_FLAGS = frozenset({
    "-r",
    "--requirement",
    "--requirements",
    "-c",
    "--constraint",
    "-e",
    "--editable",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "-t",
    "--target",
    "--prefix",
    "--root",
    "--src",
    "--install-option",
    "--global-option",
    "--config-settings",
    "--proxy",
    "--timeout",
    "--retries",
    "-b",
    "--build",
    "--no-binary",
    "--only-binary",
    "--progress-bar",
    "--implementation",
    "--abi",
    "--platform",
    "--python-version",
    "--log",
    "--root-user-action",
    "--report",
    "--dry-run",
    "--group",
    "--python",
    "--index",
    "--default-index",
    "--directory",
    "--project",
    "--script",
    "--constraints",
    "--override",
    "--link-mode",
    "--config-setting",
    "--reinstall-package",
    "--upgrade-package",
    "--keyring-provider",
    "--exclude-newer",
    "--index-strategy",
    "--prerelease",
    "--branch",
    "--tag",
    "--rev",
    "--registry",
    "--cache",
    "--userconfig",
})

_PYPI_OPS = ("===", "==", "~=", "!=", ">=", "<=", ">", "<")

_PIP_RE = re.compile(r"\bpip3?\s+install\b([^;\n`]*)", re.IGNORECASE)
_PY_PIP_RE = re.compile(
    r"\bpython(?:\d+(?:\.\d+)*)?\s+-m\s+pip\s+install\b([^;\n`]*)",
    re.IGNORECASE,
)
_UV_RE = re.compile(r"\buv\s+add\b([^;\n`]*)", re.IGNORECASE)
_NPM_RE = re.compile(r"\bnpm\s+(?:i|install)\b([^;\n`]*)", re.IGNORECASE)
_FROM_RE = re.compile(r"(?<![\w.])from\s+([A-Za-z_][\w.]*)\s+import\b")
_IMPORT_RE = re.compile(
    r"(?<![\w.])import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)"
)
_REQUIRE_RE = re.compile(r"""(?<![\w.])require\(\s*(['"])([^'"]+)\1\s*\)""")
_PY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PKG_NAME_RE = re.compile(r"^(?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9_][A-Za-z0-9._-]*$")


def from_transcripts(root: Path) -> set[str]:
    """Return package names suggested in assistant messages under `root`.

    `root` may be a `.jsonl` file or a directory. Directories are walked for
    `*.jsonl` (Claude Code logs live at `~/.claude/projects/**/*.jsonl`; a
    `--transcripts` directory is the same kind of tree). A line that is not a
    JSON object is skipped. Only assistant text is read.
    """
    root = Path(root)
    names: set[str] = set()
    for path in _jsonl_files(root):
        names |= _packages_in_jsonl(path)
    return names


def from_git_diff(repo: Path, ref: str) -> set[str]:
    """Return package names added since `ref` in `repo`.

    Uses `git diff <ref>` and keeps added lines only. A missing git binary, a
    non-repository, or any other git failure returns an empty set and emits a
    warning instead of raising.
    """
    repo = Path(repo)
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--no-color",
                "--unified=0",
                ref,
                "--",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"git diff failed: {exc}", RuntimeWarning, stacklevel=2)
        return set()
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = detail[0] if detail else f"exit {proc.returncode}"
        warnings.warn(f"git diff failed: {message}", RuntimeWarning, stacklevel=2)
        return set()
    added: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        added.append(line[1:])
    return _extract_packages("\n".join(added))


def _jsonl_files(root: Path) -> list[Path]:
    try:
        if root.is_file():
            return [root] if root.suffix == ".jsonl" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    except OSError:
        return []


def _packages_in_jsonl(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        for chunk in _assistant_texts(obj):
            names |= _extract_packages(chunk)
    return names


def _assistant_texts(obj: Any) -> list[str]:
    """Pull text out of the assistant-message shapes Claude Code writes."""
    if not isinstance(obj, dict) or not _is_assistant(obj):
        return []
    texts: list[str] = []
    _collect_text(obj.get("content"), texts)
    _collect_text(obj.get("text"), texts)
    message = obj.get("message")
    if isinstance(message, dict):
        _collect_text(message.get("content"), texts)
        _collect_text(message.get("text"), texts)
    return texts


def _is_assistant(obj: dict[str, Any]) -> bool:
    if obj.get("role") == "assistant" or obj.get("type") == "assistant":
        return True
    message = obj.get("message")
    return isinstance(message, dict) and message.get("role") == "assistant"


def _collect_text(content: Any, out: list[str]) -> None:
    if isinstance(content, str):
        out.append(content)
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    out.append(text)
        return
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            out.append(text)


def _extract_packages(text: str) -> set[str]:
    names: set[str] = set()
    for match in _FROM_RE.finditer(text):
        _add_python_module(names, match.group(1))
    without_from = _FROM_RE.sub(" ", text)
    for match in _IMPORT_RE.finditer(without_from):
        for part in match.group(1).split(","):
            _add_python_module(names, part.strip())
    for match in _REQUIRE_RE.finditer(text):
        cleaned = _clean_package(match.group(2))
        if cleaned is not None:
            names.add(cleaned)
    for pattern in (_PIP_RE, _PY_PIP_RE, _UV_RE, _NPM_RE):
        for match in pattern.finditer(text):
            names |= _packages_from_install_tail(match.group(1))
    return names


def _add_python_module(names: set[str], token: str) -> None:
    head = token.split(".", 1)[0].strip()
    if _PY_NAME_RE.fullmatch(head):
        names.add(head)


def _packages_from_install_tail(tail: str) -> set[str]:
    try:
        tokens = _shlex_split(tail)
    except ValueError:
        tokens = tail.split()
    names: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            continue
        if token.startswith("-") and token != "-":
            flag, _, inline = token.partition("=")
            if inline:
                index += 1
                continue
            index += 2 if flag in _PIP_VALUE_FLAGS else 1
            continue
        bare = token.strip().strip("'\"`()[]{}<>,").rstrip(".,;:!?").lower()
        if bare in _STOPWORDS:
            break
        cleaned = _clean_package(token)
        if cleaned is not None:
            names.add(cleaned)
        index += 1
    return names


def _shlex_split(tail: str) -> list[str]:
    return shlex.split(tail, posix=True)


def _clean_package(token: str) -> str | None:
    text = token.strip().strip("'\"`()[]{}<>,").rstrip(".,;:!?")
    if not text or text[0] in "-./" or "://" in text or text.startswith("git+"):
        return None
    bracket = text.find("[")
    if bracket > 0:
        text = text[:bracket].strip()
    for op in _PYPI_OPS:
        idx = text.find(op)
        if idx > 0:
            text = text[:idx].strip()
            break
    else:
        text = _strip_npm_version(text).strip()
    if not text or not _PKG_NAME_RE.fullmatch(text):
        return None
    return text


def _strip_npm_version(text: str) -> str:
    if text.startswith("@"):
        slash = text.find("/")
        if slash == -1:
            return text
        at = text.find("@", slash + 1)
        return text[:at] if at != -1 else text
    at = text.find("@")
    if at > 0:
        return text[:at]
    return text

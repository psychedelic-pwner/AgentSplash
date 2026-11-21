"""Find MCP client configs and map launch lines to registry packages.

No package is executed. Configs that cannot be parsed raise `McpConfigError`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SKIP_DIRS = frozenset({"node_modules", ".git", ".venv"})
_NAMED_CONFIGS = frozenset({
    "claude_desktop_config.json",
    ".mcp.json",
    "mcp.json",
})

_PYTHON_RE = re.compile(r"python(\d+(\.\d+)*)?")
_PYPI_OPS = ("===", "==", "~=", "!=", ">=", "<=", ">", "<")

# Flags whose following argument is the package, or `--flag=package`.
_NPM_PACKAGE_FLAGS = frozenset({"-p", "--package"})
_UV_PACKAGE_FLAGS = frozenset({"--from", "--spec"})
_PIPX_PACKAGE_FLAGS = frozenset({"--spec"})

# Flags that consume the next token and are not the package.
_NPM_VALUE_FLAGS = frozenset({
    "--prefix",
    "--cache",
    "--userconfig",
    "--globalconfig",
    "--registry",
    "--scope",
    "--user",
    "--otp",
    "--loglevel",
    "--logs-max",
    "--script-shell",
    "--node-options",
    "--npm",
    "--shell",
    "--call",
    "-c",
    "--workspace",
    "-w",
    "--include",
    "--omit",
})
_UV_VALUE_FLAGS = frozenset({
    "--python",
    "-p",
    "--with",
    "--with-editable",
    "--with-requirements",
    "-r",
    "--index-url",
    "--extra-index-url",
    "-i",
    "--find-links",
    "-f",
    "--directory",
    "--project",
    "--config-file",
    "--refresh-package",
    "--exclude-newer",
    "--python-preference",
    "--index-strategy",
    "--keyring-provider",
    "--prerelease",
    "--link-mode",
    "--constraint",
    "-c",
    "--override",
    "--config-setting",
    "--config-settings",
    "--extra",
    "--reinstall-package",
    "--upgrade-package",
})
_PIPX_VALUE_FLAGS = frozenset({
    "--python",
    "--pip-args",
    "--index-url",
    "-i",
    "--pypackages",
})
_PYTHON_VALUE_FLAGS = frozenset({"-X", "-W", "-c", "--check-hash-based-pycs"})


class McpConfigError(Exception):
    """An MCP config is missing, unreadable, or not valid JSON/JSONC."""


@dataclass(frozen=True, slots=True)
class McpServer:
    """One server entry from an MCP client config.

    `ecosystem` and `package` are set when `map_package` recognizes the launch
    line. `url` is set for remote servers (`"url"` in the config).
    """

    config_path: Path
    server_name: str
    command: str | None
    args: list[str]
    env: dict[str, str]
    ecosystem: str | None
    package: str | None
    url: str | None = None


def discover_configs(root: Path) -> list[Path]:
    """Return MCP config files under `root`, sorted by resolved path.

    Named files (`claude_desktop_config.json`, `.mcp.json`, `mcp.json`, which
    covers `.cursor/mcp.json` and `.vscode/mcp.json`) are always included.
    Any other `*.json` file is included when its top-level object has an
    `mcpServers` key. `node_modules`, `.git`, and `.venv` are not walked.
    Paths that resolve to the same file are returned once.
    """
    root = Path(root)
    if not root.exists():
        raise McpConfigError(f"{root} does not exist")

    found: dict[Path, None] = {}
    for path in _iter_files(root):
        if not path.is_file():
            continue
        name = path.name
        named = name in _NAMED_CONFIGS
        if named or (name.endswith(".json") and _has_top_level_mcp_servers(path)):
            found.setdefault(path.resolve(), None)
    return sorted(found)


def parse_config(path: Path) -> list[McpServer]:
    """Parse one JSON or JSONC MCP config into server entries.

    Raises `McpConfigError` when the file cannot be read or is not a config
    object. A server entry with a bad shape raises; it is not dropped.
    """
    path = Path(path)
    resolved = path.resolve()
    data = _load_json(resolved)
    if not isinstance(data, dict):
        raise McpConfigError(f"invalid JSON in {resolved}: top-level value must be an object")

    servers: list[McpServer] = []
    for name, spec in _server_table(data, resolved).items():
        servers.append(_server_from_spec(resolved, str(name), spec))
    return servers


def map_package(command: str, args: Sequence[str] | None = None) -> tuple[str, str] | None:
    """Infer `(ecosystem, package)` from a server launch line.

    Recognized forms:

    - `npx` / `npm exec` -> `("npm", package)`
    - `uvx` / `uv tool run` / `pipx run` -> `("pypi", package)`
    - `python -m X` (any `python3.x`) -> `("pypi", X)`
    - `node <path>` -> `None`

    Flags and `--yes`-style options are skipped. Version pins (`pkg@1.2.3`,
    `pkg==1.2.3`) are removed. Scoped npm names keep the scope.
    """
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        cmd_tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not cmd_tokens:
        return None

    exe = _basename(cmd_tokens[0])
    tail = cmd_tokens[1:] + _as_args(args)
    if exe in {"node", "nodejs"}:
        return None
    if _PYTHON_RE.fullmatch(exe):
        module = _python_module(tail)
        if module is None:
            return None
        return ("pypi", module)
    if exe == "npx":
        return _mapped(
            "npm",
            tail,
            skip=(),
            package_flags=_NPM_PACKAGE_FLAGS,
            value_flags=_NPM_VALUE_FLAGS,
        )
    if exe == "npm":
        return _mapped(
            "npm",
            tail,
            skip=("exec",),
            package_flags=_NPM_PACKAGE_FLAGS,
            value_flags=_NPM_VALUE_FLAGS,
        )
    if exe == "uvx":
        return _mapped(
            "pypi",
            tail,
            skip=(),
            package_flags=_UV_PACKAGE_FLAGS,
            value_flags=_UV_VALUE_FLAGS,
        )
    if exe == "uv":
        return _mapped(
            "pypi",
            tail,
            skip=("tool", "run"),
            package_flags=_UV_PACKAGE_FLAGS,
            value_flags=_UV_VALUE_FLAGS,
        )
    if exe == "pipx":
        return _mapped(
            "pypi",
            tail,
            skip=("run",),
            package_flags=_PIPX_PACKAGE_FLAGS,
            value_flags=_PIPX_VALUE_FLAGS,
        )
    return None


def scan_configs(root: Path) -> list[McpServer]:
    """Discover configs under `root` and parse each one.

    A config that fails to parse raises `McpConfigError`. Broken files are
    not skipped.
    """
    servers: list[McpServer] = []
    for path in discover_configs(root):
        servers.extend(parse_config(path))
    return servers


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def _has_top_level_mcp_servers(path: Path) -> bool:
    try:
        data = _load_json(path)
    except McpConfigError:
        return False
    return isinstance(data, dict) and "mcpServers" in data


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise McpConfigError(f"cannot read {path}: not valid UTF-8") from exc
    except OSError as exc:
        raise McpConfigError(f"cannot read {path}: {exc}") from exc
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    try:
        cleaned = _strip_jsonc(text)
    except ValueError as exc:
        raise McpConfigError(f"{exc} in {path}") from exc
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise McpConfigError(
            f"invalid JSON in {path}: {exc.msg} (line {exc.lineno})"
        ) from exc


def _strip_jsonc(text: str) -> str:
    """Drop `//` and `/* */` comments outside strings, then trailing commas."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            closed = False
            while i < n:
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    i += 2
                    closed = True
                    break
                i += 1
            if not closed:
                raise ValueError("unterminated block comment")
            continue
        out.append(ch)
        i += 1
    return _strip_trailing_commas("".join(out))


def _strip_trailing_commas(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _server_table(data: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if "mcpServers" in data:
        table = data["mcpServers"]
        if table is None:
            return {}
        if not isinstance(table, dict):
            raise McpConfigError(f"{path}: mcpServers must be an object")
        return table

    candidates: list[Any] = []
    if "servers" in data:
        candidates.append(data["servers"])
    mcp = data.get("mcp")
    if isinstance(mcp, dict) and "servers" in mcp:
        candidates.append(mcp["servers"])
    for candidate in candidates:
        if _looks_like_mcp_table(candidate):
            if not isinstance(candidate, dict):
                raise McpConfigError(f"{path}: servers must be an object")
            return candidate
    return {}


def _looks_like_mcp_table(table: Any) -> bool:
    if not isinstance(table, dict) or not table:
        return False
    return any(
        isinstance(value, dict) and ({"command", "url", "type"} & value.keys())
        for value in table.values()
    )


def _server_from_spec(path: Path, name: str, spec: Any) -> McpServer:
    if not isinstance(spec, dict):
        raise McpConfigError(f"{path}: server {name!r} must be an object")
    command = _optional_str(spec.get("command"), path, name, "command")
    url = _optional_str(spec.get("url"), path, name, "url")
    args = _string_list(spec.get("args", []), path, name)
    env = _string_dict(spec.get("env", {}), path, name)
    if command is None and url is None:
        raise McpConfigError(f"{path}: server {name!r} has neither command nor url")

    ecosystem: str | None = None
    package: str | None = None
    if command is not None:
        mapped = map_package(command, args)
        if mapped is not None:
            ecosystem, package = mapped
    return McpServer(
        config_path=path,
        server_name=name,
        command=command,
        args=args,
        env=env,
        ecosystem=ecosystem,
        package=package,
        url=url,
    )


def _optional_str(value: Any, path: Path, server: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise McpConfigError(f"{path}: server {server!r} {field} must be a string")
    stripped = value.strip()
    return stripped or None


def _string_list(value: Any, path: Path, server: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise McpConfigError(f"{path}: server {server!r} args must be a list of strings")
    return list(value)


def _string_dict(value: Any, path: Path, server: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise McpConfigError(f"{path}: server {server!r} env must be an object of strings")
    return dict(value)


def _basename(command: str) -> str:
    name = Path(command).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _as_args(args: Sequence[str] | None) -> list[str]:
    if args is None:
        return []
    if isinstance(args, str):
        try:
            return shlex.split(args, posix=True)
        except ValueError:
            return []
    return list(args)


def _mapped(
    ecosystem: str,
    args: Sequence[str],
    *,
    skip: tuple[str, ...],
    package_flags: frozenset[str],
    value_flags: frozenset[str],
) -> tuple[str, str] | None:
    package = _package_from_args(
        args,
        skip=skip,
        package_flags=package_flags,
        value_flags=value_flags,
    )
    if package is None:
        return None
    return ecosystem, package


def _package_from_args(
    args: Sequence[str],
    *,
    skip: tuple[str, ...],
    package_flags: frozenset[str],
    value_flags: frozenset[str],
) -> str | None:
    i = 0
    stage = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if tok == "--":
            if stage < len(skip):
                return None
            if i + 1 < n:
                return _strip_pin(args[i + 1])
            return None
        if tok.startswith("-") and tok != "-":
            name, has_value, inline = _split_opt(tok)
            if name in package_flags:
                if has_value:
                    return _strip_pin(inline)
                if i + 1 < n:
                    return _strip_pin(args[i + 1])
                return None
            i += 1 if has_value or name not in value_flags else 2
            continue
        if stage < len(skip):
            if tok != skip[stage]:
                return None
            stage += 1
            i += 1
            continue
        return _strip_pin(tok)
    return None


def _python_module(args: Sequence[str]) -> str | None:
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if tok == "--":
            return None
        if tok == "-m":
            if i + 1 >= n:
                return None
            return _strip_pin(args[i + 1])
        if tok.startswith("-"):
            name, has_value, _inline = _split_opt(tok)
            i += 1 if has_value or name not in _PYTHON_VALUE_FLAGS else 2
            continue
        return None
    return None


def _split_opt(tok: str) -> tuple[str, bool, str]:
    if "=" in tok:
        name, value = tok.split("=", 1)
        return name, True, value
    return tok, False, ""


def _strip_pin(spec: str) -> str | None:
    text = spec.strip().strip("'\"")
    if not text or text.startswith("-"):
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
        text = _strip_npm_version(text)
    if not text or text.startswith(("-", ".")):
        return None
    return text


def _strip_npm_version(text: str) -> str:
    if text.startswith("@"):
        slash = text.find("/")
        if slash == -1:
            return text
        at = text.find("@", slash + 1)
        if at != -1:
            return text[:at]
        return text
    at = text.find("@")
    if at > 0:
        return text[:at]
    return text

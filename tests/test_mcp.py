"""MCP config discovery, JSONC parsing, and package mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentsplash.mcp import (
    McpConfigError,
    McpServer,
    discover_configs,
    map_package,
    parse_config,
    scan_configs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mcp"


def test_discover_named_files_jsonc_and_nested_cursor() -> None:
    found = discover_configs(FIXTURES)
    relative = [path.relative_to(FIXTURES.resolve()).as_posix() for path in found]
    assert relative == [
        "claude_desktop_config.json",
        "commented.json",
        "nested/.cursor/mcp.json",
        "nested/.vscode/mcp.json",
        "npx.json",
        "uvx.json",
    ]


def test_discover_skips_vendor_dirs_and_dedupes_resolved_paths(tmp_path: Path) -> None:
    real = tmp_path / "mcp.json"
    real.write_text(json.dumps({"mcpServers": {"one": {"command": "uvx", "args": ["demo"]}}}))

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    (alias_dir / "linked.json").symlink_to(real)

    for skipped in ("node_modules", ".git", ".venv"):
        directory = tmp_path / skipped
        directory.mkdir()
        (directory / "mcp.json").write_text(
            json.dumps({"mcpServers": {"hidden": {"command": "npx", "args": ["secret"]}}})
        )
        (directory / "claude_desktop_config.json").write_text("{}")

    (tmp_path / "noise.json").write_text('{"label": "no mcpServers key"}')

    found = discover_configs(tmp_path)
    assert found == [real.resolve()]


def test_discover_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(McpConfigError, match="does not exist"):
        discover_configs(missing)


def test_jsonc_comments_do_not_eat_strings() -> None:
    servers = parse_config(FIXTURES / "commented.json")
    assert servers == [
        McpServer(
            config_path=(FIXTURES / "commented.json").resolve(),
            server_name="notes",
            command="pipx",
            args=["run", "mcp-notes"],
            env={
                "ENDPOINT": "http://example.com//keep",
                "BANNER": "/* keep */ // keep",
            },
            ecosystem="pypi",
            package="mcp-notes",
            url=None,
        )
    ]


def test_scan_fixtures_maps_every_launch_line() -> None:
    scanned = [
        (server.server_name, server.ecosystem, server.package, server.command, server.args)
        for server in scan_configs(FIXTURES)
    ]
    assert scanned == [
        ("docs", "pypi", "mcp_server_docs", "python", ["-m", "mcp_server_docs", "--port", "0"]),
        ("notes", "pypi", "mcp-notes", "pipx", ["run", "mcp-notes"]),
        ("cursor-shell", "pypi", "mcp-shell", "uv", ["tool", "run", "mcp-shell"]),
        ("vscode-node", None, None, "node", ["./server.js"]),
        (
            "fs",
            "npm",
            "@modelcontextprotocol/server-filesystem",
            "npx",
            ["-y", "@modelcontextprotocol/server-filesystem@1.2.3", "/tmp"],
        ),
        ("git", "pypi", "mcp-server-git", "uvx", ["mcp-server-git", "--repository", "/tmp/repo"]),
    ]
    docs = scan_configs(FIXTURES)[0]
    assert docs.env == {"DOCS_ROOT": "/tmp/docs"}


@pytest.mark.parametrize(
    ("command", "args", "expected"),
    [
        ("npx", ["-y", "left-pad"], ("npm", "left-pad")),
        ("npx", ["--yes", "left-pad"], ("npm", "left-pad")),
        ("npx", ["pkg@1.2.3"], ("npm", "pkg")),
        ("npx", ["-y", "@scope/pkg@1.2.3", "/tmp"], ("npm", "@scope/pkg")),
        ("npx", ["--package", "@scope/pkg@2.0.0", "binary"], ("npm", "@scope/pkg")),
        ("npx", ["--package=@scope/pkg@2.0.0", "binary"], ("npm", "@scope/pkg")),
        ("/usr/local/bin/npx", ["--yes", "pkg"], ("npm", "pkg")),
        ("npx -y pkg@1.2.3", [], ("npm", "pkg")),
        ("npm", ["exec", "pkg"], ("npm", "pkg")),
        ("npm", ["exec", "--yes", "@org/name@2.0.0"], ("npm", "@org/name")),
        ("npm", ["--loglevel=error", "exec", "--", "pkg@9.9.9"], ("npm", "pkg")),
        ("npm", ["install", "left-pad"], None),
        ("uvx", ["mcp-server-git"], ("pypi", "mcp-server-git")),
        ("uvx", ["pkg==1.2.3"], ("pypi", "pkg")),
        ("uvx", ["--from", "pkg==1.2.3", "toolname"], ("pypi", "pkg")),
        ("uv", ["tool", "run", "demo"], ("pypi", "demo")),
        ("uv", ["tool", "run", "demo[extra]==0.1.0"], ("pypi", "demo")),
        ("uv tool run demo==0.1.0", None, ("pypi", "demo")),
        ("uv", ["run", "demo"], None),
        ("pipx", ["run", "mcp-notes"], ("pypi", "mcp-notes")),
        ("pipx", ["run", "--spec", "pkg==1.4.0", "toolname"], ("pypi", "pkg")),
        ("pipx", ["install", "pkg"], None),
        ("python", ["-m", "mcp_server_docs"], ("pypi", "mcp_server_docs")),
        ("python3", ["-m", "pkg.module"], ("pypi", "pkg.module")),
        ("python3.12", ["-X", "dev", "-m", "sample_pkg"], ("pypi", "sample_pkg")),
        ("/usr/bin/python3", ["-m", "sample_pkg"], ("pypi", "sample_pkg")),
        ("python", ["script.py"], None),
        ("node", ["./server.js"], None),
        ("node", ["/tmp/server.js"], None),
        ("nodejs", ["some-pkg"], None),
        ("/usr/bin/node", ["./x.js"], None),
        ("echo", ["npx", "pkg"], None),
        ("npx", ["-y"], None),
    ],
)
def test_map_package_command_forms(
    command: str,
    args: list[str] | None,
    expected: tuple[str, str] | None,
) -> None:
    assert map_package(command, args) == expected


def test_url_is_kept_and_does_not_invent_a_package(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://example.com/mcp",
                        "env": {"TOKEN": "secret"},
                    },
                    "both": {
                        "command": "npx",
                        "args": ["pkg@1.0.0"],
                        "url": "https://example.com/also",
                    },
                }
            }
        )
    )
    remote, both = parse_config(path)
    assert remote.command is None
    assert remote.url == "https://example.com/mcp"
    assert remote.ecosystem is None
    assert remote.package is None
    assert remote.env == {"TOKEN": "secret"}
    assert both.url == "https://example.com/also"
    assert both.ecosystem == "npm"
    assert both.package == "pkg"


def test_broken_config_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    good = tmp_path / ".mcp.json"
    good.write_text(
        json.dumps({"mcpServers": {"ok": {"command": "uvx", "args": ["good-pkg"]}}})
    )
    bad = tmp_path / "claude_desktop_config.json"
    bad.write_text("{not json, // still broken\n")

    with pytest.raises(McpConfigError, match="invalid JSON") as exc:
        parse_config(bad)
    assert bad.name in str(exc.value)

    with pytest.raises(McpConfigError, match="invalid JSON") as scanned:
        scan_configs(tmp_path)
    assert bad.name in str(scanned.value)

    malformed = tmp_path / "mcp.json"
    malformed.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ok": {"command": "uvx", "args": ["good-pkg"]},
                    "bad": ["nope"],
                }
            }
        )
    )
    with pytest.raises(McpConfigError, match="server 'bad' must be an object"):
        parse_config(malformed)

    bad_args = tmp_path / "nested"
    bad_args.mkdir()
    args_path = bad_args / "mcp.json"
    args_path.write_text(json.dumps({"mcpServers": {"x": {"command": "uvx", "args": "pkg"}}}))
    with pytest.raises(McpConfigError, match="args must be a list of strings"):
        parse_config(args_path)


def test_unterminated_block_comment_raises(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": { /* never closed')
    with pytest.raises(McpConfigError, match="unterminated block comment"):
        parse_config(path)

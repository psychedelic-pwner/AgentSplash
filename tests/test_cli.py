"""Scan CLI. Offline only — no registry traffic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentsplash.cli import main, similar_names
from agentsplash.report import Finding, render_json, render_table

TRANSCRIPTS = Path(__file__).parent / "fixtures" / "transcripts"

_CONFIG = {
    "mcpServers": {
        "evil": {"command": "uvx", "args": ["evilpkg"]},
        "healthy": {"command": "uvx", "args": ["requests"]},
        "typo": {"command": "uvx", "args": ["reqests"]},
        "local": {"command": "node", "args": ["server.js"]},
    }
}


@pytest.fixture(autouse=True)
def _no_home_transcripts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "no-claude-projects"
    empty.mkdir()
    monkeypatch.setattr("agentsplash.cli._claude_project_logs", lambda: empty)


def test_similar_names_catch_a_one_edit_typo() -> None:
    assert similar_names("reqests", "pypi") == ["requests"]
    assert similar_names("requests", "pypi") == []


def test_render_orders_by_risk_then_name_and_keeps_json_keys() -> None:
    low = Finding("aaa", "pypi", "ok", 0, [], False, "/a")
    high = Finding("zzz", "npm", "missing", 100, ["gone"], True, "/z")
    mid = Finding("mmm", "pypi", "lookalike", 80, ["young"], False, "/m")
    table = render_table([low, high, mid])
    assert table.index("zzz") < table.index("mmm") < table.index("aaa")
    assert table.startswith("PACKAGE")
    payload = json.loads(render_json([low, high, mid]))
    assert [row["package"] for row in payload] == ["zzz", "mmm", "aaa"]
    assert list(payload[0]) == [
        "package",
        "ecosystem",
        "verdict",
        "risk",
        "reasons",
        "ai_suggested",
        "source",
    ]
    assert payload[0]["ai_suggested"] is True
    assert payload[0]["reasons"] == ["gone"]


def test_scan_offline_json_exit_and_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _write_config(tmp_path)
    code = main(
        [
            "scan",
            "--path",
            str(root),
            "--offline",
            "--json",
            "--transcripts",
            str(TRANSCRIPTS),
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    by_name = {row["package"]: row for row in payload}
    assert list(by_name) == ["evilpkg", "reqests", "requests"]
    assert set(by_name["evilpkg"]) == {
        "package",
        "ecosystem",
        "verdict",
        "risk",
        "reasons",
        "ai_suggested",
        "source",
    }
    assert by_name["evilpkg"]["ecosystem"] == "pypi"
    assert by_name["evilpkg"]["verdict"] == "missing"
    assert by_name["evilpkg"]["risk"] == 100
    assert by_name["evilpkg"]["ai_suggested"] is True
    assert any(reason.startswith("hard block:") for reason in by_name["evilpkg"]["reasons"])
    assert by_name["evilpkg"]["source"].endswith("mcp.json")
    assert by_name["reqests"]["verdict"] == "lookalike"
    assert by_name["reqests"]["risk"] == 80
    assert by_name["reqests"]["ai_suggested"] is False
    assert "requests" in by_name["reqests"]["reasons"][0]
    assert by_name["requests"]["verdict"] == "ok"
    assert by_name["requests"]["risk"] == 0
    assert by_name["requests"]["ai_suggested"] is False
    assert "node" not in captured.out


def test_scan_table_orders_risk_descending(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _write_config(tmp_path)
    code = main(["scan", "--path", str(root), "--offline"])
    assert code == 1
    out = capsys.readouterr().out
    assert out.index("evilpkg") < out.index("reqests") < out.index("requests")
    assert "PACKAGE" in out and "ECOSYSTEM" in out and "VERDICT" in out


def test_min_risk_hides_rows_but_fail_threshold_still_exits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_config(tmp_path)
    code = main(
        ["scan", "--path", str(root), "--offline", "--json", "--min-risk", "101"]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out) == []

    code = main(
        [
            "scan",
            "--path",
            str(root),
            "--offline",
            "--json",
            "--fail-on-risk",
            "101",
            "--min-risk",
            "100",
        ]
    )
    assert code == 0
    shown = json.loads(capsys.readouterr().out)
    assert [row["package"] for row in shown] == ["evilpkg"]


def test_healthy_package_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"ok": {"command": "uvx", "args": ["requests"]}}})
    )
    code = main(["scan", "--path", str(root), "--offline", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["risk"] == 0
    assert payload[0]["verdict"] == "ok"


def test_missing_path_and_bad_config_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["scan", "--path", str(tmp_path / "missing"), "--offline"])
    assert code == 2
    assert "error:" in capsys.readouterr().err

    root = tmp_path / "bad"
    root.mkdir()
    (root / "mcp.json").write_text("{")
    code = main(["scan", "--path", str(root), "--offline", "--json"])
    assert code == 2
    err = capsys.readouterr()
    assert "error:" in err.err
    assert err.out == ""


def test_bad_args_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", "--min-risk", "nope"]) == 2
    assert main([]) == 2
    capsys.readouterr()


def test_git_ref_marks_added_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "keep.txt").write_text("base\n")
    _git(repo, "add", "keep.txt")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--no-gpg-sign", "-m", "init")
    (repo / "mcp.json").write_text(
        json.dumps({"mcpServers": {"evil": {"command": "npx", "args": ["-y", "evilpkg"]}}})
    )
    (repo / "hint.py").write_text("npm i evilpkg\n")
    _git(repo, "add", "hint.py")
    code = main(["scan", "--path", str(repo), "--offline", "--json", "--git-ref", "HEAD"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["package"] == "evilpkg"
    assert payload[0]["ecosystem"] == "npm"
    assert payload[0]["ai_suggested"] is True


def _write_config(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mcp.json").write_text(json.dumps(_CONFIG))
    return root


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

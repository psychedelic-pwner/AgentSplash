"""Command line for the agentsplash scan pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from agentsplash.bench import BenchError, run_bench
from agentsplash.mcp import McpConfigError, scan_configs
from agentsplash.origin import from_git_diff, from_transcripts
from agentsplash.registry import PackageInfo, RegistryClient, RegistryError
from agentsplash.report import Finding, render_json, render_table
from agentsplash.score import score

# Common registry names. Offline lookups treat these as old, heavily
# downloaded packages. Similarity is measured against the same lists.
POPULAR: dict[str, tuple[str, ...]] = {
    "pypi": (
        "requests",
        "urllib3",
        "certifi",
        "idna",
        "pip",
        "setuptools",
        "wheel",
        "numpy",
        "pandas",
        "scipy",
        "flask",
        "django",
        "fastapi",
        "pydantic",
        "httpx",
        "aiohttp",
        "sqlalchemy",
        "pytest",
        "click",
        "boto3",
        "cryptography",
        "pillow",
        "pyyaml",
        "jinja2",
        "werkzeug",
        "celery",
        "redis",
        "attrs",
        "six",
        "tqdm",
        "beautifulsoup4",
        "lxml",
        "openai",
        "black",
        "mypy",
        "rich",
        "typer",
        "packaging",
        "typing-extensions",
        "charset-normalizer",
    ),
    "npm": (
        "lodash",
        "react",
        "react-dom",
        "vue",
        "express",
        "next",
        "axios",
        "typescript",
        "webpack",
        "eslint",
        "prettier",
        "jest",
        "mocha",
        "chalk",
        "commander",
        "debug",
        "fs-extra",
        "glob",
        "moment",
        "uuid",
        "rxjs",
        "redux",
        "zod",
        "vite",
        "jquery",
        "mongoose",
        "dotenv",
        "cors",
        "jsonwebtoken",
        "ws",
        "yargs",
        "semver",
        "minimist",
        "nodemon",
        "vitest",
        "playwright",
        "body-parser",
        "socket.io",
        "dayjs",
        "inquirer",
    ),
}

# Names present in the offline map that are not "healthy popular" packages.
# Keeps the young-lookalike score reachable without the network.
_OFFLINE_EXTRAS: dict[tuple[str, str], PackageInfo] = {
    ("pypi", "reqests"): PackageInfo(
        name="reqests",
        ecosystem="pypi",
        exists=True,
        first_publish_days_ago=3,
        downloads=12,
        latest_version="0.0.1",
    ),
}

_HEALTHY_AGE_DAYS = 2000
_HEALTHY_DOWNLOADS = 5_000_000
_SEPARATORS = re.compile(r"[-_.]+")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns 0, 1, or 2. Does not raise SystemExit for scan errors."""
    parser = _parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 2
    try:
        if args.command == "bench":
            return run_bench(args.cases, args.registry)
        return _scan(args)
    except (McpConfigError, RegistryError, BenchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentsplash")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="score packages launched from MCP configs")
    scan.add_argument("--path", type=Path, default=Path("."), help="directory to scan")
    scan.add_argument("--transcripts", type=Path, default=None, help="JSONL transcript directory")
    scan.add_argument("--git-ref", default=None, help="git ref; added lines count as AI suggestions")
    scan.add_argument("--offline", action="store_true", help="use the bundled registry fixture")
    scan.add_argument("--json", action="store_true", help="print JSON instead of a table")
    scan.add_argument("--min-risk", type=int, default=0, help="hide findings below this risk")
    scan.add_argument(
        "--fail-on-risk",
        type=int,
        default=50,
        help="exit 1 when any finding reaches this risk",
    )
    bench = sub.add_parser("bench", help="score the offline labeled corpus")
    bench.add_argument("--cases", type=Path, default=None, help="cases.json path")
    bench.add_argument("--registry", type=Path, default=None, help="registry.json path")
    return parser


def _scan(args: argparse.Namespace) -> int:
    servers = scan_configs(args.path)
    suggested = _suggested_names(args.path, args.transcripts, args.git_ref)
    client = None if args.offline else RegistryClient()
    findings: list[Finding] = []
    for server in servers:
        if not server.package or not server.ecosystem:
            continue
        info = _lookup(server.ecosystem, server.package, offline=args.offline, client=client)
        info = _with_scanned_name(info, server.package)
        ai_suggested = _was_suggested(server.package, suggested)
        similar = similar_names(server.package, server.ecosystem)
        risk, reasons = score(info, similar, ai_suggested)
        findings.append(
            Finding(
                package=server.package,
                ecosystem=server.ecosystem,
                verdict=_verdict(info, risk),
                risk=risk,
                reasons=reasons,
                ai_suggested=ai_suggested,
                source=str(server.config_path),
            )
        )
    shown = [finding for finding in findings if finding.risk >= args.min_risk]
    rendered = render_json(shown) if args.json else render_table(shown)
    sys.stdout.write(rendered)
    return 1 if any(finding.risk >= args.fail_on_risk for finding in findings) else 0


def _claude_project_logs() -> Path:
    """Default Claude Code session directory. Tests replace this."""
    return Path.home() / ".claude" / "projects"


def _suggested_names(repo: Path, transcripts: Path | None, git_ref: str | None) -> set[str]:
    names: set[str] = set()
    logs = _claude_project_logs()
    if logs.is_dir():
        names |= from_transcripts(logs)
    if transcripts is not None:
        names |= from_transcripts(transcripts)
    if git_ref:
        names |= from_git_diff(repo, git_ref)
    return names


def _lookup(
    ecosystem: str,
    name: str,
    *,
    offline: bool,
    client: RegistryClient | None,
) -> PackageInfo:
    if offline or client is None:
        return offline_package(ecosystem, name)
    return client.exists(ecosystem, name)


def offline_package(ecosystem: str, name: str) -> PackageInfo:
    """Registry stand-in shipped with the tool. Unknown names do not exist."""
    eco = ecosystem.strip().lower()
    key = _norm(name)
    extra = _OFFLINE_EXTRAS.get((eco, key))
    if extra is not None:
        return extra
    catalog = POPULAR.get(eco, ())
    for popular in catalog:
        if _norm(popular) == key:
            return PackageInfo(
                name=popular,
                ecosystem=eco,
                exists=True,
                first_publish_days_ago=_HEALTHY_AGE_DAYS,
                downloads=_HEALTHY_DOWNLOADS,
                latest_version="1.0.0",
            )
    return PackageInfo(
        name=name,
        ecosystem=eco,
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )


def similar_names(
    name: str,
    ecosystem: str,
    catalog: Sequence[str] | None = None,
) -> list[str]:
    """Popular names within a small edit distance of `name`, same ecosystem.

    The name itself (ignoring case and `-` / `_` / `.`) is not a lookalike.
    Short names use distance 1. Names of 8+ normalized characters use distance 2.
    `catalog` replaces the built-in popular list. The benchmark does not use
    this helper; it calls `popular_similar`, which is distance 2 against `POPULAR`.
    """
    eco = ecosystem.strip().lower()
    names = tuple(catalog) if catalog is not None else POPULAR.get(eco, ())
    norm = _norm(name)
    if len(norm) < 2:
        return []
    limit = 1 if len(norm) < 8 else 2
    hits: list[str] = []
    for other in names:
        other_norm = _norm(other)
        if not other_norm or other_norm == norm:
            continue
        if abs(len(other_norm) - len(norm)) > limit:
            continue
        if _damerau(norm, other_norm) <= limit:
            hits.append(other)
    return sorted(hits)


def popular_similar(name: str, ecosystem: str) -> list[str]:
    """Builtin popular names within edit distance 2 of `name`.

    Used by the benchmark. A package that merely exists, or that is close only
    to some other fixture name, is not similar. The name itself is excluded.
    Distance is 2 for every length, which is wider than `similar_names` on
    names shorter than 8 characters.
    """
    eco = ecosystem.strip().lower()
    norm = _norm(name)
    if len(norm) < 2:
        return []
    hits: list[str] = []
    for other in POPULAR.get(eco, ()):
        other_norm = _norm(other)
        if not other_norm or other_norm == norm:
            continue
        if abs(len(other_norm) - len(norm)) > 2:
            continue
        if _damerau(norm, other_norm) <= 2:
            hits.append(other)
    return sorted(hits)


def _was_suggested(package: str, suggested: set[str]) -> bool:
    target = _norm(package)
    if not target:
        return False
    return any(_norm(name) == target for name in suggested)


def _with_scanned_name(info: PackageInfo, name: str) -> PackageInfo:
    if info.name == name:
        return info
    return PackageInfo(
        name=name,
        ecosystem=info.ecosystem,
        exists=info.exists,
        first_publish_days_ago=info.first_publish_days_ago,
        downloads=info.downloads,
        latest_version=info.latest_version,
    )


def _verdict(info: PackageInfo, risk: int) -> str:
    if not info.exists:
        return "missing"
    if risk >= 80:
        return "lookalike"
    return "ok"


def _norm(name: str) -> str:
    text = name.strip().lower()
    if text.startswith("@") and "/" in text:
        scope, _, rest = text.partition("/")
        return f"{scope}/{_SEPARATORS.sub('', rest)}"
    return _SEPARATORS.sub("", text)


def _damerau(left: str, right: str) -> int:
    if left == right:
        return 0
    rows = len(left)
    cols = len(right)
    if rows == 0:
        return cols
    if cols == 0:
        return rows
    matrix = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        matrix[i][0] = i
    for j in range(cols + 1):
        matrix[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + 1)
    return matrix[rows][cols]


if __name__ == "__main__":
    sys.exit(main())

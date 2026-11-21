"""Render squatting findings as a fixed-width table or JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

_HEADERS = ("PACKAGE", "ECOSYSTEM", "VERDICT", "RISK", "AI", "SOURCE")
# Minimum column widths. Rows never truncate; a longer cell widens its column.
_MIN_WIDTHS = (24, 9, 9, 4, 3)


@dataclass(frozen=True, slots=True)
class Finding:
    """One package spotted in an MCP config, scored for squatting risk.

    `source` is the config path the package was mapped from. `verdict` is
    `missing` (not on the registry), `lookalike` (young, at most 500 downloads,
    similar to a popular name), or `ok`.
    """

    package: str
    ecosystem: str
    verdict: str
    risk: int
    reasons: list[str]
    ai_suggested: bool
    source: str


def render_table(findings: Sequence[Finding]) -> str:
    """Return a fixed-width table sorted by risk descending, then name."""
    body = [_cells(finding) for finding in _ordered(findings)]
    widths = list(_MIN_WIDTHS)
    for row in (_HEADERS, *body):
        for index in range(5):
            widths[index] = max(widths[index], len(row[index]))
    lines = [_format_row(_HEADERS, widths)]
    lines.extend(_format_row(row, widths) for row in body)
    return "\n".join(lines) + "\n"


def render_json(findings: Sequence[Finding]) -> str:
    """Return a JSON list with a stable key order and the same sort as the table."""
    payload = [
        {
            "package": finding.package,
            "ecosystem": finding.ecosystem,
            "verdict": finding.verdict,
            "risk": finding.risk,
            "reasons": list(finding.reasons),
            "ai_suggested": finding.ai_suggested,
            "source": finding.source,
        }
        for finding in _ordered(findings)
    ]
    return json.dumps(payload, indent=2) + "\n"


def _ordered(findings: Sequence[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: (-finding.risk, finding.package, finding.source))


def _cells(finding: Finding) -> tuple[str, str, str, str, str, str]:
    return (
        finding.package,
        finding.ecosystem,
        finding.verdict,
        str(finding.risk),
        "yes" if finding.ai_suggested else "no",
        finding.source,
    )


def _format_row(row: tuple[str, ...], widths: Sequence[int]) -> str:
    parts = [f"{row[index]:<{widths[index]}}" for index in range(5)]
    parts[3] = f"{row[3]:>{widths[3]}}"
    return " ".join(parts + [row[5]])

"""Score the labeled offline corpus with the scan detection pipeline.

Each case is looked up in the fixture registry and compared with edit
distance 2 against the builtin popular list (`popular_similar`), not against
every name in the fixture. A case is flagged when risk is at least 50.
Slopsquat, typosquat, and registered_slopsquat are the positive class.
lookalike_benign is a negative, with benign.

The hard subset drops the unregistered slopsquat and typosquat rows. Those
are caught by absence alone. The exit code uses hard-subset recall and the
false-positive rate on benign plus lookalike_benign.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentsplash.registry import PackageInfo
from agentsplash.score import score

_LABELS = (
    "benign",
    "lookalike_benign",
    "slopsquat",
    "typosquat",
    "registered_slopsquat",
)
_ATTACKS = frozenset({"slopsquat", "typosquat", "registered_slopsquat"})
_TRIVIAL = frozenset({"slopsquat", "typosquat"})
_NEGATIVES = frozenset({"benign", "lookalike_benign"})
_FLAG_RISK = 50
_ECOSYSTEMS = frozenset({"pypi", "npm"})


class BenchError(Exception):
    """The corpus or the fixture registry cannot be scored."""


@dataclass(frozen=True, slots=True)
class BenchRow:
    label: str
    n: int
    detected: int
    miss: int
    false_alarm: int


@dataclass(frozen=True, slots=True)
class BenchMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class BenchReport:
    rows: tuple[BenchRow, ...]
    overall: BenchMetrics
    hard_subset: BenchMetrics
    false_positive_rate: float

    @property
    def ok(self) -> bool:
        """True when hard-subset recall is at least 0.8 and its FPR is at most 0.05.

        No registered_slopsquat rows means there is nothing to miss, so recall
        does not fail the run. The false-positive rate still counts.
        """
        hits, total, flags, negatives = _hard_counts(self.rows)
        recall_ok = total == 0 or 5 * hits >= 4 * total
        fpr_ok = negatives == 0 or 20 * flags <= negatives
        return recall_ok and fpr_ok


def run_bench(cases_path: Path | None = None, registry_path: Path | None = None) -> int:
    """Load the corpus, print the table, and return 0 or 1."""
    report = evaluate(
        _load_cases(_resolve(cases_path, "cases.json")),
        _load_registry(_resolve(registry_path, "registry.json")),
    )
    sys.stdout.write(render_report(report))
    return 0 if report.ok else 1


def evaluate(
    cases: Sequence[Mapping[str, object]],
    registry: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> BenchReport:
    """Flag every case and aggregate overall and hard-subset metrics."""
    checked = [_case(case) for case in cases]
    flagged = [_flagged(case, registry) for case in checked]
    rows: list[BenchRow] = []
    for label in _LABELS:
        indexes = [i for i, case in enumerate(checked) if case["label"] == label]
        detected = sum(1 for i in indexes if flagged[i])
        if label in _ATTACKS:
            miss = len(indexes) - detected
            false_alarm = 0
        else:
            miss = 0
            false_alarm = detected
        rows.append(
            BenchRow(
                label=label,
                n=len(indexes),
                detected=detected,
                miss=miss,
                false_alarm=false_alarm,
            )
        )
    frozen = tuple(rows)
    overall = _metrics(*_counts(frozen, attacks=_ATTACKS))
    hard = _metrics(*_counts(frozen, attacks=_ATTACKS - _TRIVIAL))
    _hits, _total, flags, negatives = _hard_counts(frozen)
    return BenchReport(frozen, overall, hard, _div(flags, negatives, empty=0.0))


def render_report(report: BenchReport) -> str:
    """Return the per-label table plus overall and hard-subset lines."""
    lines = [f"{'label':<22} {'n':>5} {'detected':>10} {'miss':>6} {'false_alarm':>12}"]
    for row in report.rows:
        lines.append(
            f"{row.label:<22} {row.n:5d} {row.detected:10d} {row.miss:6d} {row.false_alarm:12d}"
        )
    lines.append("---")
    lines.append(_metric_line("overall", report.overall))
    lines.append(_metric_line("hard_subset", report.hard_subset))
    lines.append(f"false_positive_rate {report.false_positive_rate:.3f}")
    return "\n".join(lines) + "\n"


def _metric_line(label: str, metrics: BenchMetrics) -> str:
    return (
        f"{label:<16} precision {metrics.precision:.3f} "
        f"recall {metrics.recall:.3f} f1 {metrics.f1:.3f}"
    )


def _flagged(
    case: Mapping[str, str],
    registry: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> bool:
    # Imported lazily: cli imports this module.
    from agentsplash.cli import popular_similar

    eco = case["ecosystem"]
    name = case["name"]
    info = _info(eco, name, registry.get(eco, {}))
    # Popular-list distance, not "some other name exists in the fixture".
    similar = popular_similar(name, eco)
    risk, _reasons = score(info, similar, case["label"] in _ATTACKS)
    return risk >= _FLAG_RISK


def _info(
    ecosystem: str,
    name: str,
    records: Mapping[str, Mapping[str, object]],
) -> PackageInfo:
    from agentsplash.cli import _norm

    key = _norm(name)
    for known, meta in records.items():
        if _norm(str(known)) != key:
            continue
        if not isinstance(meta, Mapping):
            raise BenchError(f"registry entry {ecosystem}/{known} is not an object")
        return PackageInfo(
            name=name,
            ecosystem=ecosystem,
            exists=True,
            first_publish_days_ago=_nonneg(meta.get("first_publish_days_ago"), ecosystem, known),
            downloads=_nonneg(meta.get("downloads"), ecosystem, known),
            latest_version=None,
        )
    return PackageInfo(
        name=name,
        ecosystem=ecosystem,
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )


def _metrics(tp: int, fp: int, fn: int) -> BenchMetrics:
    precision = _div(tp, tp + fp, empty=1.0)
    recall = _div(tp, tp + fn, empty=1.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return BenchMetrics(precision, recall, f1)


def _counts(rows: Sequence[BenchRow], *, attacks: frozenset[str]) -> tuple[int, int, int]:
    tp = 0
    fp = 0
    fn = 0
    for row in rows:
        if row.label in attacks:
            tp += row.detected
            fn += row.miss
        elif row.label in _NEGATIVES:
            fp += row.false_alarm
    return tp, fp, fn


def _hard_counts(rows: Sequence[BenchRow]) -> tuple[int, int, int, int]:
    hits = 0
    total = 0
    flags = 0
    negatives = 0
    for row in rows:
        if row.label == "registered_slopsquat":
            hits = row.detected
            total = row.n
        elif row.label in _NEGATIVES:
            flags += row.false_alarm
            negatives += row.n
    return hits, total, flags, negatives


def _div(num: int, den: int, *, empty: float) -> float:
    if den == 0:
        return empty
    return num / den


def _load_cases(path: Path) -> list[object]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise BenchError(f"{path} must be a list of cases")
    return payload


def _load_registry(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise BenchError(f"{path} must be a registry object")
    registry: dict[str, dict[str, dict[str, object]]] = {}
    for eco, table in payload.items():
        if eco not in _ECOSYSTEMS:
            raise BenchError(f"unsupported ecosystem in registry: {eco}")
        if not isinstance(table, dict):
            raise BenchError(f"registry.{eco} must be an object")
        cleaned: dict[str, dict[str, object]] = {}
        for name, meta in table.items():
            if not isinstance(name, str) or not name.strip():
                raise BenchError(f"registry.{eco} has an empty package name")
            if not isinstance(meta, dict):
                raise BenchError(f"registry entry {eco}/{name} is not an object")
            cleaned[name] = meta
        registry[eco] = cleaned
    return registry


def _case(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise BenchError("each case must be an object")
    eco = raw.get("ecosystem")
    name = raw.get("name")
    label = raw.get("label")
    why = raw.get("why")
    if eco not in _ECOSYSTEMS:
        raise BenchError(f"case has an unsupported ecosystem: {eco!r}")
    if not isinstance(name, str) or not name.strip():
        raise BenchError("case name is empty")
    if label not in _LABELS:
        raise BenchError(f"case {name} has an unknown label: {label!r}")
    if not isinstance(why, str) or not why.strip():
        raise BenchError(f"case {name} is missing why")
    return {"ecosystem": eco, "name": name.strip(), "label": label, "why": why.strip()}


def _nonneg(value: object, ecosystem: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchError(f"registry entry {ecosystem}/{name} has a bad count: {value!r}")
    return value


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchError(f"{path} is not valid JSON") from exc


def _resolve(override: Path | None, name: str) -> Path:
    if override is not None:
        return override
    bundled = Path(__file__).resolve().parents[2] / "bench" / "data" / name
    if bundled.is_file():
        return bundled
    return Path.cwd() / "bench" / "data" / name

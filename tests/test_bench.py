"""Offline benchmark. No registry traffic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentsplash.cli import main, popular_similar, similar_names

ROOT = Path(__file__).parents[1]
_HEALTHY = {"first_publish_days_ago": 2000, "downloads": 5_000_000}


def test_similar_names_honors_an_explicit_catalog() -> None:
    assert similar_names("numppy", "pypi", ["numpy", "pandas"]) == ["numpy"]
    assert similar_names("numppy", "pypi", ["pandas"]) == []


def test_popular_similar_is_not_mere_existence() -> None:
    assert popular_similar("dateutil", "pypi") == []
    assert "requests" in popular_similar("requesst", "pypi")


def test_shipped_corpus_meets_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench"])
    out = capsys.readouterr().out
    assert code == 0
    header = out.splitlines()[0].split()
    assert header == ["label", "n", "detected", "miss", "false_alarm"]
    assert "\n---\n" in out
    assert "overall" in out
    assert "hard_subset" in out
    assert "precision" in out
    assert "recall" in out
    assert "f1" in out
    assert "false_positive_rate" in out


def test_shipped_corpus_counts() -> None:
    cases = json.loads((ROOT / "bench" / "data" / "cases.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "bench" / "data" / "registry.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["label"]] = counts.get(case["label"], 0) + 1
    assert counts == {
        "benign": 100,
        "lookalike_benign": 20,
        "slopsquat": 12,
        "typosquat": 12,
        "registered_slopsquat": 20,
    }
    for case in cases:
        table = registry[case["ecosystem"]]
        present = case["name"] in table
        if case["label"] in {"slopsquat", "typosquat"}:
            assert not present
        if case["label"] == "registered_slopsquat":
            assert present
            meta = table[case["name"]]
            assert 31 <= meta["first_publish_days_ago"] <= 60
            assert meta["downloads"] <= 500
            assert popular_similar(case["name"], case["ecosystem"])
        if case["label"] == "lookalike_benign":
            assert present
            meta = table[case["name"]]
            assert meta["first_publish_days_ago"] <= 60
            assert meta["downloads"] <= 500
            assert popular_similar(case["name"], case["ecosystem"]) == []


def test_generator_matches_committed_files(tmp_path: Path) -> None:
    script = ROOT / "bench" / "generate.py"
    subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], check=True)
    data = ROOT / "bench" / "data"
    for name in ("registry.json", "cases.json"):
        assert (tmp_path / name).read_text(encoding="utf-8") == (data / name).read_text(
            encoding="utf-8"
        )


def test_recall_at_the_gate_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 4 flagged attacks and 1 healthy registered typo → recall 0.8 exactly.
    names = {f"pkg{index}": dict(_HEALTHY) for index in range(4)}
    names["kept"] = dict(_HEALTHY)
    # Present and healthy, so this attack is a miss. Four absent attacks are flagged.
    names["registered-typo"] = dict(_HEALTHY)
    cases = [
        _case("pypi", "kept", "benign"),
        _case("pypi", "registered-typo", "typosquat"),
    ]
    cases.extend(_case("pypi", f"missing{index}", "slopsquat") for index in range(4))
    code = _run(tmp_path, cases, {"pypi": names, "npm": {}})
    assert code == 0
    assert "recall 0.800" in capsys.readouterr().out


def test_fpr_at_the_gate_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1 flagged benign out of 20 → FPR 0.05, which is still allowed.
    names = {f"pkg{index}": dict(_HEALTHY) for index in range(19)}
    cases = [_case("pypi", name, "benign") for name in names]
    cases.append(_case("pypi", "not-real", "benign"))
    cases.append(_case("pypi", "missing-attack", "slopsquat"))
    code = _run(tmp_path, cases, {"pypi": names, "npm": {}})
    assert code == 0
    assert "false_positive_rate 0.050" in capsys.readouterr().out


def test_low_recall_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Old enough that the metadata rule misses it. Hard-subset recall is 0.
    registry = {
        "pypi": {
            "requests": dict(_HEALTHY),
            "requesst": {"first_publish_days_ago": 400, "downloads": 50},
        },
        "npm": {},
    }
    cases = [
        _case("pypi", "requests", "benign"),
        _case("pypi", "requesst", "registered_slopsquat"),
    ]
    assert _run(tmp_path, cases, registry) == 1
    assert "recall 0.000" in capsys.readouterr().out


def test_fpr_above_the_gate_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = {f"pkg{index}": dict(_HEALTHY) for index in range(18)}
    cases = [_case("pypi", name, "benign") for name in names]
    cases.append(_case("pypi", "ghost-a", "benign"))
    cases.append(_case("pypi", "ghost-b", "benign"))
    cases.append(_case("pypi", "missing-attack", "typosquat"))
    assert _run(tmp_path, cases, {"pypi": names, "npm": {}}) == 1
    captured = capsys.readouterr()
    assert "false_positive_rate 0.100" in captured.out
    assert captured.err == ""


def test_registered_young_lookalike_is_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = {
        "pypi": {
            "requests": dict(_HEALTHY),
            "reqeusts": {"first_publish_days_ago": 3, "downloads": 12},
        },
        "npm": {},
    }
    cases = [
        _case("pypi", "requests", "benign"),
        _case("pypi", "reqeusts", "typosquat"),
    ]
    assert _run(tmp_path, cases, registry) == 0
    rows = {
        line.split()[0]: line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.split() and line.split()[0] in {"benign", "slopsquat", "typosquat"}
    }
    assert rows["typosquat"] == ["typosquat", "1", "1", "0", "0"]
    assert rows["benign"] == ["benign", "1", "0", "0", "0"]


def test_young_registry_neighbor_is_not_a_hard_false_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    young = {"first_publish_days_ago": 10, "downloads": 20}
    registry = {
        "pypi": {
            "requests": dict(_HEALTHY),
            "dateutil": dict(young),
            "dateutils": dict(young),
            "requesst": dict(young),
        },
        "npm": {},
    }
    cases = [
        _case("pypi", "requests", "benign"),
        _case("pypi", "dateutil", "lookalike_benign"),
        _case("pypi", "dateutils", "lookalike_benign"),
        _case("pypi", "requesst", "registered_slopsquat"),
    ]
    assert _run(tmp_path, cases, registry) == 0
    captured = capsys.readouterr().out
    rows = {
        line.split()[0]: line.split()
        for line in captured.splitlines()
        if line.split()
        and line.split()[0] in {"benign", "lookalike_benign", "registered_slopsquat"}
    }
    assert rows["lookalike_benign"] == ["lookalike_benign", "2", "0", "0", "0"]
    assert rows["registered_slopsquat"] == ["registered_slopsquat", "1", "1", "0", "0"]
    assert "hard_subset" in captured
    assert "recall 1.000" in captured


def test_bad_corpus_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = tmp_path / "cases.json"
    registry = tmp_path / "registry.json"
    cases.write_text("{", encoding="utf-8")
    registry.write_text("{}", encoding="utf-8")
    assert main(["bench", "--cases", str(cases), "--registry", str(registry)]) == 2
    assert "error:" in capsys.readouterr().err


def _run(tmp_path: Path, cases: list[dict[str, str]], registry: dict[str, object]) -> int:
    return main(
        [
            "bench",
            "--cases",
            str(_dump(tmp_path, "cases.json", cases)),
            "--registry",
            str(_dump(tmp_path, "registry.json", registry)),
        ]
    )


def _case(ecosystem: str, name: str, label: str) -> dict[str, str]:
    return {"ecosystem": ecosystem, "name": name, "label": label, "why": label}


def _dump(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

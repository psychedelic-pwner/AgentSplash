"""Squatting score rules. No registry access."""

from __future__ import annotations

from agentsplash.registry import PackageInfo
from agentsplash.score import score


def _info(**overrides: object) -> PackageInfo:
    fields: dict[str, object] = {
        "name": "reqests",
        "ecosystem": "pypi",
        "exists": True,
        "first_publish_days_ago": 7,
        "downloads": 10,
        "latest_version": "0.0.1",
    }
    fields.update(overrides)
    return PackageInfo(**fields)  # type: ignore[arg-type]


def test_nonexistent_scores_100_with_reason() -> None:
    info = _info(
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )
    risk, reasons = score(info, similar_to=["requests"], ai_suggested=False)
    assert risk == 100
    assert reasons == ["pypi package reqests does not exist"]


def test_ai_suggested_nonexistent_is_a_hard_block() -> None:
    info = _info(
        name="lodahs",
        ecosystem="npm",
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )
    risk, reasons = score(info, similar_to=[], ai_suggested=True)
    assert risk == 100
    assert reasons[0] == "npm package lodahs does not exist"
    assert any(reason.startswith("hard block:") for reason in reasons)
    assert "AI suggested" in reasons[1]
    assert "does not exist" in reasons[1]


def test_young_unpopular_lookalike_scores_80() -> None:
    info = _info(first_publish_days_ago=60, downloads=500)
    risk, reasons = score(info, similar_to=["requests", "reqwest"], ai_suggested=False)
    assert risk >= 50
    assert risk == 80
    assert len(reasons) == 1
    assert "60 days ago" in reasons[0]
    assert "500 downloads" in reasons[0]
    assert "requests" in reasons[0]
    assert "reqwest" in reasons[0]


def test_rule_requires_all_three_signals() -> None:
    young = _info(first_publish_days_ago=60, downloads=500)
    assert score(young, similar_to=[], ai_suggested=True) == (0, [])

    older = _info(first_publish_days_ago=61, downloads=0)
    assert score(older, similar_to=["requests"], ai_suggested=False) == (0, [])

    popular = _info(first_publish_days_ago=0, downloads=501)
    assert score(popular, similar_to=["requests"], ai_suggested=False) == (0, [])

    still_popular = _info(first_publish_days_ago=0, downloads=1000)
    assert score(still_popular, similar_to=["requests"], ai_suggested=False) == (0, [])

    inside = _info(first_publish_days_ago=31, downloads=0)
    assert score(inside, similar_to=["requests"], ai_suggested=False)[0] >= 50

    unknown = _info(first_publish_days_ago=1, downloads=None)
    assert score(unknown, similar_to=["requests"], ai_suggested=False) == (0, [])

    undated = _info(first_publish_days_ago=None, downloads=0)
    assert score(undated, similar_to=["requests"], ai_suggested=False) == (0, [])


def test_scores_stay_in_range() -> None:
    missing = _info(exists=False, first_publish_days_ago=None, downloads=None, latest_version=None)
    established = _info(first_publish_days_ago=400, downloads=9_000_000, latest_version="2.32.3")
    for risk, _reasons in (
        score(missing, similar_to=["requests"], ai_suggested=True),
        score(established, similar_to=[], ai_suggested=False),
        score(_info(), similar_to=["requests"], ai_suggested=False),
    ):
        assert isinstance(risk, int)
        assert 0 <= risk <= 100

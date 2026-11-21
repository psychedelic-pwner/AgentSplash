"""Turn registry metadata into a 0-100 squatting risk."""

from __future__ import annotations

from agentsplash.registry import PackageInfo

_YOUNG_DAYS = 60
_LOW_DOWNLOADS = 500


def score(
    info: PackageInfo,
    similar_to: list[str],
    ai_suggested: bool,
) -> tuple[int, list[str]]:
    """Return `(risk, reasons)`.

    A name that is not on the registry scores 100. When that name was also
    suggested by an AI, the reasons include a hard block. A name that exists,
    was first published at most 60 days ago, has at most 500 downloads, and
    resembles at least one popular package scores 80. Anything else scores 0.
    The caller decides what counts as popular; an empty `similar_to` is not a hit.
    """
    if not info.exists:
        reasons = [f"{info.ecosystem} package {info.name} does not exist"]
        if ai_suggested:
            reasons.append(
                "hard block: AI suggested "
                f"{info.ecosystem} package {info.name}, which does not exist"
            )
        return 100, reasons

    published = info.first_publish_days_ago
    downloads = info.downloads
    young = published is not None and published <= _YOUNG_DAYS
    unpopular = downloads is not None and downloads <= _LOW_DOWNLOADS
    if young and unpopular and similar_to:
        shown = ", ".join(similar_to)
        return 80, [
            (
                f"{info.ecosystem} package {info.name} was first published "
                f"{published} days ago with {downloads} downloads "
                f"and is similar to {shown}"
            )
        ]
    return 0, []

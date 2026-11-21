"""PyPI and npm lookups through one injectable HTTP transport."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


class RegistryError(Exception):
    """A registry lookup failed because of the network, a timeout, or a bad response."""


@dataclass(frozen=True, slots=True)
class PackageInfo:
    name: str
    ecosystem: str
    exists: bool
    first_publish_days_ago: int | None
    downloads: int | None
    latest_version: str | None


# (url, timeout_seconds) -> (http_status, body)
Transport = Callable[[str, float], tuple[int, bytes]]
Clock = Callable[[], datetime]

_ECOSYSTEMS = frozenset({"pypi", "npm"})
_DEFAULT_TIMEOUT = 10.0
_USER_AGENT = "agentsplash/0.1.0"


def default_transport(url: str, timeout: float) -> tuple[int, bytes]:
    """GET `url` with `timeout` seconds. HTTP statuses are returned, not raised.

    Connection failures and timeouts raise `RegistryError`. A 404 is a normal
    result meaning the package name is absent.
    """
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
            return status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        return int(exc.code), raw
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RegistryError(f"registry request failed for {url}: {exc}") from exc


class RegistryClient:
    """Look up a package on PyPI or npm.

    `transport` replaces HTTP. Tests pass a fake `(url, timeout) -> (status, body)`.
    `now` replaces the clock used to turn publish timestamps into day counts.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        now: Clock | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._transport: Transport = transport if transport is not None else default_transport
        self._timeout = float(timeout)
        self._now: Clock = now if now is not None else (lambda: datetime.now(UTC))

    def exists(self, ecosystem: str, name: str) -> PackageInfo:
        """Return metadata for `name` on `ecosystem` (`pypi` or `npm`)."""
        eco = ecosystem.strip().lower()
        if eco not in _ECOSYSTEMS:
            raise RegistryError(f"unsupported ecosystem: {ecosystem}")
        package = name.strip()
        if not package:
            raise RegistryError("package name is empty")

        url = _url_for(eco, package)
        status, body = self._call(url)
        if status == 404:
            return _missing(package, eco)
        if status < 200 or status >= 300:
            raise RegistryError(f"registry returned HTTP {status} for {url}")

        payload = _decode_object(body, url)
        if eco == "pypi":
            return self._from_pypi(package, payload)
        return self._from_npm(package, payload)

    def _call(self, url: str) -> tuple[int, bytes]:
        try:
            result = self._transport(url, self._timeout)
        except RegistryError:
            raise
        except (TimeoutError, OSError) as exc:
            raise RegistryError(f"registry request failed for {url}: {exc}") from exc
        except Exception as exc:
            raise RegistryError(f"registry request failed for {url}: {exc}") from exc

        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or isinstance(result[0], bool)
            or not isinstance(result[0], int)
            or not isinstance(result[1], (bytes, bytearray))
        ):
            raise RegistryError(f"transport returned an invalid response for {url}")
        return int(result[0]), bytes(result[1])

    def _from_pypi(self, name: str, payload: dict[str, Any]) -> PackageInfo:
        info = payload.get("info")
        if not isinstance(info, dict):
            raise RegistryError(f"pypi response for {name} is missing info")
        reported = info.get("name")
        package = reported.strip() if isinstance(reported, str) and reported.strip() else name
        version = info.get("version")
        latest = version if isinstance(version, str) and version else None
        return PackageInfo(
            name=package,
            ecosystem="pypi",
            exists=True,
            first_publish_days_ago=self._earliest_pypi(payload.get("releases")),
            downloads=_nonneg_int(info.get("downloads")),
            latest_version=latest,
        )

    def _from_npm(self, name: str, payload: dict[str, Any]) -> PackageInfo:
        # Some mirrors answer 200 with an error object and no version map.
        if payload.get("error") and "versions" not in payload and "dist-tags" not in payload:
            return _missing(name, "npm")
        reported = payload.get("name")
        package = reported.strip() if isinstance(reported, str) and reported.strip() else name
        dist_tags = payload.get("dist-tags")
        latest: str | None = None
        if isinstance(dist_tags, dict):
            raw_latest = dist_tags.get("latest")
            if isinstance(raw_latest, str) and raw_latest:
                latest = raw_latest
        downloads = _nonneg_int(payload.get("downloads"))
        return PackageInfo(
            name=package,
            ecosystem="npm",
            exists=True,
            first_publish_days_ago=self._npm_first_publish(payload.get("time")),
            downloads=downloads,
            latest_version=latest,
        )

    def _earliest_pypi(self, releases: Any) -> int | None:
        if not isinstance(releases, dict):
            return None
        earliest: datetime | None = None
        for files in releases.values():
            if not isinstance(files, list):
                continue
            for item in files:
                if not isinstance(item, dict):
                    continue
                stamp = _parse_time(item.get("upload_time_iso_8601") or item.get("upload_time"))
                if stamp is not None and (earliest is None or stamp < earliest):
                    earliest = stamp
        if earliest is None:
            return None
        return self._days_ago(earliest)

    def _npm_first_publish(self, times: Any) -> int | None:
        if not isinstance(times, dict):
            return None
        created = _parse_time(times.get("created"))
        if created is not None:
            return self._days_ago(created)
        earliest: datetime | None = None
        for key, value in times.items():
            if key in {"created", "modified"}:
                continue
            stamp = _parse_time(value)
            if stamp is not None and (earliest is None or stamp < earliest):
                earliest = stamp
        if earliest is None:
            return None
        return self._days_ago(earliest)

    def _days_ago(self, then: datetime) -> int:
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        else:
            now = now.astimezone(UTC)
        days = (now - then).days
        return max(0, days)


def _url_for(ecosystem: str, name: str) -> str:
    quoted = urllib.parse.quote(name, safe="")
    if ecosystem == "pypi":
        return f"https://pypi.org/pypi/{quoted}/json"
    return f"https://registry.npmjs.org/{quoted}"


def _missing(name: str, ecosystem: str) -> PackageInfo:
    return PackageInfo(
        name=name,
        ecosystem=ecosystem,
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )


def _decode_object(body: bytes, url: str) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry returned invalid JSON for {url}") from exc
    if not isinstance(payload, dict):
        raise RegistryError(f"registry returned unexpected JSON for {url}")
    return payload


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _nonneg_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None

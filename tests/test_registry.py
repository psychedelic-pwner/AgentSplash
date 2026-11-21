"""Registry client tests. Every lookup uses a fake transport."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any, Self

import pytest

from agentsplash.registry import (
    PackageInfo,
    RegistryClient,
    RegistryError,
    default_transport,
)

_NOW = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)


def _client(
    payload: tuple[int, bytes] | Exception,
    *,
    timeout: float = 5.0,
) -> tuple[RegistryClient, list[tuple[str, float]]]:
    seen: list[tuple[str, float]] = []

    def transport(url: str, timeout_s: float) -> tuple[int, bytes]:
        seen.append((url, timeout_s))
        if isinstance(payload, Exception):
            raise payload
        return payload

    return RegistryClient(transport=transport, timeout=timeout, now=lambda: _NOW), seen


def _pypi_body(**overrides: Any) -> bytes:
    body: dict[str, Any] = {
        "info": {"name": "Requests", "version": "2.32.3", "downloads": 50},
        "releases": {
            "1.0.0": [{"upload_time_iso_8601": "2024-01-17T12:00:00Z"}],
            "2.32.3": [{"upload_time_iso_8601": "2024-01-30T12:00:00Z"}],
        },
    }
    body.update(overrides)
    return json.dumps(body).encode()


def _npm_body(**overrides: Any) -> bytes:
    body: dict[str, Any] = {
        "name": "left-pad",
        "dist-tags": {"latest": "1.3.0"},
        "time": {
            "created": "2024-01-02T12:00:00.000Z",
            "modified": "2024-01-20T00:00:00.000Z",
            "1.3.0": "2024-01-20T00:00:00.000Z",
        },
        "downloads": 12,
    }
    body.update(overrides)
    return json.dumps(body).encode()


def test_pypi_hit_uses_earliest_upload_and_passes_timeout() -> None:
    client, seen = _client((200, _pypi_body()), timeout=3.5)
    info = client.exists("PyPI", " Requests ")
    assert info == PackageInfo(
        name="Requests",
        ecosystem="pypi",
        exists=True,
        first_publish_days_ago=15,
        downloads=50,
        latest_version="2.32.3",
    )
    assert seen == [("https://pypi.org/pypi/Requests/json", 3.5)]


def test_pypi_unknown_downloads_are_none() -> None:
    body = _pypi_body()
    parsed = json.loads(body)
    parsed["info"]["downloads"] = -1
    client, _seen = _client((200, json.dumps(parsed).encode()))
    info = client.exists("pypi", "Requests")
    assert info.downloads is None
    assert info.exists is True


def test_pypi_and_npm_missing_are_not_errors() -> None:
    client, seen = _client((404, b'{"message":"Not Found"}'))
    missing = client.exists("pypi", "no-such-pkg")
    assert missing == PackageInfo(
        name="no-such-pkg",
        ecosystem="pypi",
        exists=False,
        first_publish_days_ago=None,
        downloads=None,
        latest_version=None,
    )
    npm, npm_seen = _client((404, b""))
    assert npm.exists("npm", "no-such-pkg").exists is False
    assert seen[0][0].startswith("https://pypi.org/pypi/")
    assert npm_seen[0][0] == "https://registry.npmjs.org/no-such-pkg"


def test_npm_hit_uses_created_time_and_encodes_scope() -> None:
    client, seen = _client((200, _npm_body()))
    info = client.exists("npm", "@types/node")
    assert info == PackageInfo(
        name="left-pad",
        ecosystem="npm",
        exists=True,
        first_publish_days_ago=30,
        downloads=12,
        latest_version="1.3.0",
    )
    assert seen == [("https://registry.npmjs.org/%40types%2Fnode", 5.0)]


def test_npm_without_created_uses_earliest_version_time() -> None:
    body = _npm_body(
        time={
            "modified": "2024-01-31T00:00:00Z",
            "0.1.0": "2024-01-22T12:00:00Z",
            "1.3.0": "2024-01-28T00:00:00Z",
        }
    )
    client, _seen = _client((200, body))
    info = client.exists("npm", "left-pad")
    assert info.first_publish_days_ago == 10


def test_npm_error_object_is_missing() -> None:
    client, _seen = _client((200, b'{"error":"Not found"}'))
    info = client.exists("npm", "ghost")
    assert info.exists is False
    assert info.latest_version is None


def test_http_failure_timeout_and_bad_json_raise() -> None:
    client, _seen = _client((500, b"upstream down"))
    with pytest.raises(RegistryError, match="HTTP 500"):
        client.exists("pypi", "requests")

    timed_out, _seen = _client(TimeoutError("timed out"))
    with pytest.raises(RegistryError, match="timed out"):
        timed_out.exists("npm", "left-pad")

    garbage, _seen = _client((200, b"not-json"))
    with pytest.raises(RegistryError, match="invalid JSON"):
        garbage.exists("pypi", "requests")


def test_unsupported_ecosystem_and_empty_name() -> None:
    client, seen = _client((200, b"{}"))
    with pytest.raises(RegistryError, match="unsupported ecosystem"):
        client.exists("crates", "serde")
    with pytest.raises(RegistryError, match="empty"):
        client.exists("pypi", "   ")
    assert seen == []


def test_default_transport_passes_timeout_and_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, float | None] = {}

    class _Response:
        status = 200

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    def ok_open(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        seen["timeout"] = timeout
        assert request.get_header("User-agent") == "agentsplash/0.1.0"
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", ok_open)
    status, body = default_transport("https://pypi.org/pypi/example/json", 2.5)
    assert status == 200
    assert body == b"{}"
    assert seen["timeout"] == 2.5

    def missing_open(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=io.BytesIO(b"missing"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", missing_open)
    status, body = default_transport("https://registry.npmjs.org/missing", 1.0)
    assert status == 404
    assert body == b"missing"

    def boom_open(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", boom_open)
    with pytest.raises(RegistryError, match="timed out"):
        default_transport("https://pypi.org/pypi/example/json", 1.0)

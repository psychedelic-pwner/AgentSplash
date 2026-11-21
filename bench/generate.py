"""Build the offline benchmark corpus.

The registry starts as a fixed list of real, popular PyPI and npm names. Age
and download counts on those rows are synthetic: SHA-256(SEED | ecosystem | name)
mapped into a healthy range (at least 500 days old and at least 2_000_000
downloads), so the lookalike rule cannot fire on a benign name. They are not
live registry statistics.

Slopsquat names are a fixed list of plausible compounds an LLM might invent.
None of them are in the registry, and each one's normalized form is more
than edit distance 2 from every registry name.

Typosquat names are normalized Damerau-Levenshtein distance 1 or 2 from a
real name and are not in the registry. Mutations are adjacent transpositions
and duplicated characters (the shape of ``reqeusts``, ``numppy``, and
``react-domm``). Those three examples are kept first. Further names come from
walking parents, then transpositions, then duplications, in SHA-256(SEED | ...)
order until each ecosystem has six typos. ``SEED`` makes that walk
reproducible without ``random.Random``.

Registered slopsquats are the same kind of mutation, but the name is inserted
into the fixture with ``first_publish_days_ago`` in 31..60 and ``downloads``
in 0..500. Five of them are pinned to 60 days and five to 500 downloads, so
a strict ``<`` misses enough rows for hard-subset recall to fall under 0.8.
Detection has to come from metadata plus edit distance 2 against the builtin
popular list.

Lookalike-benign rows are real packages. Most are edit distance 1 from a
different real package that is also in the fixture. A few are maintainer-blessed
names (``python-dateutil`` and other established projects) with the same young,
low-download metadata. None of them are within distance 2 of the builtin
popular list, so a scorer that only checks existence, or that treats any
registry neighbor as similar, flags them and the false-positive rate moves.

Usage:
    python bench/generate.py
    python bench/generate.py --out bench/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentsplash.cli import POPULAR, _damerau, _norm, popular_similar

SEED = 20261121
PER_ECOSYSTEM = 50
TYPOS_PER_ECOSYSTEM = 6
REGISTERED_PER_ECOSYSTEM = 10
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_LABEL_ORDER = {
    "benign": 0,
    "lookalike_benign": 1,
    "slopsquat": 2,
    "typosquat": 3,
    "registered_slopsquat": 4,
}

# Real packages at normalized edit distance 1 from each other, and not within
# distance 2 of the builtin popular list. Both names are registered.
LOOKALIKE_PAIRS = (
    ("pypi", "dateutil", "dateutils"),
    ("pypi", "tomli", "tomli-w"),
    ("pypi", "oauthlib", "authlib"),
    ("pypi", "markdown", "markdown2"),
    ("pypi", "cbor", "cbor2"),
    ("pypi", "chardet", "cchardet"),
    ("npm", "through", "through2"),
    ("npm", "safe-buffer", "safer-buffer"),
)

# Real projects whose names are not popular-list lookalikes. The fixture still
# gives them young, low-download metadata so youth alone is not a signal.
BLESSED = (
    ("pypi", "python-dateutil"),
    ("pypi", "pendulum"),
    ("pypi", "platformdirs"),
    ("pypi", "email-validator"),
)

PYPI = (
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
    "matplotlib",
    "scikit-learn",
    "torch",
    "transformers",
    "huggingface-hub",
    "virtualenv",
    "networkx",
    "coverage",
    "gunicorn",
    "seaborn",
)

NPM = (
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
    "react-router",
    "tailwindcss",
    "postcss",
    "sass",
    "nanoid",
    "date-fns",
    "classnames",
    "immer",
    "prop-types",
    "cross-env",
)

# Invented compounds. Checked against the registry at build time.
SLOPSQUAT = {
    "pypi": (
        "huggingface-cli",
        "pandas-utils",
        "numpy-fast",
        "django-rest-utils",
        "requests-async-helper",
        "sklearn-preprocess",
    ),
    "npm": (
        "react-use-debounce-ts",
        "next-auth-helpers",
        "lodash-deep-get",
        "express-json-logger",
        "vue-composable-store",
        "axios-retry-fetch",
    ),
}

# (ecosystem, typo, parent). These stay in the corpus even if the seeded
# walk would have picked a different mutation first.
PREFERRED_TYPOS = (
    ("pypi", "reqeusts", "requests"),
    ("pypi", "numppy", "numpy"),
    ("npm", "react-domm", "react-dom"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the offline benchmark corpus.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="directory for registry.json and cases.json",
    )
    args = parser.parse_args(argv)
    registry, cases = build()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "registry.json", registry)
    _write(out / "cases.json", cases)
    print(
        f"wrote {sum(len(table) for table in registry.values())} registry names, "
        f"{len(cases)} cases -> {out}"
    )
    return 0


def build() -> tuple[dict[str, dict[str, dict[str, int]]], list[dict[str, str]]]:
    """Return ``(registry, cases)``. Pure: no filesystem access."""
    if len(PYPI) != PER_ECOSYSTEM or len(set(PYPI)) != PER_ECOSYSTEM:
        raise SystemExit(f"pypi list must be {PER_ECOSYSTEM} unique names")
    if len(NPM) != PER_ECOSYSTEM or len(set(NPM)) != PER_ECOSYSTEM:
        raise SystemExit(f"npm list must be {PER_ECOSYSTEM} unique names")

    names = {"pypi": PYPI, "npm": NPM}
    registry: dict[str, dict[str, dict[str, int]]] = {
        eco: {name: _stats(eco, name) for name in package_names}
        for eco, package_names in names.items()
    }
    norms = {eco: {_norm(name) for name in package_names} for eco, package_names in names.items()}
    cases: list[dict[str, str]] = []
    for eco, package_names in names.items():
        for name in package_names:
            cases.append(
                {
                    "ecosystem": eco,
                    "name": name,
                    "label": "benign",
                    "why": f"real popular {eco} package present in the offline fixture",
                }
            )

    taken = {eco: set(norms[eco]) for eco in names}
    for eco, invented in SLOPSQUAT.items():
        for name in invented:
            _require_slopsquat(eco, name, names[eco], taken[eco])
            taken[eco].add(_norm(name))
            cases.append(
                {
                    "ecosystem": eco,
                    "name": name,
                    "label": "slopsquat",
                    "why": (
                        "plausible invented compound in the style of an LLM package "
                        "hallucination; normalized edit distance > 2 from every fixture "
                        "name, and absent from the fixture"
                    ),
                }
            )

    typos = _typos(names, taken)
    for eco, name, parent, dist in typos:
        cases.append(
            {
                "ecosystem": eco,
                "name": name,
                "label": "typosquat",
                "why": (
                    f"normalized Damerau-Levenshtein distance {dist} from {parent}; "
                    "absent from the fixture"
                ),
            }
        )

    registered = _registered_slopsquats(taken)
    for index, (eco, name, parent, dist) in enumerate(registered):
        registry[eco][name] = _pinned_young(index, eco, name)
        cases.append(
            {
                "ecosystem": eco,
                "name": name,
                "label": "registered_slopsquat",
                "why": (
                    f"registered squat of {parent}; normalized edit distance {dist}; "
                    "first_publish_days_ago <= 60 and downloads <= 500; "
                    "not caught by absence"
                ),
            }
        )

    for eco, name, why in _lookalikes(taken):
        registry[eco][name] = _young(eco, name)
        cases.append(
            {
                "ecosystem": eco,
                "name": name,
                "label": "lookalike_benign",
                "why": why,
            }
        )

    registry = {
        eco: {name: table[name] for name in sorted(table)} for eco, table in registry.items()
    }
    cases.sort(key=lambda case: (case["ecosystem"], _LABEL_ORDER[case["label"]], case["name"]))
    _require_shape(registry, cases, names)
    return registry, cases


def _typos(
    names: dict[str, tuple[str, ...]],
    taken: dict[str, set[str]],
) -> list[tuple[str, str, str, int]]:
    chosen: list[tuple[str, str, str, int]] = []
    count = {eco: 0 for eco in names}
    for eco, name, parent in PREFERRED_TYPOS:
        dist = _accept_typo(eco, name, parent, names[eco], taken[eco])
        taken[eco].add(_norm(name))
        chosen.append((eco, name, parent, dist))
        count[eco] += 1

    for eco, package_names in names.items():
        parents = _hashed(package_names, salt=f"{eco}|parents")
        for parent in parents:
            if count[eco] >= TYPOS_PER_ECOSYSTEM:
                break
            if len(_norm(parent)) < 5:
                continue
            for candidate in _ordered_mutations(parent, eco):
                if count[eco] >= TYPOS_PER_ECOSYSTEM:
                    break
                try:
                    dist = _accept_typo(eco, candidate, parent, package_names, taken[eco])
                except ValueError:
                    continue
                taken[eco].add(_norm(candidate))
                chosen.append((eco, candidate, parent, dist))
                count[eco] += 1
                break
        if count[eco] != TYPOS_PER_ECOSYSTEM:
            raise SystemExit(f"{eco} produced {count[eco]} typos, want {TYPOS_PER_ECOSYSTEM}")
    return chosen


def _accept_typo(
    ecosystem: str,
    name: str,
    parent: str,
    package_names: tuple[str, ...],
    taken: set[str],
) -> int:
    if parent not in package_names:
        raise ValueError(f"{parent} is not a {ecosystem} registry name")
    if not _NAME.fullmatch(name):
        raise ValueError(f"bad typo spelling: {name}")
    norm = _norm(name)
    parent_norm = _norm(parent)
    if not norm or norm == parent_norm or norm in taken:
        raise ValueError(f"{name} collides with the fixture or its parent")
    dist = _damerau(norm, parent_norm)
    if dist < 1 or dist > 2:
        raise ValueError(f"{name} is distance {dist} from {parent}, want 1 or 2")
    return dist


def _require_slopsquat(
    ecosystem: str,
    name: str,
    package_names: tuple[str, ...],
    taken: set[str],
) -> None:
    if not _NAME.fullmatch(name):
        raise SystemExit(f"bad slopsquat spelling: {name}")
    norm = _norm(name)
    if norm in taken:
        raise SystemExit(f"slopsquat {ecosystem}/{name} collides with the fixture")
    nearest = min(_damerau(norm, _norm(other)) for other in package_names)
    if nearest <= 2:
        raise SystemExit(
            f"slopsquat {ecosystem}/{name} is only distance {nearest} from a real name"
        )


def _ordered_mutations(name: str, ecosystem: str) -> list[str]:
    """Transpositions first, then duplicated characters. Both are distance 1.

    Deletions and neighbor-letter swaps are not used. The kept typos match
    the corpus examples (``reqeusts``, ``numppy``, ``react-domm``).
    """
    transpositions: list[str] = []
    duplications: list[str] = []
    for index in range(len(name) - 1):
        if name[index] == name[index + 1]:
            continue
        transpositions.append(name[:index] + name[index + 1] + name[index] + name[index + 2 :])
    for index in range(len(name)):
        duplications.append(name[: index + 1] + name[index] + name[index + 1 :])
    return _hashed(transpositions, salt=f"{ecosystem}|{name}|transpose") + _hashed(
        duplications, salt=f"{ecosystem}|{name}|duplicate"
    )


def _hashed(items: tuple[str, ...] | list[str], *, salt: str) -> list[str]:
    return sorted(items, key=lambda item: _digest(f"{salt}|{item}"))


def _registered_slopsquats(
    taken: dict[str, set[str]],
) -> list[tuple[str, str, str, int]]:
    """Distance-1 mutations of builtin popular names, ten per ecosystem.

    Returned in ``(ecosystem, name)`` order so the age and download pins in
    ``_pinned_young`` are stable.
    """
    chosen: list[tuple[str, str, str, int]] = []
    for eco in ("pypi", "npm"):
        count = 0
        parents = [parent for parent in POPULAR[eco] if len(_norm(parent)) >= 6]
        candidates: list[tuple[str, str]] = []
        for parent in parents:
            for candidate in _ordered_mutations(parent, eco):
                candidates.append((candidate, parent))
        ordered = sorted(candidates, key=lambda item: _digest(f"reg|{eco}|{item[0]}|{item[1]}"))
        for candidate, parent in ordered:
            if count >= REGISTERED_PER_ECOSYSTEM:
                break
            norm = _norm(candidate)
            if not _NAME.fullmatch(candidate) or not norm or norm in taken[eco]:
                continue
            dist = _damerau(norm, _norm(parent))
            if dist != 1:
                continue
            if not popular_similar(candidate, eco):
                continue
            taken[eco].add(norm)
            chosen.append((eco, candidate, parent, dist))
            count += 1
        if count != REGISTERED_PER_ECOSYSTEM:
            raise SystemExit(
                f"{eco} produced {count} registered slopsquats, want {REGISTERED_PER_ECOSYSTEM}"
            )
    chosen.sort(key=lambda item: (item[0], item[1]))
    return chosen


def _lookalikes(taken: dict[str, set[str]]) -> list[tuple[str, str, str]]:
    """Real names that should stay under the flag threshold."""
    rows: list[tuple[str, str, str]] = []
    partners = {left: right for _eco, left, right in LOOKALIKE_PAIRS}
    partners.update({right: left for _eco, left, right in LOOKALIKE_PAIRS})
    for eco, left, right in LOOKALIKE_PAIRS:
        if _damerau(_norm(left), _norm(right)) != 1:
            raise SystemExit(f"{left} and {right} are not edit distance 1")
        for name in (left, right):
            _take_lookalike(eco, name, taken[eco])
            rows.append(
                (
                    eco,
                    name,
                    (
                        f"real {eco} package at edit distance 1 from {partners[name]}; "
                        "young and low-download in the fixture, but not within edit "
                        "distance 2 of a builtin popular name"
                    ),
                )
            )
    for eco, name in BLESSED:
        _take_lookalike(eco, name, taken[eco])
        rows.append(
            (
                eco,
                name,
                (
                    f"real {eco} package with a maintainer-blessed name; young and "
                    "low-download in the fixture, and not a popular-name lookalike"
                ),
            )
        )
    if len(rows) != 20:
        raise SystemExit(f"expected 20 lookalike_benign names, got {len(rows)}")
    return rows


def _take_lookalike(ecosystem: str, name: str, taken: set[str]) -> None:
    if not _NAME.fullmatch(name):
        raise SystemExit(f"bad lookalike spelling: {name}")
    norm = _norm(name)
    if norm in taken:
        raise SystemExit(f"lookalike {ecosystem}/{name} collides with the fixture")
    hits = popular_similar(name, ecosystem)
    if hits:
        raise SystemExit(f"lookalike {ecosystem}/{name} is within distance 2 of {hits}")
    taken.add(norm)


def _stats(ecosystem: str, name: str) -> dict[str, int]:
    digest = hashlib.sha256(f"{SEED}|{ecosystem}|{name}".encode()).digest()
    return {
        "first_publish_days_ago": 500 + int.from_bytes(digest[:2], "big") % 3000,
        "downloads": 2_000_000 + int.from_bytes(digest[2:6], "big") % 80_000_000,
    }


def _young(ecosystem: str, name: str) -> dict[str, int]:
    """Metadata inside the squat window: 31..60 days and 0..500 downloads."""
    digest = hashlib.sha256(f"{SEED}|young|{ecosystem}|{name}".encode()).digest()
    return {
        "first_publish_days_ago": 31 + digest[0] % 30,
        "downloads": int.from_bytes(digest[1:3], "big") % 501,
    }


def _pinned_young(index: int, ecosystem: str, name: str) -> dict[str, int]:
    """Like ``_young``, but pin five rows to each inclusive bound.

    Indexes 0..4 are 60 days old. Indexes 5..9 have 500 downloads. A scorer
    that uses ``<`` instead of ``<=`` misses one of those blocks (recall 0.75
    on the 20 registered rows).
    """
    stats = _young(ecosystem, name)
    if index < 5:
        stats["first_publish_days_ago"] = 60
    elif index < 10:
        stats["downloads"] = 500
    return stats


def _digest(text: str) -> str:
    return hashlib.sha256(f"{SEED}|{text}".encode()).hexdigest()


def _require_shape(
    registry: dict[str, dict[str, dict[str, int]]],
    cases: list[dict[str, str]],
    names: dict[str, tuple[str, ...]],
) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["label"]] = counts.get(case["label"], 0) + 1
    expected = {
        "benign": 100,
        "slopsquat": 12,
        "typosquat": 12,
        "registered_slopsquat": 20,
        "lookalike_benign": 20,
    }
    if counts != expected:
        raise SystemExit(f"unexpected case counts: {counts}")
    for eco, package_names in names.items():
        got = {
            case["name"]
            for case in cases
            if case["label"] == "benign" and case["ecosystem"] == eco
        }
        if got != set(package_names):
            raise SystemExit(f"benign {eco} cases must be exactly the popular fixture names")
    registry_norms = {eco: {_norm(name) for name in table} for eco, table in registry.items()}
    absent = [case for case in cases if case["label"] in {"slopsquat", "typosquat"}]
    if any(_norm(case["name"]) in registry_norms[case["ecosystem"]] for case in absent):
        raise SystemExit("an unregistered attack name is present in the fixture")
    present_labels = {"registered_slopsquat", "lookalike_benign"}
    present = [case for case in cases if case["label"] in present_labels]
    if any(_norm(case["name"]) not in registry_norms[case["ecosystem"]] for case in present):
        raise SystemExit("a registered case is missing from the fixture")
    boundary_days = 0
    boundary_downloads = 0
    for case in cases:
        if case["label"] != "registered_slopsquat":
            continue
        meta = registry[case["ecosystem"]][case["name"]]
        if (
            meta["first_publish_days_ago"] < 31
            or meta["first_publish_days_ago"] > 60
            or meta["downloads"] > 500
        ):
            raise SystemExit(f"registered squat {case['name']} is outside the metadata window")
        if meta["first_publish_days_ago"] == 60:
            boundary_days += 1
        if meta["downloads"] == 500:
            boundary_downloads += 1
        if not popular_similar(case["name"], case["ecosystem"]):
            raise SystemExit(f"registered squat {case['name']} is not similar to a popular name")
    if boundary_days < 5 or boundary_downloads < 5:
        raise SystemExit(
            "need >=5 rows on each inclusive bound, "
            f"got days={boundary_days} downloads={boundary_downloads}"
        )
    for case in cases:
        if case["label"] != "lookalike_benign":
            continue
        meta = registry[case["ecosystem"]][case["name"]]
        if meta["first_publish_days_ago"] > 60 or meta["downloads"] > 500:
            raise SystemExit(f"lookalike {case['name']} is outside the metadata window")
        if popular_similar(case["name"], case["ecosystem"]):
            raise SystemExit(f"lookalike {case['name']} matches the popular list")


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

# AgentSplash

Score package names an AI assistant wants to install, and names an MCP config is about to launch, for registry squatting risk.

Two failure modes sit on that path. **Slopsquatting**: a model invents a plausible name (`huggingface-cli`, `pandas-utils`, `react-use-debounce-ts`) that is not on PyPI or npm, and someone else publishes it first. **Typosquatting**: the name is an edit of a real package (`reqeusts`, `numppy`, `react-domm`). MCP client configs (`mcp.json`, `.mcp.json`, `claude_desktop_config.json`, and other files with an `mcpServers` table) are where those names become a process. AgentSplash reads the config, maps the launch line to a registry name, and scores it. It does not install or start the package.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/agentsplash scan --path bench/example --transcripts bench/example/transcripts --offline
.venv/bin/agentsplash bench
```

`pip install -e .` exposes the `agentsplash` package (`[tool.setuptools.packages.find] where = ["src"]`) and the `agentsplash` console script. Drop `--offline` to query `https://pypi.org/pypi/<name>/json` and `https://registry.npmjs.org/<name>`.

`scan` exits 0 when every finding is under `--fail-on-risk` (default 50), 1 when any finding reaches it, and 2 when a config or registry lookup fails.

## Example

`bench/example/mcp.json` launches three servers. `bench/example/transcripts/session.jsonl` is an assistant turn that says `npm i react-use-debounce-ts`. This run used an empty home directory, so `~/.claude/projects` did not add further names.

```bash
.venv/bin/agentsplash scan --path bench/example --transcripts bench/example/transcripts --offline
```

```
PACKAGE                  ECOSYSTEM VERDICT   RISK AI  SOURCE
react-use-debounce-ts    npm       missing    100 yes bench/example/mcp.json
reqests                  pypi      lookalike   80 no  bench/example/mcp.json
requests                 pypi      ok           0 no  bench/example/mcp.json
```

Exit code: 1.

| package | what the offline pipeline did |
| --- | --- |
| `react-use-debounce-ts` | Not in the built-in offline catalog. Risk 100, verdict `missing`. The transcript marks it AI-suggested, so the reasons include a hard block. |
| `reqests` | Present in the offline map as a 3-day-old package with 12 downloads, edit distance 1 from `requests`. Risk 80, verdict `lookalike`. |
| `requests` | Old and widely downloaded. Risk 0, verdict `ok`. |

`scan --offline` uses the popular-name list in `src/agentsplash/cli.py`, plus that `reqests` stand-in. `agentsplash bench` uses `bench/data/registry.json` instead.

## How origin tracing works

A finding is AI-suggested when the package name matches one of three sources. Matching ignores case and treats `-`, `_`, and `.` as the same character. An npm scope is kept.

1. **Claude Code transcripts.** `~/.claude/projects/**/*.jsonl` is always read when that directory exists. `--transcripts` adds another JSONL file or directory. Only assistant text is used. A line that is not a JSON object is skipped.
2. **Install and import lines in that text.** `pip install`, `python -m pip install`, `uv add`, `npm install` / `npm i`, Python `import` / `from ... import`, and `require("...")`.
3. **Git.** `--git-ref REF` runs `git diff REF` and keeps added lines only. A missing git binary or a non-repository warns and contributes no names.

The same extractors run on the transcript text and on the diff. A missing package that was AI-suggested still scores 100; the extra reason is the hard block.

Risk, from `agentsplash.score`:

- Name absent from the registry: **100**.
- First published at most 60 days ago, at most 500 downloads, and similar to a popular name: **80**.
- Anything else: **0**.

`scan` measures similarity with Damerau–Levenshtein on the normalized name against the builtin popular list. Names shorter than 8 characters use distance 1. Names of 8 or more characters use distance 2. The name itself is not a lookalike of itself. `agentsplash bench` uses the same list with distance 2 at every length, so a registry neighbor that is not on that list does not count.

## Benchmark

`agentsplash bench` loads `bench/data/cases.json` and scores every name offline against `bench/data/registry.json`. A name is flagged when its risk is at least 50. The positive class is `slopsquat`, `typosquat`, and `registered_slopsquat`. `benign` and `lookalike_benign` are negatives. The false-positive rate is flagged negatives divided by negatives. The process exits 1 when **hard-subset** recall is below 0.8 or that rate is above 0.05. The hard subset drops the unregistered `slopsquat` and `typosquat` rows, which the absence rule scores 100 without looking at metadata.

Corpus (`python bench/generate.py`, seed `20261121`, 164 cases, 140 registry names):

- **100 benign** — 50 real PyPI names and 50 real npm names. Healthy synthetic metadata (at least 500 days old, at least 2,000,000 downloads).
- **20 lookalike_benign** — real packages that look risky in the fixture and must not be flagged. Sixteen are edit distance 1 from a different real package (`dateutil`/`dateutils`, `tomli`/`tomli-w`, `oauthlib`/`authlib`, `through`/`through2`, and four more pairs). Four are maintainer-blessed names (`python-dateutil`, `pendulum`, `platformdirs`, `email-validator`). All twenty are inside the squat window (31–60 days, at most 500 downloads) and none are within distance 2 of the builtin popular list.
- **12 slopsquat** — invented compounds (`huggingface-cli`, `pandas-utils`, `react-use-debounce-ts`, and nine more). Absent from the fixture, and more than edit distance 2 from every fixture name.
- **12 typosquat** — distance-1 transpositions or duplicated characters (`reqeusts`, `numppy`, `react-domm`, and nine more). Absent from the fixture.
- **20 registered_slopsquat** — the same shape of mutation, but the name is in the fixture (`panndas`, `jqurey`, `webppack`, and seventeen more) with `first_publish_days_ago` <= 60 and `downloads` <= 500. Five rows are exactly 60 days old and five have exactly 500 downloads. Similarity to the builtin popular list is what flags them.

Ages and download counts are synthetic. `SHA-256(20261121 | ecosystem | name)` picks the healthy range for the original 100 names, and `SHA-256(20261121 | young | ecosystem | name)` picks the squat window for the other registered rows. They are not live registry statistics. `bench/generate.py` writes both JSON files. Running it twice produces the same bytes.

This is the output of `.venv/bin/agentsplash bench` (exit 0):

```
label                      n   detected   miss  false_alarm
benign                   100          0      0            0
lookalike_benign          20          0      0            0
slopsquat                 12         12      0            0
typosquat                 12         12      0            0
registered_slopsquat      20         20      0            0
---
overall          precision 1.000 recall 1.000 f1 1.000
hard_subset      precision 1.000 recall 1.000 f1 1.000
false_positive_rate 0.000
```

| label | n | detected | miss | false_alarm |
| --- | ---: | ---: | ---: | ---: |
| benign | 100 | 0 | 0 | 0 |
| lookalike_benign | 20 | 0 | 0 | 0 |
| slopsquat | 12 | 12 | 0 | 0 |
| typosquat | 12 | 12 | 0 | 0 |
| registered_slopsquat | 20 | 20 | 0 | 0 |

| metric | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| overall | 1.000 | 1.000 | 1.000 |
| hard subset | 1.000 | 1.000 | 1.000 |

False-positive rate on benign plus lookalike_benign: 0.000.

The benchmark checks a naming heuristic against a synthetic registry. It does not prove the tool detects malware. Unregistered `slopsquat` and `typosquat` rows are absent, so the missing-name rule scores them 100; that half of the corpus is circular and is excluded from the hard subset. The hard subset is the part with content: a registered name is flagged only when it is at most 60 days old, has at most 500 downloads, and is within edit distance 2 of the builtin popular list. Real packages that are merely close to each other, and young blessed names, stay under 50. Nothing here installs a package, reads code, or checks a maintainer. A live registry will disagree with the hashed ages and with which attack names already exist. A new legitimate package that is also an edit of a popular name still scores 80.

## Limitations

- Similarity is edit distance on the normalized name. A name missing from the registry scores 100 at any distance. A published name scores 80 only when it was first published at most 60 days ago, has at most 500 downloads, and is close to a popular name. An older or more-downloaded typosquat that is already published scores 0.
- There is no payload inspection. The tool never installs a package, never downloads a release, and never reads code, install scripts, or maintainer metadata.
- The offline fixture is synthetic. The popular names and the lookalike names are real. The ages, download counts, and the squat spellings were generated. A live registry will disagree with both the counts and which attack names already exist.
- `scan --offline` and `agentsplash bench` do not share one existence catalog. Scan treats the builtin popular list as old and heavily downloaded, plus the `reqests` stand-in. Bench reads existence and metadata from `bench/data/registry.json`. Both measure similarity against the builtin popular list.

AgentSplash complements manifest scanners. Those tools inspect lockfiles, dependency trees, and published artifacts. This one scores the name an assistant suggested and the name an MCP config is about to launch.

## Credits

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

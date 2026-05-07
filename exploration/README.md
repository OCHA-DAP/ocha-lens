# `exploration/` — transitional refactor source

This directory holds **verbatim copies** of code from other branches/repos that is being ported into the ocha-lens public API. It is intentionally tracked in git for transparency: anyone reviewing the resulting `src/ocha_lens/datasources/` modules can compare them against the original source.

## What's here

| Subdir | Source | Why it's here |
|---|---|---|
| `from_hannah_pr2/` | `OCHA-DAP/ds-storm-impact-harmonisation` PR [#2](https://github.com/OCHA-DAP/ds-storm-impact-harmonisation/pull/2), branch `gdacs-adam-data` | Initial GDACS & ADAM historical exposure pipeline implementation. To be refactored into `src/ocha_lens/datasources/gdacs.py` (and possibly `adam.py`) per ocha-lens [#41](https://github.com/OCHA-DAP/ocha-lens/issues/41). |
| `from_zack_merge_cerf_exposure/` | `OCHA-DAP/ds-storm-impact-harmonisation` branch `merge-cerf-exposure` | GDACS real-time client + matching scaffolding referenced from ocha-lens [#40](https://github.com/OCHA-DAP/ocha-lens/issues/40). |

Each file's header comment cites the exact source repo, ref, commit SHA, and original path.

## Rules

1. **Do not modify files in this directory.** They are the refactor reference. Treat as read-only history.
2. **Do not import from this directory in `src/ocha_lens/`.** This dir is not part of the package (the wheel only ships `src/ocha_lens/`).
3. **Delete this directory once the refactor lands.** When the refactored functions are in `src/ocha_lens/datasources/` and the upstream source PRs are closed/merged, drop `exploration/` in a single cleanup commit.

## Why tracked, not gitignored

Tracking lets reviewers diff the public API against the original implementation and verify nothing was silently lost in translation. The wheel build (`packages = ["src/ocha_lens"]` in `pyproject.toml`) excludes everything outside `src/ocha_lens/`, so this never ships to users.

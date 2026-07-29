# LuGRE Reviewer Reproduction

> Publication staging draft. Replace every remaining `TO_BE_CONFIRMED` field
> before making the repository public.

This repository contains the Python code, task registry, and reproducibility
documentation used to regenerate the LuGRE manuscript figures and independently
recoverable scientific panels.

## Reproduction scope

- 9 complete manuscript figures are regenerated exactly from packaged
  analysis-ready inputs.
- 38 scientific panel redraws are runnable for figures whose final assembly
  chain is incomplete.
- 6 full-figure records and 1 reference-only panel are explicitly marked as
  blocked.
- Reference images are validation targets and are never plotting inputs.

The precise level and known limitation of every task are recorded in
`manifest/tasks.json` and `docs/REPRODUCIBILITY_MATRIX.md`.

## Data download

The data, generated figures, validation references, and machine-readable file
manifest are deposited separately:

- Zenodo dataset DOI: `10.5281/zenodo.21672261`
- Dataset version: `1.0.0`

Download and extract the Zenodo archive into the repository root so that the
following directories exist:

```text
data/
assets/
reference/
outputs/
```

## Environment

Create a Python environment and install the pinned dependencies:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r environment\requirements.txt
```

The published GitHub repository does not contain the 454.7 MiB portable Python
runtime from the reviewer handoff package. The dependency versions required to
recreate the environment are listed in `environment/requirements.txt`.

## Run

On Windows:

```text
run_one.cmd --list
run_one.cmd Fig3
run_all.cmd --group exact
run_all.cmd --group panel
run_all.cmd
verify_release.cmd
```

Or call the Python entry points directly:

```text
python run_one.py --list
python run_all.py --group exact
python run_all.py --group panel
```

## Data and code boundaries

The deposit contains analysis-ready numerical tables, panel-ready products,
selected raw LuGRE telemetry, generated figures, and validation references. It
contains no third-party data and does not contain complete external provider
archives. Excluded commercial, very large, or unavailable source families are
documented in `manifest/external_large_sources.csv`.

## Citation

Software DOI: `TO_BE_CONFIRMED_SOFTWARE_DOI`

Dataset DOI: `10.5281/zenodo.21672261`

See `CITATION.cff` for the software citation draft. Author names, ORCID records,
affiliations, funding, and final DOI values must be completed before release.

## Licence

Copyright (c) 2026 SJTU-GNC. Original project code is released under the MIT
License; see `LICENSE`. Python dependencies retain their upstream terms; see
`THIRD_PARTY_NOTICES.md`.

Project-owned data and figures are handled separately in the Zenodo dataset
record under CC BY 4.0. SJTU-GNC confirmed their public redistribution rights
on 2026-07-29.

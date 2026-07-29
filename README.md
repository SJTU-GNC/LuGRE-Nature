# LuGRE Reviewer Reproduction

> Public reproducibility repository for the manuscript *Lunar GNSS reveals a
> polar-dominant ionospheric amplitude response*.

This repository contains the Python code, task registry, and reproducibility
documentation used to regenerate the manuscript figures and independently
recoverable scientific panels.

## Associated manuscript

- Title: *Lunar GNSS reveals a polar-dominant ionospheric amplitude response*
- Status: ready to submit (unpublished; no article DOI has been assigned)
- Authors: Rong Yang, Zhihong Li, Andrea Nardin, Alex Minetto, Fabio Dovis,
  Y. Jade Morton, Xirui Miao, Yuquan Ma, Wei Gao, Li-Ta Hsu, and Xingqun Zhan
- Corresponding authors: Rong Yang and Y. Jade Morton

All project-original code, data, figures, images, and documentation in this
repository and its associated data deposit were prepared for this manuscript
and are attributed to its authors. Full authorship and contact metadata are
provided in `docs/MANUSCRIPT_METADATA.md`.

## Research context and responsible reuse

The dataset and software DOI records document the creators, release date,
version, and provenance of this reproducibility package. Reuse is governed by
CC BY 4.0 for the data and figures and by the MIT License for the software.

Users must comply with the applicable licence terms, retain copyright and
licence notices, provide appropriate attribution where required, and clearly
identify modifications. Reuse must not imply endorsement by the manuscript
authors or misrepresent the original data, methods, figures, or findings.

For scholarly use, please cite both the dataset DOI and software DOI below.
Publications based substantially on these resources should clearly distinguish
reused material from new contributions and should cite the associated
manuscript when a public citation becomes available. See
[REUSE_AND_CITATION.md](REUSE_AND_CITATION.md) for the complete statement and
corresponding-author contacts.

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

Software DOI: `10.5281/zenodo.21675358`

Dataset DOI: `10.5281/zenodo.21672261`

See `CITATION.cff` for the software citation metadata. The associated
manuscript is ready to submit but is not yet published; an article DOI,
complete affiliation text, ORCID records, and funding metadata may be added
when available.

## Licence

Copyright (c) 2026 SJTU-GNC. Original project code is released under the MIT
License; see `LICENSE`. Python dependencies retain their upstream terms; see
`THIRD_PARTY_NOTICES.md`.

Project-owned data and figures are handled separately in the Zenodo dataset
record under CC BY 4.0. SJTU-GNC confirmed their public redistribution rights
on 2026-07-29.

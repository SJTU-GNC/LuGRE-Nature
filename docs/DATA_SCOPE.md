# Data Scope and Size Boundary

The lite package is based primarily on a **consumed-input rule**: a file is
included when a runnable exact-figure or panel task reads it, or when it is
needed to validate that task. The only deliberate exception is an 86.65 MiB
scientific provenance bundle under `data/analysis_ready/ED3` and `ED4`. It is
retained to document the blocked full-figure chains and is not presented as a
runnable dependency.

Large text tables are stored as deterministic lossless gzip containers. This
changes only the storage representation; the plotting and QA code read the
decompressed scientific values directly.

## Included

- Accepted analysis-ready numerical tables for exact figures.
- Compact panel-ready numerical tables for the recoverable panels of the six
  blocked full figures.
- Compact archived ED3/ED4 scientific sources retained for provenance of the
  blocked accepted full-figure chains.
- Seven losslessly gzip-compressed LuGRE S-phase `TLM_RAW` text files consumed
  by the ED6–ED9 hybrid raw rebuild.
- The pinned portable Python runtime and Cartopy map resources.
- Non-numerical scientific input assets required by a plotting task.
- Accepted figures and panel references under `reference/`, never under
  `data/`.

## Excluded

- The roughly 54 GiB archival reproduction tree.
- Duplicate intermediate caches and generated display products.
- Raw source families for which no packaged script can transform the source
  into the accepted plotted table.
- Complete provider archives above 10 GiB.
- Licensed inputs whose redistribution permission is not confirmed.
- Exploratory scripts and outputs that are not on an accepted task path.

Excluding a raw archive is not hidden: `manifest/external_large_sources.csv`
records the source family, estimated size, reason, and effect on
reproducibility.

## Why unused raw files are not copied

Copying a provider archive without the pinned ingestion, quality-control, join,
and selection code does not create an end-to-end reproduction. It only makes
the package larger. This package instead ships the lowest input for which the
transformation to a plotted result is executable and auditable, and labels
that starting level explicitly.

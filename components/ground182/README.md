# Ground182 figure reproduction component

This component regenerates the current R3 Fig.3, Fig.4 and four Extended Data
figures from the frozen numerical inputs used by the selected 182-station
analysis. It works from its own directory after copying, without any E:, F:,
H:, D: or network data dependency. Install the Python requirements once.

```text
python -m pip install -r requirements.txt
python validate_inputs.py
python reproduce.py all
python verify_outputs.py
```

Individual commands are `python reproduce.py fig3`, `fig4`, `illumination`,
`geometry` (two figures), and `context`. Use `--output-dir PATH` to choose a
separate output directory. PNGs are exported at 600 dpi. `reference/` contains
the exact approved PNGs; `outputs/` contains newly regenerated results and QA.
Input data are never overwritten by these commands.

## Exact scientific version

- Selected Ground: 182 contributing stations, 29,803,114 accepted native 1-Hz
  receiver-satellite epochs, 498,430 accepted minute windows.
- Source selection SHA256:
  `07214f1264ee81a24c80ff475a5c3d516539042cada62413f1df791eedde7b32`.
- LuGRE and LEO-RO remain the separately accepted populations. Their temporal
  support and sampling need not be identical to Ground.
- No new refit, amplitude clipping, selection change or scientific rebinning is
  performed during figure regeneration.
- Ground quartic residuals, LuGRE h50 E23-method residuals and LEO-RO rolling
  median residuals retain their distinct source methods.
- Fig.3 absolute-residual P95 is different from minute width W=P95-P05.

## Files and computational boundary

| Directory | Purpose |
|---|---|
| `fig34/amplitude_source/` | Exact histogram counts, quantiles, height envelopes and actual near-space height/residual points for Fig.3 |
| `fig34/response_source/` | Every accepted Ground/LEO/LuGRE response window and Fig.4 angle-bin summaries |
| `illumination/ground/`, `illumination/nearspace/` | Actual per-window solar geometry and width inputs, not screenshots |
| `geometry/plot_data/` | Frozen per-platform/band angle CDFs, speed histograms, support counts and selection binding |
| `context/data/` | Frozen station/context layers, actual observing tracks, mission schematic assets and offline Natural Earth shapefiles |
| `context/frozen/` | Exact station coordinates and context provenance; historical coordinate-table epoch totals are not used as current analysis counts |
| `reference/` | Unmodified six user-approved figure PNGs |
| `provenance/` | Original selection manifest, per-part masks, QC decisions, source map and upstream construction-script archive |

This compact release reproduces figures and checks the retained-window
statistics from processed source data. It does not claim a complete rerun from
all original raw GNSS files: those multi-station acquisition archives, precise
orbit inputs and historical raw processing environments are documented by the
source manifests. The upstream source scripts in
`provenance/upstream_source_archive/` and the assembly script are historical
evidence, not portable entrypoints. They deliberately preserve original paths.

The observing-context panel includes separate lower-rate polar GNSS,
high-rate scintillation, radar and lunar/LEO observing layers. These are not all
part of the 182-station amplitude population, and counts across categories must
not be summed. The polar network's 30-s map label is the user-confirmed context
label, not a remeasurement of every archived raw file's cadence.

## Portability and fonts

All supported entrypoints use script-relative paths. Numeric plotting inputs,
map coastlines and tracking tables are bundled, so plotting requires no data
downloads. Arial is used when installed. DejaVu Sans, supplied with Matplotlib,
is the fallback; line wrapping or pixel-level typography can differ on another
OS or font configuration while the scientific values remain identical. No
proprietary Windows fonts are redistributed.

Python 3.12 is recommended. See the package-level environment lock and
verification report for the actually tested dependency versions. Original
provider credits and rights remain with their respective providers; this
component does not relicense third-party observations or icons.

# Main Figures 1–2 and associated Extended Data

This component freezes the R3 manuscript selected on 30 October 2026. The Ground
station expansion does not change these LuGRE-only figures. All operational
inputs are inside this directory. Paths containing original drive letters in
`provenance/` document source history and are not reproduction dependencies.

## Run

After installing the package-level Python dependencies:

```text
python components/main12/run.py --task all
python components/main12/run.py --task fig2 --output my_results
python components/main12/code/refit_lugre_h50.py --output full_refit_results
```

The last command is an optional full LuGRE refit and is slower than redrawing the
figures. It uses native measured C/N0 and verified WGS84 geometry, applies the
archived E23 robust quartic function, and compares results with frozen outputs.
Expected results never enter the fit. The height selection is >=50 km, retaining
context above 1,000 km; the ionospheric statistical shell is 50–1,000 km. Tracks
are grouped by phase, satellite and signal and split only at observed gaps
greater than 7,200 s after height selection. No synthetic gap observations enter
fits or statistics. Absolute GPS 60-s bin medians, square-root count weights and
the archived MAD rejection are retained. Constant fallbacks are explicitly
excluded from qualified quartic residuals.

## Outputs and reproduction level

| Task | Source | Reproduction target |
|---|---|---|
| Fig.1 | Editable `source_artwork/Fig1_R3.pptx` | Authored schematic, with approved PNG and optional native PowerPoint export |
| Fig.2 | Frozen h50 spatial cells, spectra, window/event/paired-frequency summaries, actual case measurements and WGS84 points | Complete numerical figure using original renderer and styles |
| E21 detrending example | Actual C/N0, exact E23 robust fit, independent geometry | Refit and complete figure; a=20–40 dB-Hz and b=−2–2 dB are display zooms only |
| Space weather | Archived hourly context and actual 2,244 accepted 60-s windows | Complete figure with OP mean widths independently recalculated |
| Antenna geometry | LuGRE point cache, NAVCEN GPS measurements and JRC GRAP workbooks | Recalculated distributions/pattern envelopes plus authored left schematic |

Figure 1's default task independently computes its finite-source Fresnel chart
and checks all six embedded series (646 points) against the editable PPTX cache.
The remaining schematic artwork is explicitly not a numerical reconstruction:
the task copies the approved reference and reports `REFERENCE_AUTHORED_SCHEMATIC`. On Windows
with Microsoft PowerPoint installed, the editable source can be exported with
`powershell -NoProfile -ExecutionPolicy Bypass -File components/main12/code/export_fig1.ps1`.
The export preserves the source document and applies the original in-memory
typography recipe. Native PowerPoint version and installed fonts can affect
rendering. This optional export is not required to run the Python data plots.

The selected antenna composite had no deterministic final assembly script in the
previous archive. This package preserves its authored left schematic as bitmap
artwork and supplies a new deterministic assembly that genuinely replots the
right-hand scientific panels from data. It does not claim final-image pixel
identity. All approved manuscript PNGs remain unchanged in `reference/`.

## Data and definitions

`data/fig2/` contains the actual plotting tables and measured samples. Quantile
widths are P95−P05; they differ from P95 absolute residuals. The event threshold is
1.5 dB for the accepted 60-s width. The current Fig.2 has 2,244 accepted windows
and 266 event windows. Map colour saturation is display-only. The matched
paired-frequency and original QC/cohort identities remain frozen.

`data/e21/` contains the measured E21 example, WGS84 geometry, archived E23 example
used to verify the algorithm and frozen expected outputs. The model evaluated
across the occultation gap is distinguished from actual observations. No
main-/sidelobe shading is added to this example.

`data/space_weather/` contains a frozen hourly grid for solar/geomagnetic indices,
the 60-s LuGRE windows and OP timing/summary metadata. The OP mean is an arithmetic
mean of accepted window widths; all-latitude values include the polar subset.

`data/antenna/` retains NAVCEN antenna-pattern source files and JRC GRAP workbooks,
source caches and derived pattern tables. The science is inherited from the
selected original antenna figure, not newly reselected by amplitude. Missing
public patterns for a spacecraft are not replaced by another spacecraft. The
schematic and first-valley diagnostics are descriptive, not fixed official lobe
boundaries. The current off-boresight sample shell is 50–1,000 km.
The inherited antenna geometry cache spans 15 January–16 March 2025 across
mission phases; it is a separate source population from the main lunar-surface
amplitude comparison. The regenerated envelopes, point counts and angle
summaries are checked against the selected source tables.

Offline Natural Earth 50m and 110m land/coastline files are included under
`assets/cartopy/`; figure generation requires no downloads. Arial is used if
installed; DejaVu Sans is the fallback. Fonts are not redistributed.

## Scope and rights

This is source-table/native-C/N0-to-figure reproduction, not mission telemetry
decoding, orbit determination, or downloading the complete public archives.
Historical source scripts in `provenance/original_code/` are retained for
traceability and are not supported entry points. Third-party data and artwork
retain their original rights; the package-level rights notice applies. Source
identifiers, original paths and SHA-256 values are in
`provenance/copied_input_manifest.json`. Do not infer a public redistribution
licence solely from inclusion in the research package.

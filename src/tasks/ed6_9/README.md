# ED6-ED9 analysis-ready reproduction

This task family redraws one Extended Data figure per entry point:

- `run_ed6.py`: OP38
- `run_ed7.py`: OP40, including the original sparse-tail trimming rule
- `run_ed8.py`: OP74
- `run_ed9.py`: compact OP76 + OP77 + OP77_1 + OP78_1

In the default mode, the observations and ECEF/WGS84 tangent geometry are read
from packaged numeric analysis-ready CSV tables. The runner recomputes trailing
60-s ROTI and redraws every panel.

With `--rebuild-derived`, the runner first executes
`src/raw_pipeline/lugre/build_observations_s_phase.py` and reads observations
from:

```text
work/derived/lugre_raw/observations_s_phase_from_raw.csv
```

That raw builder reconstructs the telemetry-backed fields and emits explicit
QA/provenance records for all processed fallbacks. The exact detrended C/N0
implementation and precise-orbit tangent-point generator are not available.
Consequently, detrended C/N0 and the ECEF/WGS84 geometry tables remain
documented packaged analysis-ready fallbacks; `--rebuild-derived` is a partial,
audited raw-data reconstruction rather than a claim of complete independence.

All runtime paths are resolved from the package location. Natural Earth 110-m
map assets are stored beside the task code so Cartopy does not download data.

Run from the package root with the bundled Python runtime:

```powershell
runtime\python_exact\python.exe -s -B ED6_main.py
runtime\python_exact\python.exe -s -B ED6_main.py --rebuild-derived
```

`prepare_analysis_data.py` documents how the packaged CSV subsets were
assembled from the development workspace. It is provenance tooling and is not
required to run the standalone reproduction package.

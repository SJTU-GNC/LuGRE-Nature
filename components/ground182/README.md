# R4 Ground182 reproduction component

This component regenerates R4 Fig.3, Fig.4 and the observing-context, geometry-angle,
scan-speed/Fresnel-time and local-time/illumination Extended Data figures. All data
lookups are relative to this directory. After installing the supplied dependencies,
it runs without the original E/H/D drives or a network connection.

```text
python validate_inputs.py
python reproduce.py all
python verify_outputs.py
```

Use `python reproduce.py fig3`, `fig4`, `illumination`, `geometry`, or `context`
for a subset. `python reproduce.py recompute` reconstructs only the numerical R4
LuGRE QC and summaries. `--output-dir PATH` selects a separate output location.
The component-level requirements and package-level offline wheels provide the
software environment. No original-machine dependency directories are imported.

## What is recomputed

`inputs_r4/` contains the original 82,761 processed LuGRE comparison epochs,
1,654 original UTC-minute/run/band candidate windows, and the original solar
geometry/metadata. The supported entrypoint reapplies point C/N0 >=25 dB-Hz,
at least 30 valid 1-Hz samples and absolute signed-residual median <=1 dB. Failed
windows are excluded in full. It reconstructs the retained samples, quantiles,
height envelopes, angle-bin summaries and solar-window selection, then checks
those results against the frozen selected R4 tables before plotting.

Fig.3 uses 77,945 actual epochs from 1,354 quality windows. Fig.4 and illumination
use 1,342 windows, preserving their original additional complete-geometry condition.
Their difference is intentional; unavailable scan-angle geometry does not remove
an otherwise valid amplitude window from Fig.3. The comparison UTC partition
must not be substituted with the different source-origin event windows of Fig.2.

Ground remains 182 contributing stations,29,803,114 accepted native 1-Hz epochs and
498,430 accepted windows. Those windows represent 28,810,247 actual receiver-satellite epoch memberships; the remaining epoch samples do not meet the separate window eligibility rules. Its selection masks and all LEO processed inputs remain
unchanged. Their existing quartic/rolling-median methods are not replaced.
Geometry-only CDFs and speed/time histograms also keep their established source
population and bins. The context map retains the approved182 station identities
and the separately defined polar/scintillation/radar/observing-track layers.

## Files and computational boundary

- `inputs_r4/`: original processed LuGRE inputs for actual QC reconstruction.
- `code/recompute_r4.py`: portable numerical reconstruction and comparison.
- `fig34/amplitude_source/`, `fig34/response_source/`: selected R4 figure tables
  and unchanged Ground/LEO processed inputs.
- `illumination/`: package-local actual solar geometry and width inputs.
- `geometry/plot_data/`: exact geometry CDF and speed/time histograms.
- `context/`: frozen station/track tables, satellite illustrations and offline
  Natural Earth map resources.
- `reference/`: unchanged copies of the six selected formal R4 PNGs.
- `DATA_DICTIONARY_R4.json`: variables, units, cohorts and counting boundaries.
- `provenance/`: source checksums and original selection evidence. Assembly and
  historical upstream scripts here deliberately retain original paths and are
  not supported portable entrypoints.

The release reproduces these results from processed observations and frozen
near-Earth products. It is not a full raw multi-network acquisition or orbit/
geomagnetic processing rerun. No missing samples are filled, no residual is
refitted/recentered, and no new station or window is admitted during reproduction.
Numeric tables are verified; font-dependent pixel identity is not required.

All new exports are 600 dpi PNG. Arial is used when installed and DejaVu Sans is
available as a Matplotlib fallback; no proprietary fonts are redistributed.
Legacy `R3` input basenames are stable internal interface names, not a claim
that the selected scientific version is R3. Main outputs are `Fig3_R4.png` and
`Fig4_R4.png`; descriptive Extended Data names remain stable.

Provider credits and licences already supplied with the package remain in force.
This component does not invent additional licences for third-party observations,
mission images or map resources.

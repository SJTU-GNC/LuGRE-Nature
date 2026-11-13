# R4 lunar amplitude processing and supporting figures

Run from any folder with the declared Python requirements installed:

```bash
python components/main12/run_processing_r4.py --task all --output /path/to/output
```

Tasks are `fig2`, `e21`, `qc_background`, and `thresholds`. These are the current R4 tasks; main12/run.py forwards to these R4 tasks rather than silently invoking the older plotting workflow.

Fig.2 is rebuilt by applying point C/N0 >=25 dB-Hz, then at least 30 measured 1-Hz samples and absolute signed-residual median <=1dB within each frozen 60-s window. The current fixed cohort has 2196 valid windows and 254 events at W>=1.5dB: Equatorial 21/515, Mid-latitude 26/1001, Polar 207/680. Original map, event/PSD and paired-run cohort boundaries remain distinct and are preserved. Whole failing windows are rejected; percentile summaries do not remove original tail samples. PSD is recomputed on continuous actual 1-Hz runs, without inserted time samples or interpolated frequency values. Both polar maps use the strict accepted epoch identities.

The QC/background task recomputes the same three rules from native measurements and their verified archived E23 fits. It retains 9178 high-altitude windows, with pooled median W approximately 0.49dB. Per-signal and daily summaries are checked against the selected R4 values. This empirical background can contain receiver C/N0 estimation noise and other remaining effects; it is not a calibrated pure-noise measurement.

The threshold task keeps all 2196 valid windows fixed and tests 0.75, 1.0 and 1.5dB. These are 1.5, 2 and 3 times a nominal 0.5dB median background width, not sigma levels. It independently cross-checks pandas counts with plain CSV counting.

The E21 renderer uses all 6586 actual first-pass observations and 6542 existing qualified fourth-order residuals. Its geometry-only points, reference antenna patterns and polynomial display through occultation are identified separately; no gap observations are synthesized. The optional native refit command in the original component remains available and verifies the complete archived algorithm. Supply its output parquet with `--refit-results` to rebuild the QC/background figure from those newly recomputed fits.

## File roles

- `code/*r4*.py` and `run_processing_r4.py`: executable processing and figure code.
- `data/r4/inputs`: immutable pre-R4-QC measured samples, fixed window/block identities, and spatial metadata used as the upper-bound cohort. Their presence does not admit old QC results into R4 statistics.
- `data/r4/e21`: measured example, geometry-only tracks, fitted-arc coefficients and reference antenna pattern inputs.
- `data/r4/qc_background`: native source-segment identities needed to anchor background windows.
- `data/r4/expected`: small selected R4 numerical validation targets; these do not enter fitting or quality decisions.
- `data/full_lugre`: compact native data, verified geometry and frozen E23 validation targets already bundled by the native-refit route.
- `assets/cartopy`: offline Natural Earth coast/land data; no web download is required.
- Generated `outputs`: recomputed tables, ledgers, checks and 600-dpi PNGs. These are reproducible outputs, not immutable inputs.

No pixel-exact match across operating systems or font installations is asserted; the tested scientific counts, widths and spectra must agree. Source-orbit determination and mission-archive acquisition are outside this compact reanalysis package.

After the tasks complete, run `python components/main12/code/verify_processing_r4.py --output /path/to/output` to compare event windows, spatial quantiles, spectra and paired widths against the selected R4 numerical targets.

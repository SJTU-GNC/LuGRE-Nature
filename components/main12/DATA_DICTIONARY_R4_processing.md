# R4 lunar-amplitude processing fields

All coordinates and residuals retain their selected scientific definitions. The existing component dictionary covers the shared orbit, geometry and antenna fields.

| Field or table | Meaning | Units / scope |
|---|---|---|
| `phase`, `op_id`, `sat`, `signal_id`, `epoch_key` | Immutable source-observation identity | `S` denotes lunar-surface operation; epoch key is rounded GPS time in milliseconds |
| `rx_gps_seconds`, `time_utc` | Actual observation time | GPS seconds and UTC timestamp |
| `cn0_dbhz` | Receiver-reported carrier-to-noise density ratio | dB-Hz |
| `quartic_background_dbhz` | Archived reconstructed-arc fourth-order polynomial background | dB-Hz |
| `e23_residual_db` | Measured C/N0 minus its fitted background | dB, signed |
| `fit_eligible`, `fit_degree` | Admission to the archived fourth-order residual population | Degree-0 fallbacks are retained as provenance but do not enter amplitude statistics |
| `segment_id` / `native_segment_context.parquet` | Original tracking-segment identity used to anchor source windows | Joins to the compact native input by the immutable observation key |
| `window_origin_gps_s`, `window_idx` | Frozen source-window origin and index | 60-s half-open windows, preserving each scientific cohort's original boundaries |
| `point_three_condition_eligible` / `point_quality_eligible` | Qualified finite fourth-order residual with native C/N0 >=25 dB-Hz | Boolean observation gate |
| `n_points`, `n_valid`, `n_point_quality_eligible` | Actual retained 1-Hz observation count before the whole-window offset check | At least 30 for a valid window |
| `signed_p05_db`, `signed_median_db`, `signed_p95_db` / `pre_bias_*` | Quantiles of signed residuals after point screening | dB; the median's absolute value must be <=1 dB |
| `signed_p95_p05_width_db`, `W_db`, `pre_bias_W` | Central90% residual width, P95(delta)-P05(delta) | dB per valid 60-s window |
| `qc_keep`, `window_qc_keep`, `analysis_qc_keep` | Window or observation inclusion after the declared three conditions | Entire offset-failed or undersampled windows are excluded |
| `lat_group` | Existing per-observation or assigned-window geomagnetic latitude band | Equatorial, Mid-latitude, Polar; original cohort assignments preserved |
| `metric` in spatial cells | P95 of absolute accepted residuals pooled within a 5degree x 5degree tangent-point cell | dB over the observing campaign |
| `median_psd_dbhz2_per_hz` | Median native-bin Welch power across qualified continuous 1-Hz runs | dB^2/Hz; no time/frequency interpolation |
| Paired block widths | Central90% widths at both frequencies using the same accepted actual seconds within each frozen complete60-s block | dB; track display uses the median of accepted block widths |
| High-altitude window tables | Same three-condition screening after selecting tangent height >1000 km from the lunar-surface campaign | 9178 valid windows; pooled median W approximately0.49 dB |
| Threshold sensitivity tables | Counts obtained by applying W >=0.75,1.0,1.5 dB to the same2196 valid windows | Nominal background-width multipliers1.5,2,3; all latitude denominators remain fixed |

`P05`/`P95` are quantile summaries. They define widths or summary magnitudes; the data retain the original tail samples unless an independent point/window quality condition rejects them. The high-altitude background is empirical and may include receiver estimation noise and other residual effects.

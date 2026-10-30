# Source data dictionary

CSV files are UTF-8, use a header row and represent missing numerical values by
an empty field or NaN. Parquet preserves native numeric/Boolean/string types.
UTC timestamps retain the meaning of their named UTC/time columns. Distances
are kilometres unless the field explicitly specifies otherwise; C/N0 is dB-Hz,
and a C/N0 residual or percentile width is dB.

## Figure 2

- `updated_points.csv`: measured observation identities, mission phase, OP,
  satellite and signal, UTC/GPS time, receiver-reported C/N0, verified WGS84
  tangent height/latitude/longitude, ROTI and corrected E23 h50 backgrounds and
  residuals. `legacy_...` fields retain older residuals for provenance and are
  never substituted for current E23 residuals. `fit_eligible`, `fit_status`,
  `fit_degree` and `e23_local_arc_id` describe actual fit eligibility.
- `spatial_amplitude_cells.csv` and `spatial_roti_cells.csv`: populated map cells
  and their plotted amplitude/ROTI statistics. These summarize actual measured
  samples; unpopulated map cells are not inferred measurements.
- `fig3_panel_a_psd_no_time_interpolation.csv`: selected spectra; frequency is Hz
  and power spectral density is the source-recorded dB²/Hz. The historical
  filename reflects its earlier figure location, not the current figure number.
- `psd_retained_runs.csv`, `psd_excluded_runs.csv` and
  `psd_per_run_frequency_bins.csv`: spectral segment selection and intermediate
  frequency bins, retaining actual continuity without interpolating C/N0.
- `all_60s_windows.csv`: each accepted measured 60-s window and its signed
  P95−P05 width (`signed_p95_p05_width_db`); `is_event` denotes width>=1.5 dB,
  while `n_points` denotes actual valid measured seconds.
- `fig3_panel_b_event_level_points.csv`: accepted event windows used in Fig.2;
  historical filename retained to preserve exact source identity.
- `fig3_panel_d_altitude_layer_heatmap.csv`: measured-window denominator,
  event numerator and occurrence by ionospheric height layer and latitude band.
- `panel_c_paired_frequency_segments.csv` / `paired_geometry.csv`: matched
  paired-frequency segments and their actual geometric/latitude metadata.
  `high_width_60sample_block_median_db` and
  `low_width_60sample_block_median_db` are median widths of accepted complete
  paired 60-s blocks, not every point from unmatched signal tracks.
- `OP76_G01_measured_samples.csv`, `OP38_E21_measured_samples.csv`: actual case
  measurements used for the adjacent C/N0, height and ROTI panels.
- `selected_track_summary.csv`, `selected_track_geometry_QA.csv`: selected case
  support and WGS84 geometry checks. Display labels are plot identifiers, not
  independent physical-event counts.

## E21 detrending example

- `E21_plotted_observations_R3.csv`: actual measured C/N0 and measured-epoch
  geometry, original sample identities and earlier fitting fields.
- `E21_independent_WGS84_geometry.csv`: independently calculated geometric path
  through the Earth-occultation gap. These are geometry samples, not invented
  signal observations.
- `E21_measured_CNo_joint_quartic_candidate.csv`: frozen expected h50 fits and
  residuals, which the rendering script recomputes from measured C/N0.
- `E21_quartic_model_evaluation_not_measurements.csv`: evaluated model curve
  across a missing-data interval; it is expressly not measurement data.
- `E23_archived_validation.csv`: archived E23 input/example output used to check
  that the same archived robust quartic implementation is being applied.

## Space weather

- `unchanged_space_weather_context.csv`: frozen hourly plot grid. Columns `f107`
  (sfu), `bz` (nT), `kp` (dimensionless Kp), `pcn` (mV/m) and `ae`, `au`, `al`
  (nT) produce panels a–e. Additional saved speed/pressure context is retained
  when present. This is a plotting product, not the native archive cadence.
- `all_eligible_60s_windows_h50.csv`: same accepted 2,244 LuGRE windows as the
  corrected Fig.2 analysis, with 693 polar windows and 266 event windows.
- `op_mean_all_latitude_and_polar_h50.csv`: OP timing and expected arithmetic
  means of within-window widths; statistics are independently regenerated from
  the window file. A missing polar OP statistic remains missing.

## Antenna geometry

- `lugre_gnss_main_lobe_point_check.csv`: point-level LuGRE geometry cache.
  `signal_name`, `h_tan_km`, `tx_off_boresight_deg` identify the plotted four
  signals, tangent-height selection and transmitter angle in degrees.
  This inherited antenna cache spans 15 January–16 March 2025 across mission
  phases, separately from the main lunar-surface amplitude population.
- `lugre_full_geometry_noise_windows_refined.csv`: observation windows used to
  identify actual GPS spacecraft represented in the public pattern collection.
- `GPS_antenna_patterns/`: NAVCEN reference pattern files and PRN-to-SVN metadata.
- `GRAP_metadata/`: JRC Galileo E1/E5a population-model workbooks.
- `derived_data/`: inherited pattern envelopes, first-valley summaries and source
  catalogues; each original CSV header retains its scientific quantity name.
- Recomputed envelopes are relative EIRP (dB) by off-boresight angle (degrees).
  The point-distribution quantiles and pattern percentiles have different
  denominators; they are not confidence intervals or official fixed boundaries.

## Optional full LuGRE refit

`native_cn0_verified_geometry_and_expected_results.parquet` contains all 993,093
positive native-context observations. Keys are `phase`, `op_id`, `sat`,
`signal_id`, `epoch_key`; `rx_gps_seconds` is measured GPS time; `cn0_dbhz` is
native C/N0; `lugre_*_wgs84` / `lugre_h_tan_wgs84_km` contain verified tangent
coordinates. Columns prefixed `expected_` are frozen validation outputs, never
algorithm inputs. The optional command compares eligibility, arc identities,
backgrounds and residuals with those values before reporting success.

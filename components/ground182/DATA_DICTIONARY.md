# Numerical source-data dictionary

All files retain the original field names so plotted values can be traced to
the frozen source manifests. CSV files use UTF-8; compressed arrays use NumPy
NPZ; typed tables use Apache Parquet. Blank numeric cells/NaN denote unavailable
values, not zero. Boolean eligibility flags are explicitly retained.

## Shared quantities

| Field | Meaning | Unit |
|---|---|---|
| `platform` / `system` | Ground, LEO-RO, LuGRE observing platform | category |
| `source_unit` | Receiver identity for Ground; original source identity for other platforms | identifier |
| `satellite`, `signal` | Original GNSS satellite and carrier selection (GPS L1/Galileo E1) | identifier |
| `latitude_band` | Equatorial, Mid-latitude, Polar, from the observation's ionospheric reference point | category |
| `fit_arc_id`, `qc_segment_id`, `continuous_run_id` | Declared source-fit/quality/continuity identities | identifier |
| `utc_minute_start_s`, `first_utc_s`, `last_utc_s` | UTC times expressed as seconds since the Unix epoch | s |
| `n_points` | Number of retained actual 1-Hz measurements in a response window (30–60) | count |
| `signed_p05_db`, `signed_median_db`, `signed_p95_db` | Quantiles of signed detrended receiver C/N0 in the minute window | dB |
| `width_db` | W = P95(delta C/N0) − P05(delta C/N0) in that window | dB |
| `ray_B_acute_deg` | Acute ray direction–local magnetic field angle | degrees |
| `scan_Bperp_acute_deg` | Acute transverse scan direction–projected magnetic field angle | degrees |
| `magnetic_meridian_acute_deg` | Acute ray azimuth to the local magnetic meridian | degrees |
| `point_speed_mps` | Speed of the ionospheric reference point | m/s |
| `scan_speed_mps` | Transverse geometric scan speed | m/s |
| `r1_L1_km`, `r1_L1E1_km` | First Fresnel-zone radius for the reference carrier | km |
| `fresnel_crossing_time_s` | Geometric first-Fresnel-radius crossing time; excludes unknown plasma drift | s |
| `height_reference_km` | Ground IPP shell altitude or near-space tangent height | km |

Ground latitude classification uses the per-epoch 350-km IPP. Near-space
observations use tangent points in the 50–1,000-km shell. The contributing
station counts per IPP band overlap: Equatorial64, Mid123, Polar33. These are
not disjoint station groups. Fig.3 contains29,803,114 accepted Ground epochs;
the498,430 qualified windows in Fig.4 and the illumination comparison represent
28,810,247 of those actual epochs. The difference is the declared window
eligibility/continuity requirement, not synthetic resampling.

## Fig.3 inputs

`histogram_source_data.csv` contains signed residual bins (`bin_left_db`,
`bin_right_db`, `bin_center_db`), actual `count`, group `n_total` and
`percent_of_all_group_records`. `quantiles.csv` contains pre-display P50/P95/P99
absolute-residual quantiles, in dB, and group counts. These are epoch-level
statistics, not within-minute widths.

`height_envelopes.csv` contains32-km-bin counts and height centres in km;
`p95_abs` and `p95_abs_smoothed` are the source and display-median P95 absolute
residual envelopes in dB. Older P90 fields remain as provenance and are not
plotted. The two height NPZ files include `amplitude` (dB), `height_km`,
`band_code` (0Eq/1Mid/2Polar) and explicit `all_valid` membership.

## Geometry distribution inputs

`geometry_angle_CDF_curves_R3.csv` contains `metric`, `angle_deg` and
`cumulative_fraction` in[0,1]. The speed/time histogram table contains
`bin_left`, `bin_right`, `weighted_count` and `weighted_proportion_percent`.
Units are m/s for point/scan speed and s for Fresnel crossing time. Weights are
one per actual retained native epoch. Ground inset histograms retain their
separate display scales. The freeze file records the exact selection binding
and source counts.

## Illumination inputs

The Ground/near-space solar Parquet files contain accepted response windows
plus observed-epoch solar geometry. `plot_solar_local_time_h` is local solar
time in hours; `plot_sza_deg` is solar zenith angle in degrees;
`plot_latitude_deg` and `representative_mlat_centered_dipole_deg` are geographic
and centred-dipole magnetic latitude in degrees; `plot_height_km` is the
reference height in km. `sunlit_fraction` is the actual fraction of retained
native measurements whose Sun-directed ray avoids the solid WGS84 Earth;
`representative_sunlit` is its explicit representative-epoch flag.

## Observing context inputs

`Ground_final_contributors_R3.csv` uses one full station ID and unchanged
geodetic `lat`/`lon` in degrees per accepted station. Its historical epoch-count
columns belong to the original coordinate-table preparation and are not used
to define current scientific counts. `context_current_binding.json` is produced
at rendering to bind the same182 coordinates to the current selection.

The polar GNSS, scintillation, radar, mission-orbit and lunar/LEO track files
are separate contextual layers. Counts cannot be summed across categories.
Lunar context tracks are selected by the actual lunar-surface phase `S`:
461 satellite-signal tracks,38 satellites,139,809 source points on the five
research days. This context population differs from the main amplitude cohort.

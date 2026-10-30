# Principal numerical inputs and units

| File/column | Meaning | Unit |
|---|---|---|
| `Fig5_LuGRE_h50_current.csv`: `_utc_s` | Actual measured integer UTC second used for source matching | Unix second |
| `hemisphere`, `signal_id`, `label` | Exact case/signal membership, northern OP76 G01 or southern OP38 E21 | identifier |
| `cn0_dbhz` | Receiver-reported C/N0 | dB-Hz |
| `e23_residual_db` | Current h50 E23-method robust fourth-order residual plotted in Fig.5 d/f | dB |
| `lugre_h_tan_km` | WGS84 tangent height | km |
| `fit_degree`, `fit_eligible` | Declared background degree and eligibility; all plotted current case samples have degree4 | integer/boolean |
| `Fig5_southern_FIR_current.csv`: `beam`, `range_gate` | Accepted radar cell geometry | identifier |
| `mean_abs_velocity_ms` | Mean of absolute line-of-sight velocities, not absolute mean velocity | m/s |
| `mean_power_db` | Power of accepted echoes; retained separately from velocity | dB |
| `panel_ready/Fig5`: ROTI fields | Regional GNSS variability context | TECU/min |
| `panel_ready/Fig5`: ground `S4`/`s4` | Ground high-rate amplitude scintillation context | dimensionless |
| `common/leoro_rolling_points.csv`: `delta_rolling_db` | Accepted 1-Hz LEO-RO rolling-median amplitude residual | dB |
| `common/leoro_highrate_points.csv`: `delta_highrate_db` | High-rate LEO amplitude context | dB |
| `restricted_yunyao/*/ionprf_position_time_ne_profile.csv`: `altitude_km` | Retrieved profile altitude | km |
| `electron_density_m3`, `electron_density_cm3` | The same profile density in SI and conventional units | m^-3, cm^-3 |
| `latitude_deg`, `longitude_deg`, `event_time_utc` | Profile location/time supplied with those two observations | degree, UTC |
| `data/joint/lugre_paired_minutes_selection.csv`: `cn0_p95_abs_delta_db` | Minute P95 of absolute current h50 LuGRE residual | dB |
| `roti_1min_independent` | Paired phase-derived ROTI | TECU/min |
| `lat_group`, `selected` | Historical contextual latitude grouping and frozen selection; distinct from Fig.3 epoch-centred-dipole grouping | category/boolean |
| `leo_profiles_selected_P95.csv.gz`: `s4_p95` | Profile P95 of accepted high-rate-derived 10-s S4 values | dimensionless |
| `sigma_phi_rad_p95` | Profile P95 of accepted 10-s phase scintillation indices | rad |
| `n_native_windows` | Number of retained 10-s windows used for that profile | count |
| `LEO_P95_selected_10s_window_membership.csv.gz` | Exact 64,988 native-window identities and index values for independent P95 verification | source-specific |
| `ground_station_minutes_selected.csv.gz`: `s4`, `sigma_phi` | Ground high-rate index context summarized for station-minute/band | dimensionless, rad |

Empty CSV fields mean unavailable values, not zero. Original metadata/path
columns are retained for provenance. Only the current residual fields named
above are used by the runtime; historical residual columns do not replace the
selected h50 values. `provenance/` describes the preserved cohort and definitions.

All source C/N0 residuals and large values remain in the input tables. Axis
limits are display limits. The joint LuGRE plot shows ROTI 0–60 and amplitude
0–6 dB while retaining all 920 selected pairs in its complete numerical table.

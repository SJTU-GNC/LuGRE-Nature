# Figure 5 and the joint-observable Extended Data figure

This GitHub component contains the **current R3 numerical plotting code** and
its provenance metadata. Public numerical inputs, selected analysis outputs,
and permitted reference targets are distributed through the associated Zenodo
data version. It is part of the 182-station R3 release, although these two
figures use their own frozen case-study/high-rate comparison populations.
The 182-station native 1-Hz population must not be substituted for the separate
ground scintillation reference dataset here.

## Run on another computer

Use Python 3.12 with NumPy, pandas, Matplotlib, SciPy and Pillow installed. No
Cartopy installation or coastline download is needed. Paths are resolved from
this component, so the folder can be moved to a different drive or operating
system. From this folder:

```text
python verify_inputs.py
python run.py --figure all --output-dir results
```

The historical `fig5_sources/south_source.py` also retains an optional
`netCDF4.Dataset` import inside a raw-conPhs fallback function. The supported
entry point uses the included frozen 1-Hz rolling-median tables, so this
fallback is never invoked. Reproducing the supported two figures does not
require netCDF4 or the original raw conPhs archives. The historical fallback
is retained as source provenance, not advertised as a complete raw-data route.

Individual figures:

```text
python run.py --figure Fig5 --output-dir results
python run.py --figure ED8 --output-dir results
```

Both figures are redrawn from numerical inputs. **The official Figure 5 and
Extended Data PNGs were assembled or updated through historical composite
workflows. These runnable full-figure redraws preserve the scientific values
but are not claimed to be pixel-exact reproductions of those composites.**
The files in `reference/` are QA targets; the runtime never reads them as the
scientific artwork. It generates all plotted measurements, maps, profiles and
radar cells from the included data. Font layout and some decorative placement
can differ from the approved manuscript PNG. The generated figures use the
Matplotlib-bundled DejaVu Sans font to avoid a dependency on Arial availability.

## Figure 5

- `data/Fig5_LuGRE_h50_current.csv`: the exact 591 northern and 930 southern
  selected epochs, using the current h>=50 km E23-method fourth-order residuals.
  `e23_residual_db` is the plotted dB residual; `lugre_h_tan_km` is WGS84 tangent
  height in km. Historical residual columns in this provenance-rich table are
  retained but are not plotted.
- `data/panel_ready/Fig5/`: frozen track points, polar ROTI and ground S4 fields,
  SuperDARN power/cell geometry, accepted 1-Hz rolling-median LEO profiles,
  high-rate LEO amplitude context, PFISR electron-density profiles, and
  case-identifying JSON. These are selected scientific inputs, not complete raw
  network archives.
- `data/Fig5_southern_FIR_current.csv`: corrected mean(abs(v_LOS)) in m/s. Power
  maps retain their measured power; the fan alone uses the velocity aggregate.
  The included accepted-echo table and provenance proof document the correction.
- `data/restricted_yunyao/`: the code expects the two authorised profiles used
  in Figure 5, Y038/E05 (2025-03-15 16:00 UTC) and Y040/C19 (2025-03-03 12:17
  UTC). These profiles, extracted numerical tables, matched metadata, and the
  Fig.5 PNGs that display them are **not included in the public GitHub or Zenodo
  R3 updates** because provider permission for public redistribution has not
  been documented.

The original numerical panel renderers are under `code/fig5_sources/`; their
component roots were localized, obsolete duplicate Ne inputs were removed, and
`runtime_fig5.py` selects current LuGRE residuals and corrected southern fan
values. The preserved source functions define the map projections, geographic
reference overlays, geometric height/time axes, LEO comparison and Ne panels.

## Extended Data joint observables

- `data/joint/lugre_paired_minutes_selection.csv`: 920 selected LuGRE amplitude
  P95/ROTI pairs; the figure display range shows 916. The complete source table
  retains the four points above ROTI=60 TECU/min.
- `data/joint/leo_profiles_selected_P95.csv.gz`: 1,619 accepted profile pairs,
  with actual P95 S4 and sigma_phi computed from 64,988 accepted 10-s windows.
- `LEO_P95_selected_10s_window_membership.csv.gz`: those exact windows, allowing
  `verify_inputs.py` to recompute the profile quantiles independently.
- `ground_station_minutes_selected.csv.gz`: ground station-minute high-rate
  scintillation context, distinct from the native 1-Hz Ground182 amplitude data.
- Additional tables retain window membership, UTC support, summary statistics,
  original/source-specific latitude grouping and binned-density counts.

LuGRE amplitude is a 1-Hz C/N0 fluctuation proxy paired with ROTI. The LEO and
ground observables are high-rate scintillation indices. The source-specific
latitude definitions and historical UTC selection in this contextual figure
are intentionally preserved; they are not silently replaced by the main
Figure 3/4 per-epoch population. See `provenance/joint_h50_provenance_R3.json`.

## Data-use conditions

Read `THIRD_PARTY_ACCESS.md` before assembling any non-public copy of this
component. The YunYao measurements were supplied on request and require the
provider's approval for public redistribution. They are excluded from the
public R3 updates. Third-party data retain their original rights and attribution
requirements; the authors' code licence does not extend to these data. Natural
Earth geography is public-domain map data.

Original machine paths in `provenance/` document the source chain only; they are
not runtime dependencies. `_packaging/` is an original-machine assembly aid and
is excluded from the distributable payload by the root release builder.

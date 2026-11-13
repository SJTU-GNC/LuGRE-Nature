# R4 ground182 data dictionary

`DATA_DICTIONARY_R4.json` is the machine-readable definition of cohorts, units,
QC and computational boundaries. The six figures use different statistical units:

| Table or variable | Unit and meaning |
|---|---|
| Original LuGRE epochs | 82,761 actual processed1Hz measurements,with original independent quartic residuals,UTC,geometry and per-epoch magnetic bands |
| Original candidate windows | 1,654 half-open UTC-minute/continuous-run/source-fit-QC-segment/band groups |
| R4 amplitude selection | 77,945 epochs in1,354 quality windows; CN0>=25dB-Hz,Nvalid>=30,absolute signed residual median<=1dB |
| R4 geometry/LT windows | 1,342 of those quality windows,retaining the original complete-geometry condition |
| Ground selection | 182 stations,29,803,114 accepted native1Hz epochs,498,430 accepted response windows; original selection masks preserved |
| `delta_cn0_db` | dB; signed1Hz measurement minus original qualified background |
| `width_db` | dB; P95-P05 of signed1Hz residuals within each accepted minute window |
| `P95_abs_db` | dB; platform/band95th percentile of absolute1Hz residuals,not the minute width |
| Geometry angles | degrees,acute0-90; local IGRF14 field at the IPP/tangent point |
| Reference/scan speeds | m/s; geometric motion,without plasma drift |
| Fresnel crossing time | s; geometric first-zone radius divided by scan speed |
| Solar local time/SZA | hours/degrees at the actual representative epoch; Ground350km IPP or near-space tangent point |
| Geometry Extended Data | Frozen geometry-only populations and bins; not replaced with the reduced amplitude-QC population |
| Observing context | Frozen station,radar and observing-track layers; categories overlap and are not all native1Hz amplitude contributors |

Original processed inputs are in`inputs_r4/`; selected plotting tables are in
`fig34/`, `illumination/`, and `geometry/plot_data/`. Run`python reproduce.py recompute`
to recreate the R4 LuGRE selection and summaries,and`python validate_inputs.py`
to verify their numerical agreement with the selected reference tables.

No raw-data acquisition,orbit reconstruction,new detrending,synthetic gap values,
new stations or new scientific windows are introduced. Original-source paths in
`provenance/` record ancestry; supported runtime entrypoints use only local inputs.

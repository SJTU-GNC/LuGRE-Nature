# R4 variables and scientific boundaries

The release contains several deliberately different cohorts. Do not concatenate tables just because their columns share names. Each component dictionary gives its exact schema and counting unit:

- `components/main12/DATA_DICTIONARY_R4_processing.md`, `README_R4_processing.md` and `R4_MAIN12_PROCESSING.json`: native lunar input, Fig.2 satellite-signal windows, paired blocks, E21 example and high-altitude background. `components/main12/DATA_DICTIONARY.md` also describes inherited source formats; older event totals there belong to the historical baseline and are superseded by the R4 processing metadata.
- `components/ground182/DATA_DICTIONARY_R4.json`: Fig.3/4 comparison epochs, UTC windows, geometry and solar metadata. `DATA_DICTIONARY.md` supplies inherited Ground/LEO source-table formats.
- `components/fig5joint/DATA_DICTIONARY.md`: separate polar cases and joint-observable populations; see the R4 public-access exception in that component README.

## Shared quantities

| Quantity | Meaning and unit |
|---|---|
| C/N0 | Receiver-reported carrier-to-noise density ratio, dB-Hz |
| delta C/N0 | Signed observation minus its declared background model, dB |
| abs(delta C/N0) | Absolute magnitude of that signed fluctuation, dB |
| W | P95(delta C/N0) minus P05(delta C/N0) within an eligible60-s window, dB |
| P95(abs(delta C/N0)) | Upper magnitude summary; not the same quantity as W |
| valid count | Number of actual eligible1-Hz observations; no interpolated gap measurements |
| tangent height | WGS84 height at the ray's Earth-centred closest-approach point, km |
| IPP | Ground-link intersection with the declared350-km reference shell |
| magnetic-latitude band | Absolute centred-dipole latitude: <20deg equatorial,20 to<60deg mid-latitude,>=60deg polar; per-reference-point classification |
| ray/scan/meridian angles | Geometric angles in degrees; vector/projection and folding conventions are retained in source methods and component tables |
| geometric speed | m/s; does not add an unmeasured plasma drift velocity |
| solar geometry | Solar local time in hours, SZA in degrees; reference position and elevated-point Earth-shadow method are source-specific |

Fig.2 source-origin event windows and the UTC-partitioned comparison windows have different identities and denominators. Source epoch times retain their original GPS/UTC column definitions; do not infer a time system from a bare numeric field. Fig.4/illumination impose complete geometry beyond amplitude eligibility. The geometry-only distributions and observing-context layers have their own populations. The context map includes separate polar30-s, scintillation50-Hz and native1-Hz station layers; their counts do not sum to the182-station amplitude cohort.

Fitting, QC and validation columns are distinguished in code. `expected_*` fields are frozen targets used to check a recomputation and never enter its fit. Failed-window points remain traceable in ledgers but do not enter accepted statistics. Quantile endpoints characterize the retained distribution without deleting its tails. Native/processed input formats are CSV, Parquet, JSON and standard source files; dependencies for these formats are locked in `requirements.txt`.

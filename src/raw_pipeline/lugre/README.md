# LuGRE S-phase raw-to-observation reconstruction

This directory contains a package-relative reconstruction of the S-phase
observation table used by Extended Data Figures 6--9.

Run the complete seven-operation reconstruction from the package root:

```powershell
.\runtime\python_exact\python.exe .\src\raw_pipeline\lugre\build_observations_s_phase.py
```

For a quick smoke test:

```powershell
.\runtime\python_exact\python.exe .\src\raw_pipeline\lugre\build_observations_s_phase.py `
  --ops 38_0 `
  --output-dir .\work\derived\lugre_raw_smoke_op38
```

The default primary result is:

```text
work/derived/lugre_raw/observations_s_phase_from_raw.csv
```

The same directory also receives:

- `raw_input_manifest.csv`: packaged raw inputs, sizes, hashes, and parsed rows;
- `column_comparison.csv`: column-by-column comparison with the accepted table;
- `field_provenance.csv`: counts split by raw reconstruction and fallback;
- `stec_mismatch_details.csv`, `rot_mismatch_details.csv`, and
  `roti_mismatch_details.csv`: every candidate/reference disagreement;
- `qa_summary.json`: machine-readable run status, counts, hashes, and tolerances.
- `validation_report.md`: reviewer-facing assessment and remaining blockers.

## What is reconstructed from raw data

The seven packaged `TLM_RAW_*_S_OP*.txt.gz` files are lossless gzip containers
whose decompressed raw text determines:

- operation, satellite, signal, receiver GPS time, and C/N0;
- UTC second using the 18-second GPS--UTC offset applicable in March 2025;
- continuity segments, with a new segment after a gap greater than 10 seconds;
- dual-frequency STEC from pseudorange and accumulated Doppler, using a
  per-pair-arc median code-minus-carrier bias;
- ROT from consecutive STEC samples;
- ROTI as the inclusive trailing 300-second sample standard deviation of ROT.

Raw-derived STEC/ROT/ROTI candidates are retained only when they reconcile
with the accepted table within the tolerances recorded in `qa_summary.json`.

## Explicit processed fallbacks

This is a maximally raw-backed reconstruction, but it is not yet a completely
independent end-to-end scientific recomputation. The following values remain
traceable processed fallbacks:

1. `plot_cn0_detrended_db`: the accepted workflow names the method
   `robust_poly4_median60s`, but the exact fitting implementation and all
   boundary rules are absent.
2. `lugre_lat`, `lugre_lon`, and `lugre_h_tan_km`: precise SP3 products and
   spacecraft state vectors are packaged, but the exact interpolation,
   frame/time conversion, and tangent-point generator are absent.
3. A small number of STEC/ROT/ROTI rows cross carrier-continuity boundaries
   that are not encoded in the available raw files or scripts. Those rows are
   listed individually in the mismatch detail files.

The accepted processed table is therefore an input to this QA reconstruction,
not a hidden substitute. A fully independent chain requires the missing
detrending implementation and exact precise-orbit geometry/continuity logic.

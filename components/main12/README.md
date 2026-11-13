# R4 lunar observations and supporting figures

Use the package-level `run_all.py` for the selected R4 results. This component has two current public entrypoints:

```text
python components/main12/run_processing_r4.py --task all --output outputs/main12
python components/main12/run_other_r4.py --task all --output outputs/main12
```

The first recalculates current Fig.2 QC, maps, spectra, paired-frequency widths, the full E21 example, QC/background figure and threshold sensitivity. See `README_R4_processing.md` and `R4_MAIN12_PROCESSING.json` for inputs and exact counts. The second verifies Fig.1 theory and exports its authored reference, and redraws the current space-weather and antenna figures. See `R4_OTHER_TASKS_README.md`.

The selected event cohort is2196 valid windows and254 events; the space-weather figure intentionally retains its separate2244-window context cohort. Legacy `data/fig2` contains immutable upper-bound/frozen inputs used to reconstruct current QC, not permission to reuse old event totals. Supported R4 outputs and figure mappings are registered in `component.json` and the root `RESULTS_INDEX.csv`.

`code/refit_lugre_h50.py --output outputs/full_lugre_refit` independently reruns the archived fit on all993093 compact native observations and verified geometry. Expected columns are verification targets, never fitting inputs. Original algorithm, source identities and platform-specific processing remain unchanged. This package does not redetermine satellite orbits or download raw mission archives.

`run.py` forwards to the same current R4 workflows for compatibility. `provenance/legacy_run_R3.py` and `provenance/original_code` preserve historical R3 implementations and are not current entrypoints. Inherited source-format definitions are in `DATA_DICTIONARY.md`; current cohort/quality definitions are in the R4 processing documentation and root dictionary.

Numerical redraws and authored reference copies are explicitly distinguished. Do not infer new third-party data/artwork licences from their inclusion; root and component source notices apply. No proprietary fonts are redistributed.

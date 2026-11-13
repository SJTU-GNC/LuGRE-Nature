# LuGRE R4 data, results and reproducible analysis

Prepared 13 November 2026 for the existing data record **10.5281/zenodo.21672260** and code record **10.5281/zenodo.21675357**. This local release does not assign a new version DOI or publish either record.

## Start here

Extract **both** `LuGRE_R4_Data_20261113_References_checked.zip` and `LuGRE_R4_Code_20261113_References_checked.zip` into the same parent directory. They populate one folder, `LuGRE_R4_release/`. Shared documentation is identical; the archives contain complementary scientific inputs and software.

On 64-bit Windows with Python 3.12, run `setup_windows.cmd`, then:

```text
.venv\Scripts\python.exe verify_package.py
.venv\Scripts\python.exe run_all.py --list
.venv\Scripts\python.exe run_all.py
```

Alternatively install `requirements.txt` in your Python 3.12 environment and use `python` for these commands. The bundled wheels target Windows x64/Python 3.12; other platforms need corresponding dependency builds and have not been execution-tested. Scientific plotting does not download data. Results and logs go to `outputs/`; immutable inputs are preserved.

## What this version contains

- The **5 main and 9 Extended Data figures** selected in the R4 manuscript, copied unchanged into `official_figures/` and bound by SHA-256. Full-quality English/Chinese PDFs and their sources are in `manuscript_snapshot/`.
- Processed observations, source tables, background/QC examples, geometry and station metadata, plotting resources, and a figure-to-input/task index in `RESULTS_INDEX.csv`.
- Current R4 lunar QC, spectra, paired-frequency widths and threshold-sensitivity calculations. An optional native-C/N0 refit independently verifies the archived LuGRE fourth-order polynomial algorithm.
- The selected **182-station** Ground results, accepted LEO products and their source/selection provenance. The full29.8-million-epoch Ground residual matrix and raw multi-network GNSS archives are not duplicated: Ground distributions start from frozen histogram/quantile source tables, while all retained comparison-window inputs are supplied. Mission telemetry acquisition, orbit determination and full upstream geometry production are outside this compact reproduction boundary; access routes and historical processing scripts are provided.

The current Fig.2 event cohort contains **2,196 valid one-minute satellite-signal windows**: 21/515 equatorial, 26/1,001 mid-latitude and 207/680 polar windows exceed W=1.5 dB. Simultaneous signals and tracks retain their own source identities. The comparison cohort for Fig.3/4 is intentionally different: Fig.3 has 77,945 LuGRE epochs from 1,354 QC windows; Fig.4/illumination use 1,342 windows with complete geometry. Ground retains 29,803,114 accepted 1-Hz epochs and 498,430 windows from 182 contributing stations. The space-weather figure uses its separately frozen context cohort; do not substitute its counts for the event denominator.

For lunar amplitude statistics, C/N0 below25 dB-Hz is removed, then60-s windows require at least30 actual valid1-Hz samples and an absolute signed-residual median no greater than1 dB. Entire failing windows are excluded. P95 and P05 summarize the accepted distribution; they are not an additional deletion of tail observations. W=P95(delta C/N0)-P05(delta C/N0), while P95(abs(delta C/N0)) is a different summary. The0.75/1.0/1.5-dB sensitivity tests keep the same cohort. A nominal0.5-dB empirical background width is not a Gaussian sigma.

Ground long-arc fourth-order fits, LuGRE archived E23 fourth-order fits and LEO rolling-median detrending remain distinct methods. No amplitude-based station selection or platform-wide detrending replacement is introduced by packaging.

## Reproduction levels and access boundary

Supported commands are registered in `components/*/component.json`. Use `run_all.py`, `main12/run_processing_r4.py`, `main12/run_other_r4.py`, `ground182/reproduce.py` and `fig5joint/run_R4.py`. Legacy R3 basenames and scripts in provenance preserve source history; they are not the current entrypoints.

Most tasks recalculate numerical summaries and redraw panels from packaged measurements or processed source inputs. Geometry distributions and some context products start from frozen source tables rather than original orbit/field calculations. Fig.1 is authored artwork: its theory curves are independently checked and its approved PNG is explicitly copied. Antenna artwork and historical montage layout are distinguished from numerical panels. Font-dependent pixel identity is not claimed. The selected paper figures remain unchanged regardless of redraw differences.

**YunYao exception:** the public package excludes the two provider-restricted original electron-density profiles and their extracted numerical profiles. The complete approved Fig.5 remains a reference result, but default numerical reproduction writes `Fig5_R4_public_partial.png`, marking the unavailable panels. A full numerical redraw requires separately obtained authorized inputs. ED9 and the remaining public calculations are unaffected. See `THIRD_PARTY_ACCESS.md`.

For the optional full LuGRE refit:

```text
python components/main12/code/refit_lugre_h50.py --output outputs/full_lugre_refit
```

Expected columns in the native compact input are validation targets and never enter fitting. The source contains positive native observations and verified geometry; synthetic gap measurements are not used.

## Files and provenance

`FILE_MANIFEST.csv` and `SHA256SUMS.txt` verify the merged static release. `provenance/MANUSCRIPT_FIGURE_SNAPSHOT.json` identifies the exact R4 figures and manuscripts. Component dictionaries describe variables, units, cohorts and source boundaries. `validation/` contains the executed numerical/portability checks; historical R3 metadata is labelled by its original revision and is not a substitute for these R4 checks.

Author code retains MIT licensing. Author-owned derived source data retain CC BY4.0. Third-party observations, maps, antenna references, artwork, templates and dependency wheels retain their own notices; inclusion does not grant a new licence. No proprietary Windows fonts, email drafts, coauthor review files or unrelated acquisition caches are included. See `ZENODO_UPLOAD_NOTES.md` for repository/version instructions.

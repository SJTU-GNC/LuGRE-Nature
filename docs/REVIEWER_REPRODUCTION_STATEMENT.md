# Reviewer Reproduction Statement

This archive registers 54 reviewable tasks:

- 9 complete manuscript figures whose regenerated PNG SHA256 values match the
  bundled accepted references;
- 38 independently runnable scientific panel redraws for figures whose final
  assembly chain is incomplete;
- 6 blocked full-figure records; and
- 1 blocked reference-only panel record (`Fig1_A`).

The 10 Fig5 panel tasks are pixel-identical to bundled standalone
source-render references. That equality is limited to those standalone panel
references and does not claim equality to crops of the accepted Fig5 montage
or reproduce its unavailable manual assembly.

Every runnable task starts from packaged numerical data or scientific source
assets. Accepted PNGs under `reference/` are validation targets only and are
never used as plotting inputs. Generated PNGs are written under `outputs/`;
the `data/` tree contains no generated display image.

The portable Python runtime is bundled. On Windows, reviewers can run:

```text
run_one.cmd --list
run_one.cmd <task_id>
run_all.cmd --group exact
run_all.cmd --group panel
verify_package.cmd
```

For ED6–ED9, `run_all.cmd --group exact --rebuild-derived` parses seven
packaged, losslessly gzip-compressed raw LuGRE telemetry text files and
reconstructs the shared receiver observation table before plotting. This path
is intentionally described as a hybrid raw rebuild: exact robust C/N0
detrending and tangent-geometry generation remain explicit, audited processed
fallbacks.

The reasons for every incomplete full chain are summarized in
`REPRODUCIBILITY_MATRIX.md` and recorded in machine-readable form in
`manifest/tasks.json` and `manifest/known_missing_chain.csv`.

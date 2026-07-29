# Reproducibility Matrix

The matrix distinguishes an accepted full-figure hash from a scientifically
recomputed panel. “Panel redraw” means that Matplotlib receives numerical
tables or scientific source assets; it does not mean that an accepted PNG was
cropped and returned as output.

| Figure | Accepted full figure | Independently runnable content | Principal remaining gap |
|---|---|---|---|
| Fig1 | Blocked | b: latitude-band common-grid relation; c: global P95 absolute delta-C/N0; d: global P95 one-minute ROTI | a is an illustrative schematic with no accepted editable source; final four-panel assembler and some point-to-grid code are absent |
| Fig2 | Blocked | a: north polar tracks; b: south polar tracks; c1–c6: selected C/N0 versus tangent-height tracks | Accepted map-plus-six-track assembler and one archived import dependency are absent |
| Fig3 | Exact | Complete figure | Accepted panel tables cannot all be rebuilt from provider files |
| Fig4 | Exact | Complete figure; final mask derivatives are recalculated | Lower-level LuGRE, LEO-RO, and ground ingestion/QC generators are incomplete |
| Fig5 | Blocked | A–E north case and F–J south case, each as an independent scientific panel; all ten match bundled standalone source-render references exactly | Accepted ten-panel manual composition and pinned raw-to-panel multisource preprocessing are absent; panel equality does not claim equality to accepted-montage crops |
| Fig6 | Exact | Complete figure; response and projected-scale derivatives are recalculated | Complete raw ground/LEO-RO and FITACF-to-gate chain is absent |
| ED1 | Exact | Complete figure | Accepted raw-to-analysis-ready table builder is absent |
| ED2 | Exact | Complete figure | Accepted raw station/orbit/occultation cache builders are absent |
| ED3 | Blocked | A: off-boresight distribution; B: scientific schematic alternative; C: four transmit-pattern plots | Accepted three-part assembler, exact B artwork, and point-table upstream provenance are incomplete |
| ED4 | Blocked | A: raw C/N0 and polynomial; B: residual; C: tangent height; D: footprint | Accepted helper revision is missing; panel tasks use a documented helper-independent redraw |
| ED5 | Blocked | A–E north case and F–J south case, each as an independent scientific panel | Accepted event selection/reordering/final assembly and pinned raw-to-panel chain are absent |
| ED6–ED9 | Exact | Complete figures; raw telemetry hybrid rebuild is available | Exact robust C/N0 detrending and tangent-geometry generators remain explicit processed fallbacks |

The machine-readable version is `manifest/tasks.json`. Every task records:

- its input level;
- whether lower-level rebuilding is supported;
- whether pixel equality is claimed;
- its output and reference;
- its known scientific or provenance limitation.

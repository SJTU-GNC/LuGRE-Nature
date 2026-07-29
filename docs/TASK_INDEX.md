# Runnable Task Index

Run a task with `run_one.cmd <task-id>`. Every runnable task leaves exactly one PNG in its declared output directory.

## Full figures

| Task | Status | Starting level | Exact accepted PNG |
|---|---|---|---|
| Fig1 | blocked | blocked | no |
| Fig2 | blocked | blocked | no |
| Fig3 | runnable | analysis-ready numeric tables to exact figure | yes |
| Fig4 | runnable | analysis-ready numeric tables to recomputed final derivatives to exact figure | yes |
| Fig5 | blocked | blocked | no |
| Fig6 | runnable | analysis-ready numeric tables to recomputed final derivatives to exact figure | yes |
| ED1 | runnable | analysis-ready numeric tables to exact figure | yes |
| ED2 | runnable | analysis-ready numeric tables to exact figure | yes |
| ED3 | blocked | blocked | no |
| ED4 | blocked | blocked | no |
| ED5 | blocked | blocked | no |
| ED6 | runnable | raw telemetry to validated hybrid observations with explicit processed fallbacks to exact figure | yes |
| ED7 | runnable | raw telemetry to validated hybrid observations with explicit processed fallbacks to exact figure | yes |
| ED8 | runnable | raw telemetry to validated hybrid observations with explicit processed fallbacks to exact figure | yes |
| ED9 | runnable | raw telemetry to validated hybrid observations with explicit processed fallbacks to exact figure | yes |

## Independent panels

| Task | Parent | Status | Starting level | Reference equality |
|---|---|---|---|---|
| Fig1_A | Fig1 | reference only | formal-raster reference only | blocked_reference_only |
| Fig1_B | Fig1 | runnable | analysis-ready grid-cell reproduction | scientific_reproduction_only |
| Fig1_C | Fig1 | runnable | analysis-ready artifact-screened grid-cell reproduction | scientific_reproduction_only |
| Fig1_D | Fig1 | runnable | analysis-ready grid-cell reproduction | scientific_reproduction_only |
| Fig2_A | Fig2 | runnable | analysis-ready WGS84 point-track reproduction | scientific_reproduction_only |
| Fig2_B | Fig2 | runnable | analysis-ready WGS84 point-track reproduction | scientific_reproduction_only |
| Fig2_C1 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig2_C2 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig2_C3 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig2_C4 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig2_C5 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig2_C6 | Fig2 | runnable | analysis-ready numerical inputs to independent panel redraw | scientific_reproduction_only |
| Fig5_A | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_B | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_C | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_D | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_E | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_F | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_G | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_H | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_I | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| Fig5_J | Fig5 | runnable | analysis-ready numeric inputs to source-code redraw | pixel-exact to declared standalone reference |
| ED3_A | ED3 | runnable | analysis-ready numeric observations to panel | scientific_reproduction_only |
| ED3_B | ED3 | runnable | numeric geometry to alternative schematic | scientific_reproduction_only |
| ED3_C | ED3 | runnable | analysis-ready transmit-pattern envelopes to panel | scientific_reproduction_only |
| ED4_A | ED4 | runnable | analysis-ready numeric table to helper-independent panel | scientific_reproduction_only |
| ED4_B | ED4 | runnable | analysis-ready residual table to helper-independent panel | scientific_reproduction_only |
| ED4_C | ED4 | runnable | analysis-ready WGS84 tangent geometry to helper-independent panel | scientific_reproduction_only |
| ED4_D | ED4 | runnable | analysis-ready simulated WGS84 footprint to helper-independent panel | scientific_reproduction_only |
| ED5_A | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_B | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_C | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_D | ED5 | runnable | selected-event YunYao/IRI/PFISR numeric profiles to panel | scientific_reproduction_only |
| ED5_E | ED5 | runnable | selected-event 2026-07-27 filled beam-range velocity cells to panel | scientific_reproduction_only |
| ED5_F | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_G | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_H | ED5 | runnable | selected-event panel-ready numeric inputs to panel | scientific_reproduction_only |
| ED5_I | ED5 | runnable | selected-event YunYao/IRI numeric profiles to panel | scientific_reproduction_only |
| ED5_J | ED5 | runnable | selected-event 2026-07-27 filled beam-range velocity cells to panel | scientific_reproduction_only |

A panel reference is a QA target only. Pixel equality for a panel does not imply that the missing accepted full-figure assembly has been recovered.

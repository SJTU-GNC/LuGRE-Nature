from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifest"
BASE = MANIFEST_DIR / "full_figures_base.json"
OUT = MANIFEST_DIR / "tasks.json"

FULL_ORDER = [
    "Fig1",
    "Fig2",
    "Fig3",
    "Fig4",
    "Fig5",
    "Fig6",
    "ED1",
    "ED2",
    "ED3",
    "ED4",
    "ED5",
    "ED6",
    "ED7",
    "ED8",
    "ED9",
]

FULL_REFERENCES = {
    "Fig1": "reference/Fig1_R1.png",
    "Fig2": "reference/Fig2.png",
    "Fig3": "reference/Fig3.png",
    "Fig4": "reference/Fig4.png",
    "Fig5": "reference/Fig5_R1.png",
    "Fig6": "reference/Fig6.png",
    "ED1": "reference/Extended_Data_Fig_01_space_weather_lugre_context_surface_phase.png",
    "ED2": "reference/Extended_Data_Fig_02_observing_geometry.png",
    "ED3": "reference/Extended_Data_Fig_03_main_lobe_antenna_geometry_R1.png",
    "ED4": "reference/Extended_Data_Fig_04_cn0_detrending_op38_e23.png",
    "ED5": "reference/Extended_Data_Fig_05_polar_validation_diagnostics_R1.png",
    "ED6": "reference/Extended_Data_Fig_06_s_phase_day1_op38_R1.png",
    "ED7": "reference/Extended_Data_Fig_07_s_phase_day2_op40_R1.png",
    "ED8": "reference/Extended_Data_Fig_08_s_phase_day3_op74_R1.png",
    "ED9": "reference/Extended_Data_Fig_09_s_phase_day4_op76_op78_R1.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parent_figure(task_id: str) -> str:
    match = re.match(r"^(Fig\d+|ED\d+)(?:_|$)", task_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer parent figure from panel task ID: {task_id}")
    token = match.group(1)
    if token.lower().startswith("fig"):
        return "Fig" + token[3:]
    return "ED" + token[2:]


def panel_letter(task_id: str) -> str:
    parts = task_id.split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def normalize_full_tasks() -> list[dict[str, object]]:
    source = json.loads(BASE.read_text(encoding="utf-8"))
    by_id = {str(task["id"]): dict(task) for task in source}
    result: list[dict[str, object]] = []
    for order, task_id in enumerate(FULL_ORDER, start=1):
        task = by_id[task_id]
        task["kind"] = "full_figure"
        task["parent_figure"] = task_id
        task["task_order"] = order
        task["reference"] = FULL_REFERENCES[task_id]
        reference = ROOT / str(task["reference"])
        task["reference_sha256"] = sha256(reference)
        if bool(task.get("runnable", False)):
            task["entry"] = f"{task_id}_main.py"
            task["output_sha256"] = str(task["reference_sha256"])
            task["comparison_class"] = "pixel_exact_accepted_full_figure"
            task["reference_scope"] = "accepted manuscript full figure"
        else:
            task["pixel_exact_verified"] = False
            task["comparison_class"] = "blocked_full_figure"
            task["reference_scope"] = "accepted manuscript full figure; validation only"
        result.append(task)
    return result


def fragments() -> list[tuple[Path, dict[str, object]]]:
    result: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(MANIFEST_DIR.glob("*_panels.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.append((path, payload))
    return result


def panel_records(payload: dict[str, object]) -> list[dict[str, object]]:
    for key in ("panels", "tasks"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value]
    figures = payload.get("figures")
    if isinstance(figures, list):
        rows: list[dict[str, object]] = []
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            nested = figure.get("panels") or figure.get("tasks")
            if isinstance(nested, list):
                for item in nested:
                    row = dict(item)
                    row.setdefault("figure", figure.get("figure") or figure.get("id"))
                    rows.append(row)
        return rows
    raise ValueError("Panel fragment has no panels/tasks/figures list")


def qa_for(task_id: str) -> dict[str, object]:
    candidates = [
        ROOT / "work" / "qa" / f"{task_id}.json",
        ROOT / "work" / "qa" / f"{task_id.lower()}.json",
        ROOT / "work" / "panel_qa" / f"{task_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    qa_root = ROOT / "work" / "qa"
    if qa_root.is_dir():
        nested = sorted(qa_root.rglob(f"{task_id}.json"))
        if not nested:
            nested = sorted(qa_root.rglob(f"{task_id.lower()}.json"))
        if nested:
            return json.loads(nested[0].read_text(encoding="utf-8"))
    return {}


def first(row: dict[str, object], *keys: str, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_panel(
    row: dict[str, object],
    payload: dict[str, object],
    fragment_path: Path,
    order: int,
) -> dict[str, object]:
    task_id = str(first(row, "id", "task_id"))
    if not task_id:
        raise ValueError(f"{fragment_path.name}: panel is missing id")
    parent = str(
        first(row, "parent_figure", "figure", default=parent_figure(task_id))
    )
    status = str(first(row, "status", default="runnable")).lower()
    runnable = status not in {
        "blocked",
        "blocked_reference_only",
        "reference_only",
        "not_runnable",
    } and bool(first(row, "entry", "entrypoint", default=True))
    if not runnable:
        reference_rel = str(first(row, "reference", default=""))
        reference_hash = ""
        if reference_rel and (ROOT / reference_rel).is_file():
            reference_hash = sha256(ROOT / reference_rel)
        reason = str(
            first(
                row,
                "blocked_reason",
                "reason",
                default="The accepted panel source or generator was not archived.",
            )
        )
        missing = first(
            row,
            "missing_items",
            default=["Accepted editable source or deterministic panel generator"],
        )
        if not isinstance(missing, list):
            missing = [str(missing)]
        return {
            "id": task_id,
            "kind": "panel",
            "parent_figure": parent,
            "panel": str(first(row, "panel", default=panel_letter(task_id))),
            "task_order": order,
            "description": f"{task_id} reference-only panel.",
            "runnable": False,
            "reproduction_level": str(
                first(row, "reproduction_level", "level", default="blocked")
            ),
            "raw_to_figure_complete": False,
            "rebuild_derived_supported": False,
            "pixel_exact_verified": False,
            "comparison_class": "blocked_reference_only",
            "reference_scope": "accepted manuscript panel raster; validation only",
            "reference": reference_rel.replace("\\", "/"),
            "reference_sha256": reference_hash,
            "blocked_reason": reason,
            "missing_items": missing,
            "limitations": [str(first(row, "exactness", default=reason))],
            "source_fragment": f"manifest/{fragment_path.name}",
        }
    qa = qa_for(task_id)
    entry = str(first(row, "entry", "entrypoint", default=f"{task_id}_main.py"))
    output_rel = str(first(row, "output", default=qa.get("output", "")))
    reference_rel = str(first(row, "reference", default=qa.get("reference", "")))
    if not output_rel or not reference_rel:
        raise ValueError(f"{task_id}: missing output/reference in fragment and QA")
    output = ROOT / output_rel
    reference = ROOT / reference_rel
    if not output.is_file() or not reference.is_file():
        raise FileNotFoundError(f"{task_id}: output or reference is absent")

    output_hash = sha256(output)
    reference_hash = sha256(reference)
    validation = str(first(row, "validation", default="")).lower()
    qa_pixel_exact = bool(qa.get("pixel_exact", False))
    pixel_exact = output_hash == reference_hash and (
        validation in {"pixel_exact", "exact", "sha256_exact"}
        or qa_pixel_exact
        or bool(row.get("pixel_exact", False))
    )
    comparison_class = str(
        first(
            row,
            "comparison_class",
            default=qa.get(
                "comparison_class",
                "pixel_exact_standalone_panel_reference"
                if pixel_exact
                else "scientific_reproduction_only",
            ),
        )
    )
    reproduction_level = str(
        first(
            row,
            "reproduction_level",
            "level",
            default=qa.get(
                "reproduction_level",
                payload.get(
                    "scientific_source_level",
                    "analysis-ready numerical inputs to independent panel redraw",
                ),
            ),
        )
    )
    method = str(first(row, "method", "observable", default=qa.get("method", "panel redraw")))
    case = str(first(row, "case", "event", "selection", default=qa.get("case", "")))
    inputs = qa.get("inputs", row.get("inputs", row.get("data", [])))
    if not isinstance(inputs, list):
        inputs = []

    default_reference_scope = (
        "archived standalone source-redrawn panel reference; "
        "not an accepted full-figure assembly"
        if fragment_path.name == "fig5_panels.json"
        else "panel-only visual-QA raster; not used as a plotting input"
    )
    reference_scope = str(
        first(
            row,
            "reference_scope",
            default=default_reference_scope,
        )
    )
    reference_type = str(
        first(
            row,
            "reference_type",
            default=payload.get("reference_type", "panel_visual_qa_raster"),
        )
    )
    reference_is_accepted_full_figure_crop = bool(
        first(
            row,
            "reference_is_accepted_full_figure_crop",
            default=payload.get("reference_is_accepted_full_figure_crop", False),
        )
    )
    pixel_exact_scope = str(
        first(
            row,
            "pixel_exact_scope",
            default=payload.get(
                "pixel_exact_scope",
                "standalone_panel_reference_only"
                if pixel_exact
                else "not_claimed",
            ),
        )
    )
    full_figure_status = str(
        first(
            row,
            "full_figure_status",
            default=payload.get(
                "full_figure_status",
                "blocked_final_assembly_chain_incomplete",
            ),
        )
    )
    limitations = first(
        row,
        "limitations",
        "limitation",
        default=payload.get(
            "limitations",
            qa.get(
                "exactness",
                [
                    "The accepted full-figure assembly chain is incomplete.",
                    "Pixel equality, when true, applies to the standalone panel reference only.",
                ],
            ),
        ),
    )
    if not isinstance(limitations, list):
        limitations = [str(limitations)]

    return {
        "id": task_id,
        "kind": "panel",
        "parent_figure": parent,
        "panel": str(first(row, "panel", default=panel_letter(task_id))),
        "task_order": order,
        "entry": entry,
        "description": (
            f"Redraw {task_id} ({method})"
            + (f" for {case}." if case else ".")
        ),
        "method": method,
        "case": case,
        "runnable": True,
        "reproduction_level": reproduction_level,
        "raw_to_figure_complete": bool(
            first(
                row,
                "raw_to_figure_complete",
                "raw_telemetry_to_panel_complete",
                default=payload.get("raw_telemetry_to_panel_complete", False),
            )
        ),
        "rebuild_derived_supported": bool(
            first(row, "rebuild_derived_supported", default=False)
        ),
        "pixel_exact_verified": pixel_exact,
        "pixel_exact_scope": pixel_exact_scope,
        "accepted_full_figure_pixel_exact_verified": False,
        "comparison_class": comparison_class,
        "reference_scope": reference_scope,
        "reference_type": reference_type,
        "reference_is_accepted_full_figure_crop": (
            reference_is_accepted_full_figure_crop
        ),
        "full_figure_status": full_figure_status,
        "output": output_rel.replace("\\", "/"),
        "reference": reference_rel.replace("\\", "/"),
        "output_sha256": output_hash,
        "reference_sha256": reference_hash,
        "inputs": inputs,
        "limitations": limitations,
        "scientific_checks": qa.get("scientific_checks", {}),
        "visual_reference_comparison": qa.get(
            "visual_reference_comparison",
            {
                "available": True,
                "byte_identical": output_hash == reference_hash,
            },
        ),
        "source_fragment": f"manifest/{fragment_path.name}",
    }


def main() -> None:
    tasks = normalize_full_tasks()
    panel_tasks: list[dict[str, object]] = []
    order = 100
    for fragment_path, payload in fragments():
        for row in panel_records(payload):
            order += 1
            panel_tasks.append(
                normalize_panel(row, payload, fragment_path, order)
            )
    panel_tasks.sort(
        key=lambda task: (
            FULL_ORDER.index(str(task["parent_figure"]))
            if str(task["parent_figure"]) in FULL_ORDER
            else 99,
            str(task["id"]),
        )
    )
    for offset, task in enumerate(panel_tasks, start=101):
        task["task_order"] = offset
    tasks.extend(panel_tasks)
    ids = [str(task["id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate task IDs while assembling manifest")
    OUT.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(tasks)} task records: "
        f"{len(tasks) - len(panel_tasks)} full figures and {len(panel_tasks)} panels."
    )


if __name__ == "__main__":
    main()

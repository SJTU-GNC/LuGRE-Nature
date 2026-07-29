from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest"
DOCS = ROOT / "docs"


def text_limitations(task: dict[str, object]) -> str:
    value = task.get("limitations") or task.get("blocked_reason") or ""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def main() -> None:
    tasks = json.loads((MANIFEST / "tasks.json").read_text(encoding="utf-8"))
    fields = [
        "id",
        "kind",
        "parent_figure",
        "panel",
        "runnable",
        "reproduction_level",
        "raw_to_figure_complete",
        "rebuild_derived_supported",
        "pixel_exact_verified",
        "comparison_class",
        "output",
        "reference",
        "limitations",
    ]
    rows: list[dict[str, object]] = []
    for task in tasks:
        row = {field: task.get(field, "") for field in fields}
        row["limitations"] = text_limitations(task)
        rows.append(row)
    with (MANIFEST / "task_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    inventory: list[dict[str, object]] = []
    for child in sorted(ROOT.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        files = list(child.rglob("*"))
        file_paths = [path for path in files if path.is_file()]
        inventory.append(
            {
                "directory": child.name,
                "files": len(file_paths),
                "bytes": sum(path.stat().st_size for path in file_paths),
            }
        )
    with (MANIFEST / "package_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["directory", "files", "bytes"]
        )
        writer.writeheader()
        writer.writerows(inventory)

    full = [task for task in tasks if task.get("kind") == "full_figure"]
    panels = [task for task in tasks if task.get("kind") == "panel"]
    lines = [
        "# Runnable Task Index",
        "",
        "Run a task with `run_one.cmd <task-id>`. Every runnable task leaves "
        "exactly one PNG in its declared output directory.",
        "",
        "## Full figures",
        "",
        "| Task | Status | Starting level | Exact accepted PNG |",
        "|---|---|---|---|",
    ]
    for task in full:
        status = "runnable" if task.get("runnable") else "blocked"
        exact = "yes" if task.get("pixel_exact_verified") else "no"
        lines.append(
            f"| {task['id']} | {status} | "
            f"{task.get('reproduction_level', '')} | {exact} |"
        )
    lines.extend(
        [
            "",
            "## Independent panels",
            "",
            "| Task | Parent | Status | Starting level | Reference equality |",
            "|---|---|---|---|---|",
        ]
    )
    for task in panels:
        status = "runnable" if task.get("runnable") else "reference only"
        equality = (
            "pixel-exact to declared standalone reference"
            if task.get("pixel_exact_verified")
            else task.get("comparison_class", "scientific-only")
        )
        lines.append(
            f"| {task['id']} | {task.get('parent_figure', '')} | {status} | "
            f"{task.get('reproduction_level', '')} | {equality} |"
        )
    lines.extend(
        [
            "",
            "A panel reference is a QA target only. Pixel equality for a panel "
            "does not imply that the missing accepted full-figure assembly has "
            "been recovered.",
            "",
        ]
    )
    (DOCS / "TASK_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote {len(tasks)} task rows, {len(inventory)} inventory rows, "
        "and docs/TASK_INDEX.md."
    )


if __name__ == "__main__":
    main()

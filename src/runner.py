from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest" / "tasks.json"
MPL_CONFIG = ROOT / "work" / "mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
# Several archived renderers resolve style resources from the process working
# directory. Normalize it once so root entry points are location-independent.
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


def _load_manifest() -> dict[str, dict[str, object]]:
    with MANIFEST.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    return {str(task["id"]): task for task in tasks}


def _declared_output(task: dict[str, object]) -> Path:
    relative = Path(str(task["output"]))
    output = (ROOT / relative).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output_root not in output.parents:
        raise RuntimeError(f"Unsafe output path in manifest: {relative}")
    return output


def _clear_task_output(task: dict[str, object]) -> Path:
    output = _declared_output(task)
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return output


def _keep_only_declared_output(task: dict[str, object]) -> None:
    output = _declared_output(task)
    if not output.is_file():
        raise RuntimeError(f"Task did not create its declared output: {output}")
    for child in output.parent.iterdir():
        if child == output:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_task(task_id: str, argv: list[str] | None = None) -> None:
    tasks = _load_manifest()
    if task_id not in tasks:
        raise SystemExit(f"Unknown task: {task_id}")

    task = tasks[task_id]
    if not bool(task.get("runnable", True)):
        reason = str(task.get("blocked_reason", "The scientific workflow is incomplete."))
        missing = task.get("missing_items", [])
        details = "\n".join(f"  - {item}" for item in missing)
        message = f"{task_id} is not runnable from the packaged scientific inputs.\n{reason}"
        if details:
            message += f"\nMissing items:\n{details}"
        raise RuntimeError(message)

    module_name = str(task["module"])
    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser(
        prog=f"{task_id}_main.py",
        description=str(task["description"]),
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Recalculate derived tables from the lowest packaged inputs before plotting.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not compare the generated output with the packaged reference.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.rebuild_derived and not bool(task.get("rebuild_derived_supported", False)):
        raise RuntimeError(
            f"{task_id} cannot rebuild its analysis-ready inputs because the "
            "required upstream preprocessing chain is not packaged."
        )

    output = _clear_task_output(task)
    try:
        module.main(
            root=ROOT,
            rebuild_derived=args.rebuild_derived,
            validate=not args.skip_validation,
        )
        _keep_only_declared_output(task)
    except Exception:
        output.unlink(missing_ok=True)
        raise

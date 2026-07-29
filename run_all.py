from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest" / "tasks.json"
RAW_OBSERVATION_BUILDER = (
    ROOT / "src" / "raw_pipeline" / "lugre" / "build_observations_s_phase.py"
)
RAW_HYBRID_TASKS = {"ED6", "ED7", "ED8", "ED9"}


def isolated_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(ROOT / "work" / "mplconfig"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def prepare_shared_raw_rebuild() -> None:
    if not RAW_OBSERVATION_BUILDER.is_file():
        raise RuntimeError(f"Missing raw-observation builder: {RAW_OBSERVATION_BUILDER}")
    print("Rebuilding shared LuGRE S-phase observations from raw telemetry...", flush=True)
    result = subprocess.run(
        [sys.executable, "-s", "-B", str(RAW_OBSERVATION_BUILDER)],
        cwd=ROOT,
        env=isolated_environment(),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"shared raw-observation builder exited with code {result.returncode}"
        )


def run_isolated(
    task: dict[str, object],
    *,
    rebuild_derived: bool,
    reuse_validated_raw: bool,
) -> None:
    task_id = str(task["id"])
    entry = (ROOT / str(task.get("entry", f"{task_id}_main.py"))).resolve()
    if ROOT not in entry.parents or not entry.is_file():
        raise RuntimeError(f"Missing or unsafe task entry point: {entry}")

    command = [sys.executable, "-s", "-B", str(entry)]
    if rebuild_derived:
        command.append("--rebuild-derived")
    env = isolated_environment()
    if reuse_validated_raw:
        env["LUGRE_REUSE_VALIDATED_RAW_REBUILD"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"isolated task process exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the packaged exact figures and/or independent panel tasks."
    )
    parser.add_argument(
        "--group",
        choices=("all", "exact", "panel"),
        default="all",
        help="Select all runnable tasks, exact full figures, or panel tasks.",
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Use the lowest packaged inputs for tasks with a validated rebuild path.",
    )
    args = parser.parse_args()

    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected: list[dict[str, object]] = []
    for task in tasks:
        if not bool(task.get("runnable", False)):
            continue
        kind = str(task.get("kind", "full_figure"))
        if args.group == "exact" and kind != "full_figure":
            continue
        if args.group == "panel" and kind != "panel":
            continue
        selected.append(task)

    if args.rebuild_derived and any(
        bool(task.get("rebuild_derived_supported", False))
        and str(task["id"]) in RAW_HYBRID_TASKS
        for task in selected
    ):
        prepare_shared_raw_rebuild()

    failures: list[tuple[str, str]] = []
    completed = 0
    for task in selected:
        task_id = str(task["id"])
        task_rebuild = args.rebuild_derived and bool(
            task.get("rebuild_derived_supported", False)
        )
        print(f"\n=== {task_id} ===", flush=True)
        try:
            run_isolated(
                task,
                rebuild_derived=task_rebuild,
                reuse_validated_raw=task_rebuild and task_id in RAW_HYBRID_TASKS,
            )
            completed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append((task_id, str(exc)))

    if failures:
        details = "\n".join(f"- {task_id}: {reason}" for task_id, reason in failures)
        raise SystemExit(f"{len(failures)} task(s) failed:\n{details}")
    print(f"\nCompleted {completed} task(s), group={args.group}.")


if __name__ == "__main__":
    main()

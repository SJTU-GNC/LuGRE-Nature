from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest" / "tasks.json"


def load_tasks() -> list[dict[str, object]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one exact-figure or independently reproducible panel task."
    )
    parser.add_argument("task_id", nargs="?", help="Task ID from manifest/tasks.json")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available task IDs and reproduction levels.",
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Use the lowest packaged inputs when that task supports it.",
    )
    args = parser.parse_args()

    tasks = load_tasks()
    if args.list or not args.task_id:
        for task in tasks:
            state = "RUN" if bool(task.get("runnable", True)) else "BLOCKED"
            level = task.get("reproduction_level", "not stated")
            print(f"{task['id']:<12} {state:<7} {level}")
        if not args.task_id:
            return

    by_id = {str(task["id"]).lower(): task for task in tasks}
    key = str(args.task_id).lower()
    if key not in by_id:
        raise SystemExit(f"Unknown task ID: {args.task_id}. Use --list to inspect IDs.")
    task = by_id[key]
    if not bool(task.get("runnable", True)):
        raise SystemExit(
            f"{task['id']} is blocked: {task.get('blocked_reason', 'workflow incomplete')}"
        )
    if args.rebuild_derived and not bool(task.get("rebuild_derived_supported", False)):
        raise SystemExit(
            f"{task['id']} does not expose a lower-level rebuild path in this package."
        )

    entry = (ROOT / str(task.get("entry", f"{task['id']}_main.py"))).resolve()
    if ROOT not in entry.parents or not entry.is_file():
        raise SystemExit(f"Missing or unsafe task entry point: {entry}")

    command = [sys.executable, "-s", "-B", str(entry)]
    if args.rebuild_derived:
        command.append("--rebuild-derived")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=isolated_environment(),
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

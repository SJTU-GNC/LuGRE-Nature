from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
TASKS_PATH = ROOT / "manifest" / "tasks.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_png(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"not a PNG container: {path}")
        size = image.size
        image.verify()
    if min(size) < 100:
        raise ValueError(f"implausibly small PNG dimensions {size}: {path}")
    return size


def main() -> None:
    if not TASKS_PATH.is_file():
        raise SystemExit("Missing manifest/tasks.json.")

    missing_roots = [
        name for name in ("data", "assets", "reference", "outputs")
        if not (ROOT / name).is_dir()
    ]
    if missing_roots:
        raise SystemExit(
            "Zenodo dataset has not been extracted into the repository root. "
            f"Missing directories: {', '.join(missing_roots)}"
        )

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    runnable = [task for task in tasks if bool(task.get("runnable", False))]
    exact_full = 0
    panel_count = 0
    for task in runnable:
        task_id = str(task["id"])
        output = ROOT / str(task["output"])
        reference = ROOT / str(task["reference"])
        for label, path in (("output", output), ("reference", reference)):
            if not path.is_file():
                failures.append(f"{task_id}: missing {label}: {path.relative_to(ROOT)}")
                continue
            try:
                verify_png(path)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{task_id}: invalid {label} PNG: {exc}")

        if not output.is_file() or not reference.is_file():
            continue
        output_hash = sha256(output)
        reference_hash = sha256(reference)
        expected_output = str(task.get("output_sha256", "")).upper()
        expected_reference = str(task.get("reference_sha256", "")).upper()
        if expected_output and output_hash != expected_output:
            failures.append(f"{task_id}: output SHA256 differs from task manifest")
        if expected_reference and reference_hash != expected_reference:
            failures.append(f"{task_id}: reference SHA256 differs from task manifest")
        if bool(task.get("pixel_exact_verified", False)) and output_hash != reference_hash:
            failures.append(f"{task_id}: pixel-exact claim fails")

        if task.get("kind") == "full_figure" and bool(task.get("pixel_exact_verified", False)):
            exact_full += 1
        if task.get("kind") == "panel":
            panel_count += 1

        inputs = task.get("inputs", [])
        if isinstance(inputs, list):
            for item in inputs:
                relative = str(item.get("path", "") if isinstance(item, dict) else item)
                if relative and not (ROOT / relative).is_file():
                    failures.append(f"{task_id}: missing declared input: {relative}")

    if failures:
        raise SystemExit(
            "Public release verification failed:\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )

    print(f"Public release data OK: {len(runnable)} runnable tasks.")
    print(f"Exact full figures: {exact_full}; runnable panel redraws: {panel_count}.")
    print("All declared inputs, output PNGs, references, and registered hashes passed.")


if __name__ == "__main__":
    main()

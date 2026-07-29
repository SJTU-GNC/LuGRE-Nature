from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUNDLED_PYTHON = ROOT / "runtime" / "python_exact" / "python.exe"
TASK_IDS = {
    "ED3_A",
    "ED3_B",
    "ED3_C",
    "ED4_A",
    "ED4_B",
    "ED4_C",
    "ED4_D",
    "ED5_A",
    "ED5_B",
    "ED5_C",
    "ED5_D",
    "ED5_E",
    "ED5_F",
    "ED5_G",
    "ED5_H",
    "ED5_I",
    "ED5_J",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _ensure_bundled_runtime() -> None:
    """Re-launch a root entry with the packaged interpreter when necessary."""
    if not BUNDLED_PYTHON.is_file():
        raise RuntimeError(f"Bundled Python runtime is missing: {BUNDLED_PYTHON}")
    try:
        current = Path(sys.executable).resolve()
        expected = BUNDLED_PYTHON.resolve()
    except OSError:
        current = Path(sys.executable)
        expected = BUNDLED_PYTHON
    if current == expected:
        return
    command = [
        str(BUNDLED_PYTHON),
        "-s",
        "-B",
        str(Path(sys.argv[0]).resolve()),
        *sys.argv[1:],
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def _clear_output(task_id: str) -> tuple[Path, Path]:
    output_dir = (ROOT / "outputs" / task_id).resolve()
    output_root = (ROOT / "outputs").resolve()
    if output_root not in output_dir.parents:
        raise RuntimeError(f"Unsafe task output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return output_dir, output_dir / f"{task_id}.png"


def _reference_path(task_id: str) -> Path:
    figure = task_id.split("_", 1)[0]
    return ROOT / "reference" / "panels" / figure / f"{task_id}_reference.png"


def _validate_single_png(output_dir: Path, output: Path) -> dict[str, object]:
    from PIL import Image

    children = list(output_dir.iterdir())
    pngs = [path for path in children if path.is_file() and path.suffix.lower() == ".png"]
    if children != pngs or len(pngs) != 1 or pngs[0] != output:
        raise RuntimeError(
            f"{output_dir} must contain exactly one PNG named {output.name}; "
            f"found {[path.name for path in children]}"
        )
    if output.stat().st_size < 10_000:
        raise RuntimeError(f"Rendered PNG is unexpectedly small: {output.stat().st_size} bytes")
    with Image.open(output) as image:
        image.verify()
    with Image.open(output) as image:
        width, height = image.size
        mode = image.mode
    if width < 600 or height < 400:
        raise RuntimeError(f"Rendered PNG is too small for review: {width}x{height}")
    return {
        "output_bytes": output.stat().st_size,
        "output_width_px": width,
        "output_height_px": height,
        "output_mode": mode,
        "output_sha256": _sha256(output),
    }


def run_panel(task_id: str) -> None:
    if task_id not in TASK_IDS:
        raise SystemExit(f"Unknown panel task: {task_id}")
    _ensure_bundled_runtime()

    os.chdir(ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "mplconfig"))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    (ROOT / "work" / "mplconfig").mkdir(parents=True, exist_ok=True)

    output_dir, output = _clear_output(task_id)
    try:
        from .plots import render_panel

        panel_qa = render_panel(task_id, ROOT, output)
        file_qa = _validate_single_png(output_dir, output)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    reference = _reference_path(task_id)
    qa: dict[str, object] = {
        "task_id": task_id,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": str(BUNDLED_PYTHON.relative_to(ROOT)).replace("\\", "/"),
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "reference": str(reference.relative_to(ROOT)).replace("\\", "/"),
        **file_qa,
        **panel_qa,
    }
    if reference.is_file():
        from PIL import Image

        with Image.open(reference) as image:
            qa.update(
                {
                    "reference_sha256": _sha256(reference),
                    "reference_width_px": image.width,
                    "reference_height_px": image.height,
                }
            )
    qa_dir = ROOT / "work" / "qa" / "panel_ed3_ed5"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"{task_id}.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"PASS {task_id}: {output.relative_to(ROOT)} "
        f"({file_qa['output_width_px']}x{file_qa['output_height_px']}, "
        f"SHA256 {file_qa['output_sha256']})"
    )

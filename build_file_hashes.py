from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manifest" / "file_hashes.json"
WORKERS = min(8, max(2, os.cpu_count() or 2))


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative == Path("manifest/file_hashes.json"):
        return False
    if relative.parts and relative.parts[0] == "work":
        return False
    if "__pycache__" in relative.parts:
        return False
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return False
    return True


def record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest().upper(),
    }


def main() -> None:
    paths = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix().lower(),
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records = list(pool.map(record, paths))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} SHA256 records to {OUT}")


if __name__ == "__main__":
    main()

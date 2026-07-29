from __future__ import annotations

import hashlib
from pathlib import Path

from . import build_tail_mask


EXPECTED_SHA256 = "752D2277C6E60FB9E8BB0B77BA4E45671A99803D8B7C90813B9C567417288804"
OUTPUT_NAME = "Fig4.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del rebuild_derived  # This task always rebuilds its tail-mask derivative tables.
    output = root / "outputs" / "Fig4" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    build_tail_mask.main()

    if not output.is_file():
        raise RuntimeError(f"Fig4 renderer did not create {output}")
    actual = _sha256(output)
    if validate and actual != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fig4 SHA256 mismatch: expected {EXPECTED_SHA256}, generated {actual}"
        )
    print(f"Fig4: {output} SHA256={actual}")

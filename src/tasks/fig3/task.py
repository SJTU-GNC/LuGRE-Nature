from __future__ import annotations

import hashlib
from pathlib import Path

from . import plot_figure


EXPECTED_SHA256 = "C0850CD88B270C74B7137B7B814BD9EB412E400DC14E83E2C20A2AD8C7B67043"
OUTPUT_NAME = "Fig3.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    if rebuild_derived:
        raise RuntimeError(
            "Fig3 packages the six analysis-ready panel tables as its lowest "
            "available scientific inputs; their raw-data generation chain is incomplete."
        )

    output = root / "outputs" / "Fig3" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    plot_figure.make_figure()

    if not output.is_file():
        raise RuntimeError(f"Fig3 renderer did not create {output}")
    actual = _sha256(output)
    if validate and actual != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fig3 SHA256 mismatch: expected {EXPECTED_SHA256}, generated {actual}"
        )
    print(f"Fig3: {output} SHA256={actual}")

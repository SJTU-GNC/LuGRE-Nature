from __future__ import annotations

import hashlib
from pathlib import Path

from . import plot_figure


EXPECTED_SHA256 = "5D4159D048C62687320F4FB26B7828764D54F64A3FD7E4B17047A7ED5D7D8D84"
OUTPUT_NAME = "Fig6.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del rebuild_derived  # This task always recomputes its final derived RA/LOS tables.
    output = root / "outputs" / "Fig6" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    plot_figure.main()

    if not output.is_file():
        raise RuntimeError(f"Fig6 renderer did not create {output}")
    actual = _sha256(output)
    if validate and actual != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fig6 SHA256 mismatch: expected {EXPECTED_SHA256}, generated {actual}"
        )
    print(f"Fig6: {output} SHA256={actual}")

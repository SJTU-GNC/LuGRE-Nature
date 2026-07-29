from __future__ import annotations

import hashlib
from pathlib import Path

from .plot_ed1 import draw_figure


EXPECTED_SHA256 = "a4dfeef6db968038af261397374b32c0b29173fca8c9f03f57e10238675b61e1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del root, rebuild_derived
    output = draw_figure()
    actual = _sha256(output)
    if validate and actual != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"ED1 SHA256 mismatch: expected {EXPECTED_SHA256}, generated {actual}"
        )
    print(f"ED1 wrote {output}")
    print(f"ED1 SHA256 {actual}")

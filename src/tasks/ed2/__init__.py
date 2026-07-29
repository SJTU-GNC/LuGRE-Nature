from __future__ import annotations

import hashlib
from pathlib import Path

from . import create_global_plus_polar_zoom_station_map_v5_fig3_consistent_time_matched as plotter


EXPECTED_SHA256 = "39f2a78c06530bc7dd3852812f5c15e1d14cc2e66df7a9050b453faf1d12a95a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del root
    if rebuild_derived:
        raise RuntimeError(
            "ED2 cannot rebuild its analysis-ready occultation caches because "
            "the original upstream preprocessing chain is not packaged."
        )
    plotter.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plotter.main()
    actual = _sha256(plotter.OUT_PNG)
    if validate and actual != EXPECTED_SHA256:
        plotter.OUT_PNG.unlink(missing_ok=True)
        raise RuntimeError(
            f"ED2 SHA256 mismatch: expected {EXPECTED_SHA256}, generated {actual}"
        )
    print(f"ED2 wrote {plotter.OUT_PNG}")
    print(f"ED2 SHA256 {actual}")

from __future__ import annotations

import json
from pathlib import Path

from .run_analysis_ready import build_one, cli, default_output


def main(*, root: Path, rebuild_derived: bool, validate: bool) -> None:
    del root
    result = build_one(
        "ED8",
        default_output("ED8"),
        rebuild_derived=rebuild_derived,
    )
    if validate and not result["matches_reference_sha256"]:
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli("ED8")

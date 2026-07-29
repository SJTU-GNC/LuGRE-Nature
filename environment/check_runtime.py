from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = PACKAGE_ROOT / "work" / "mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
sys.dont_write_bytecode = True


EXPECTED = {
    "numpy": "2.5.0",
    "pandas": "3.0.3",
    "scipy": "1.18.0",
    "matplotlib": "3.11.0",
    "seaborn": "0.13.2",
    "cartopy": "0.25.0",
    "pyproj": "3.7.2",
    "shapely": "2.1.2",
    "PIL": "12.2.0",
    "h5py": "3.16.0",
    "netCDF4": "1.7.4",
    "openpyxl": "3.1.5",
    "pyarrow": "20.0.0",
}


def main() -> None:
    failures: list[str] = []
    print(f"Python {platform.python_version()} ({platform.architecture()[0]})")
    for name, expected in EXPECTED.items():
        module = importlib.import_module(name)
        actual = str(getattr(module, "__version__", "unknown"))
        print(f"{name}: {actual}")
        if actual != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")
    if failures:
        raise SystemExit("Runtime version check failed:\n" + "\n".join(f"- {x}" for x in failures))
    print("Runtime version check passed.")


if __name__ == "__main__":
    main()

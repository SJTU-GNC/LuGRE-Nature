from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_IN = ROOT / "output" / "lugre_ionosphere_tangent_points.csv"

TRACK_KEYS = ["phase", "op_id", "sat", "signal_id", "segment_id"]
CN0_RESID_COL = "cn0_detrend_zero_mean_dbhz"
CN0_ABS_RESID_COL = "cn0_detrend_zero_mean_abs_dbhz"


def read_ionosphere_points() -> pd.DataFrame:
    if not CSV_IN.exists():
        raise FileNotFoundError(CSV_IN)
    return pd.read_csv(CSV_IN)


def split_dateline(track: pd.DataFrame) -> list[pd.DataFrame]:
    if len(track) < 2:
        return [track]
    sub = track.sort_values("rx_gps_seconds").copy()
    lon = sub["lugre_lon"].to_numpy(dtype=float)
    time_gap = sub["rx_gps_seconds"].diff().fillna(1.0).to_numpy(dtype=float) > 3.0
    lon_gap = np.r_[False, np.abs(np.diff(lon)) > 180.0]
    run_id = np.cumsum(time_gap | lon_gap)
    return [piece for _, piece in sub.groupby(run_id, sort=False) if len(piece) >= 2]

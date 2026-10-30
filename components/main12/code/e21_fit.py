from __future__ import annotations
def median_abs_deviation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.nanmedian(arr))
    return float(np.nanmedian(np.abs(arr - med)))

def binned_median_fit_points(
    t_s: np.ndarray, y: np.ndarray, t0: float, half_span: float, bin_s: float = 60.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"time_s": t_s, "cn0": y})
    df["bin_s"] = np.floor(df["time_s"] / bin_s) * bin_s + 0.5 * bin_s
    med = (
        df.groupby("bin_s", sort=True)
        .agg(cn0=("cn0", "median"), count=("cn0", "size"))
        .reset_index()
    )
    x_fit = (med["bin_s"].to_numpy(dtype=float) - t0) / half_span
    y_fit = med["cn0"].to_numpy(dtype=float)
    weights = np.sqrt(np.clip(med["count"].to_numpy(dtype=float), 1.0, None))
    return med["bin_s"].to_numpy(dtype=float), x_fit, y_fit, weights

def fit_poly_trend(
    time_s: np.ndarray, cn0: np.ndarray, degree: int
) -> tuple[np.ndarray, dict[str, object]]:
    t_s = np.asarray(time_s, dtype=float)
    y = np.asarray(cn0, dtype=float)
    finite = np.isfinite(t_s) & np.isfinite(y)
    if int(finite.sum()) <= max(20, degree + 5):
        trend = np.full_like(y, float(np.nanmedian(y[finite])), dtype=float)
        return trend, {
            "degree": 0,
            "method": "median_too_few_points",
            "fit_points": 0,
            "fit_rms_db": float("nan"),
            "residual_std_db": float("nan"),
            "coefficients": [],
            "fit_bin_time_s": [],
            "fit_bin_cn0_dbhz": [],
        }

    t0 = float(np.nanmedian(t_s[finite]))
    half_span = max(float(np.nanmax(np.abs(t_s[finite] - t0))), 1.0)
    x = (t_s - t0) / half_span
    bin_time_s, x_fit, y_fit, weights = binned_median_fit_points(
        t_s[finite], y[finite], t0, half_span
    )
    keep = np.isfinite(x_fit) & np.isfinite(y_fit) & np.isfinite(weights)
    if int(keep.sum()) <= degree + 1:
        trend = np.full_like(y, float(np.nanmedian(y_fit[keep])), dtype=float)
        resid = y[finite] - trend[finite]
        return trend, {
            "degree": 0,
            "method": "median_sparse_fit_points",
            "fit_points": int(keep.sum()),
            "fit_rms_db": float(np.sqrt(np.nanmean(resid * resid))),
            "residual_std_db": float(np.nanstd(resid)),
            "coefficients": [],
            "fit_bin_time_s": bin_time_s.tolist(),
            "fit_bin_cn0_dbhz": y_fit.tolist(),
        }

    coeff = np.polyfit(x_fit[keep], y_fit[keep], degree, w=np.clip(weights[keep], 1.0, None))
    for _ in range(4):
        fit_resid = y_fit - np.polyval(coeff, x_fit)
        scale = 1.4826 * median_abs_deviation(fit_resid[keep])
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(fit_resid[keep]))
        if not np.isfinite(scale) or scale <= 0:
            break
        new_keep = keep & (np.abs(fit_resid) <= max(1.0, 2.8 * scale))
        if int(new_keep.sum()) <= degree + 1 or int(new_keep.sum()) == int(keep.sum()):
            keep = new_keep
            break
        keep = new_keep
        coeff = np.polyfit(x_fit[keep], y_fit[keep], degree, w=np.clip(weights[keep], 1.0, None))

    trend = np.polyval(coeff, x)
    resid = y[finite] - trend[finite]
    return trend, {
        "degree": int(degree),
        "method": f"robust_poly{degree}_median60s",
        "fit_points": int(keep.sum()),
        "fit_rms_db": float(np.sqrt(np.nanmean(resid * resid))),
        "residual_std_db": float(np.nanstd(resid)),
        "coefficients": [float(v) for v in coeff],
        "fit_bin_time_s": bin_time_s[keep].tolist(),
        "fit_bin_cn0_dbhz": y_fit[keep].tolist(),
    }

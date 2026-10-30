import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
SMOOTH_SIGMA_BINS=0.5
BIN_WIDTH_DB=0.25

def plot_smoothed_histogram(ax, centers, edges, count, percent, color, order):
    positive = np.flatnonzero(count > 0)
    segments = np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1)
    smoothed = np.zeros_like(percent, dtype=float)
    reports = []
    for seg in segments:
        assert len(seg) > 0
        y = percent[seg].copy()
        if len(seg) > 1:
            y = gaussian_filter1d(y, sigma=SMOOTH_SIGMA_BINS, mode='reflect', truncate=3.0)
            y *= percent[seg].sum() / y.sum()
        smoothed[seg] = y
        assert np.isclose(y.sum(), percent[seg].sum(), atol=1e-12, rtol=1e-12)
        left, right = float(edges[seg[0]]), float(edges[seg[-1] + 1])
        if len(seg) == 1:
            x = np.array([left, right])
            values = np.repeat(y[0], 2)
        else:
            knots = np.r_[left, centers[seg], right]
            heights = np.r_[y[0], y, y[-1]]
            x = np.unique(np.r_[np.linspace(left, right, 32*(len(seg)+1)), knots])
            values = PchipInterpolator(knots, heights)(x)
            assert values.min() >= y.min() - 1e-12 and values.max() <= y.max() + 1e-12
        assert np.all(values > 0) and np.all(np.isfinite(values))
        ax.fill_between(x, .0001, values, color=color, alpha=.12, linewidth=0,
                        edgecolor='none', zorder=1+.02*order)
        ax.plot(x, values, color=color, linewidth=1.05, linestyle='-',
                solid_capstyle='butt', solid_joinstyle='round', zorder=3+order)
        reports.append({'first_bin': int(seg[0]), 'last_bin': int(seg[-1]),
                        'support_db': [left, right], 'n_bins': int(len(seg)),
                        'original_sum_percent': float(percent[seg].sum()),
                        'smoothed_sum_percent': float(y.sum()),
                        'singleton_unchanged': bool(len(seg) == 1)})
    assert np.array_equal(smoothed > 0, count > 0)
    assert np.isclose(smoothed.sum(), percent.sum(), atol=1e-12, rtol=1e-12)
    original_peak = float(percent.max())
    return {'sigma_bins': SMOOTH_SIGMA_BINS, 'sigma_db': SMOOTH_SIGMA_BINS*BIN_WIDTH_DB,
            'linear_proportion_smoothing': True, 'zero_bins_remain_zero': True,
            'original_sum_percent': float(percent.sum()),
            'smoothed_sum_percent': float(smoothed.sum()),
            'original_peak_percent': original_peak,
            'smoothed_peak_percent': float(smoothed.max()),
            'peak_reduction_percent': 100*(1-float(smoothed.max())/original_peak),
            'segments': reports}

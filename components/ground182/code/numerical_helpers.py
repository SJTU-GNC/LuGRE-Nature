"""Unchanged numerical helpers from the accepted amplitude-response implementation."""
import numpy as np
import pandas as pd
BANDS=["Equatorial","Mid-latitude","Polar"]
PLATFORMS=["Ground","LEO-RO","LuGRE"]
ANGLES=["ray_B_acute_deg","scan_Bperp_acute_deg","magnetic_meridian_acute_deg"]

def build_angle_bins(windows):
    """Equal-window medians and IQRs in the original 15-degree bins."""
    rows = []
    for platform in PLATFORMS:
        for angle in ANGLES:
            for band in BANDS:
                group = windows[windows.platform.eq(platform) & windows.latitude_band.eq(band)]
                for index, left in enumerate(range(0, 90, 15)):
                    right = left + 15
                    q = group[group[angle].ge(left) & (group[angle].le(right) if right == 90 else group[angle].lt(right))]
                    rows.append(dict(platform=platform, angle=angle, latitude_band=band, bin_id=index,
                        bin_left_deg=left, bin_right_deg=right, bin_center_deg=(left + right) / 2,
                        n_windows=len(q), n_GPS_L1=int(q.signal.eq('GPS L1').sum()),
                        n_Galileo_E1=int(q.signal.eq('Galileo E1').sum()),
                        n_sources=q.source_unit.nunique() if 'source_unit' in q else np.nan,
                        n_fitting_arcs=q.fit_arc_id.nunique() if 'fit_arc_id' in q else np.nan,
                        median_width_db=q.width_db.median(), q25_width_db=q.width_db.quantile(.25),
                        q75_width_db=q.width_db.quantile(.75)))
    return pd.DataFrame(rows)

def _amplitude_tables(groups):
    hist, signed, quantiles = [], [], []
    for (platform, band), values in groups.items():
        x = np.asarray(values, dtype='float64')
        if not len(x):
            raise ValueError(f'No observations for {platform}/{band}; do not manufacture an empty cohort.')
        # Keep the frozen quarter-dB bin grid, extending it only if real tails require it.
        left = min(-12.125, np.floor((x.min() + .125) / .25) * .25 - .125)
        right = max(12.125, np.ceil((x.max() + .125) / .25) * .25 - .125)
        if right <= x.max():
            right += .25
        edges = np.arange(left, right + .125, .25)
        counts = np.histogram(x, bins=edges)[0]
        assert counts.sum() == len(x)
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            hist.append(dict(platform=platform, latitude_band=band, bin_left_db=lo, bin_right_db=hi,
                bin_center_db=(lo + hi) / 2, bin_width_db=.25, count=int(count), n_total=len(x),
                fraction_of_all_group_records=count / len(x), percent_of_all_group_records=100 * count / len(x)))
        p50, p95, p99 = np.quantile(np.abs(x), [.5, .95, .99])
        quantiles.append(dict(platform=platform, latitude_band=band, method='provided_platform_specific',
            n_records=len(x), P50_abs_db=p50, P95_abs_db=p95, P99_abs_db=p99,
            n_source_units=np.nan, n_fitting_arcs=np.nan))
        sq = np.quantile(x, [.01, .05, .25, .5, .75, .95, .99])
        signed.append(dict(platform=platform, latitude_band=band, n_records=len(x),
            minimum_signed_db=x.min(), **dict(zip(['Q01_signed_db','Q05_signed_db','Q25_signed_db',
                'Q50_signed_db','Q75_signed_db','Q95_signed_db','Q99_signed_db'], sq)),
            maximum_signed_db=x.max(), P95_absolute_db=p95,
            n_records_outside_center_view=int((np.abs(x) > 3).sum()),
            n_records_outside_display=int((np.abs(x) > 10).sum())))
    return pd.DataFrame(hist), pd.DataFrame(signed), pd.DataFrame(quantiles)

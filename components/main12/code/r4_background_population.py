"""Three-condition measured-observation/window screening and quantile summaries."""
def qstats(y, prefix=''):
    a = np.asarray(y, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {prefix + k: None for k in ['p05', 'median', 'p95', 'W', 'mean', 'p95_abs', 'max_abs']}
    p05, med, p95 = np.quantile(a, [0.05, 0.5, 0.95])
    return dict(zip([prefix + k for k in ['p05', 'median', 'p95', 'W', 'mean', 'p95_abs', 'max_abs']], map(float, [p05, med, p95, p95 - p05, a.mean(), np.quantile(abs(a), 0.95), abs(a).max()])))

def make_population(z, name):
    z = z.copy()
    assert not z.duplicated(KEY).any()
    led = []
    for uid, q in z.groupby('window_uid', sort=False):
        eligible = q[q.point_quality_eligible].sort_values('rx_gps_seconds')
        before25 = q[q.qualified_original_quartic & q.cn0_dbhz.ge(25)]
        n = int(eligible.rx_gps_seconds.nunique())
        assert n == len(eligible)
        stats = qstats(eligible.e23_residual_db, 'pre_bias_')
        enough = n >= 30
        bias = bool(enough and abs(stats['pre_bias_median']) > 1.0)
        keep = bool(enough and (not bias))
        row = {k: q[k].iloc[0] for k in SEG + ['signal_name', 'source_segment_uid', 'window_idx', 'window_origin_gps_s', 'window_start_gps_s', 'window_start_utc', 'utc_day']}
        row.update(window_uid=uid, population=name, n_raw_points=len(q), n_qualified_quartic=int(q.qualified_original_quartic.sum()), n_degree4_and_point_cn0_ge25=len(before25), n_point_quality_eligible=n, valid_minimum_seconds=enough, bias_test_eligible=enough, reject_window_n_lt30=not enough, reject_window_median_bias=bias, window_qc_keep=keep, first_actual_utc=q.time_utc.min(), last_actual_utc=q.time_utc.max(), n_low_cn0=int(q.reject_point_cn0_lt25.sum()), n_bad_segment_cn0=int(q.reject_segment_median_cn0_lt25.sum()), n_bad_local_cn0=int(q.reject_local_60s_median_cn0_lt25.sum()), n_no_quartic=int(q.reject_no_qualified_quartic.sum()), height_min_km=float(q.lugre_h_tan_wgs84_km.min()), height_max_km=float(q.lugre_h_tan_wgs84_km.max()), median_cn0_pre_bias_dbhz=float(eligible.cn0_dbhz.median()) if n else None, max_gap_pre_bias_s=float(np.diff(eligible.rx_gps_seconds).max()) if n > 1 else None, pre_bias_strict_1Hz=bool(n > 1 and np.allclose(np.diff(eligible.rx_gps_seconds), 1, atol=1e-07, rtol=0)), **stats)
        if not enough:
            row['window_status'] = 'fewer_than_30_point_quality_seconds'
        elif bias:
            row['window_status'] = 'absolute_signed_median_exceeds_1dB'
        else:
            row['window_status'] = 'retained'
        led.append(row)
    w = pd.DataFrame(led)
    z = z.merge(w[['window_uid', 'n_point_quality_eligible', 'pre_bias_p05', 'pre_bias_median', 'pre_bias_p95', 'pre_bias_W', 'reject_window_n_lt30', 'reject_window_median_bias', 'window_qc_keep', 'window_status']], on='window_uid', validate='many_to_one')
    z['analysis_qc_keep'] = z.point_quality_eligible & z.window_qc_keep
    z['window_width_W_db'] = z.pre_bias_W.where(z.window_qc_keep)
    z['display_grey'] = ~z.analysis_qc_keep
    reasons = [('reject_no_qualified_quartic', 'no_qualified_original_degree4_fit'), ('reject_point_cn0_lt25', 'point_CN0_below25'), ('reject_segment_median_cn0_lt25', 'native_segment_median_CN0_below25'), ('reject_local_60s_median_cn0_lt25', 'centered60s_median_CN0_below25'), ('reject_window_n_lt30', 'window_has_fewer_than30_valid_seconds'), ('reject_window_median_bias', 'whole_window_abs_signed_median_above1dB')]
    z['exclusion_reasons'] = ''
    for field, reason in reasons:
        z.loc[z[field], 'exclusion_reasons'] = z.loc[z[field], 'exclusion_reasons'] + reason + ';'
    z['exclusion_reasons'] = z.exclusion_reasons.str.rstrip(';')
    assert z.loc[z.analysis_qc_keep, 'exclusion_reasons'].eq('').all()
    assert z.loc[z.display_grey, 'exclusion_reasons'].ne('').all()
    assert not z.loc[z.reject_window_median_bias, 'analysis_qc_keep'].any()
    z.to_parquet(DATA / f'{name}_all_points_with_QC_flags.parquet', index=False)
    z.to_csv(DATA / f'{name}_all_points_with_QC_flags.csv', index=False)
    w.to_csv(DATA / f'{name}_window_QC_ledger.csv', index=False)
    kept = w[w.window_qc_keep].copy()
    kept['W_db'] = kept.pre_bias_W
    kept.to_csv(DATA / f'{name}_retained_60s_windows.csv', index=False)
    cases = []
    for flag, _ in reasons:
        f = z[z[flag]]
        if f.empty:
            continue
        ids = f.groupby('window_uid').size().sort_values(ascending=False).head(3).index
        x = w[w.window_uid.isin(ids)].copy()
        x['case_reason'] = flag
        cases.append(x)
    if cases:
        pd.concat(cases, ignore_index=True).to_csv(DATA / f'{name}_QC_example_windows.csv', index=False)
    summaries = []
    for sig, q in z.groupby('signal_name', sort=False):
        window = w[w.signal_name.eq(sig)]
        for stage, mask in [('qualified_degree4_positive_CN0', q.qualified_original_quartic), ('degree4_point_CN0_ge25', q.qualified_original_quartic & q.cn0_dbhz.ge(25)), ('point_quality_before_window_screen', q.point_quality_eligible), ('final_QC_retained', q.analysis_qc_keep)]:
            summaries.append(dict(population=name, signal_name=sig, stage=stage, n_actual_points=int(mask.sum()), **qstats(q.loc[mask, 'e23_residual_db'])))
    pd.DataFrame(summaries).to_csv(DATA / f'{name}_point_before_after_statistics.csv', index=False)
    pointflags = {flag: int(z[flag].sum()) for flag, _ in reasons}
    pre = w[w.valid_minimum_seconds]
    report = {'population': name, 'raw_n': len(z), 'qualified_quartic_n': int(z.qualified_original_quartic.sum()), 'point_CN0_ge25_and_quartic_n': int((z.qualified_original_quartic & z.cn0_dbhz.ge(25)).sum()), 'point_quality_n': int(z.point_quality_eligible.sum()), 'final_retained_point_n': int(z.analysis_qc_keep.sum()), 'grey_point_n': int(z.display_grey.sum()), 'overlapping_rejection_reason_counts': pointflags, 'window_status_counts': w.window_status.value_counts().to_dict(), 'before_bias_valid_window_n': len(pre), 'retained_window_n': len(kept), 'before_bias_W': qstats(pre.pre_bias_W), 'retained_window_W': qstats(kept.W_db), 'retained_window_W_median': float(kept.W_db.median()) if len(kept) else None, 'retained_window_W_P95': float(kept.W_db.quantile(0.95)) if len(kept) else None, 'retained_windows_W_ge1p5': int(kept.W_db.ge(1.5).sum()), 'no_point_magnitude_outlier_filter': True, 'whole_bias_window_exclusion_verified': True}
    return (z, w, kept, report)

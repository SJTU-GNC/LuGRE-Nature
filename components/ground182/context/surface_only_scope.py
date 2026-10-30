"""Restrict panel e to the source's S (lunar-surface) phase."""
import json
import hashlib


def install_surface_scope(m, output_dir):
    full = m.load_luna_tracks()
    surface = full.loc[full['phase'].eq('S')].copy()
    dates = sorted(surface['time_utc'].dt.strftime('%Y-%m-%d').unique())
    assert dates == ['2025-03-03','2025-03-04','2025-03-14','2025-03-15','2025-03-16']
    assert surface['lugre_h_tan_km'].between(50,1000).all()
    assert len(surface)==139809
    assert m.luna_track_count(surface)==461 and surface['sat'].nunique()==38
    assert set(surface.index).issubset(full.index)
    full_colors = m.luna_satellite_color_map(full)
    def preserved_satellite_colors(tracks):
        return {sat:full_colors[sat] for sat in sorted(tracks['sat'].astype(str).unique())}
    m.luna_satellite_color_map = preserved_satellite_colors
    m.load_luna_tracks = lambda: surface.copy()
    summary = (surface.groupby(m.LUNA_TRACK_GROUP_COLS,dropna=False)
               .agg(points=('time_utc','size'),start_utc=('time_utc','min'),end_utc=('time_utc','max'),
                    height_min_km=('lugre_h_tan_km','min'),height_max_km=('lugre_h_tan_km','max')).reset_index())
    summary.to_csv(output_dir/'lunar_surface_track_ledger.csv',index=False)
    phases = {}
    for phase, group in full.groupby('phase'):
        phases[phase] = {'points':len(group),'satellite_signal_tracks':m.luna_track_count(group),
                         'satellites':int(group['sat'].nunique()),
                         'start_utc':group['time_utc'].min().isoformat(),
                         'end_utc':group['time_utc'].max().isoformat()}
    with m.LUNA_TRACK_ZIP.open('rb') as f:
        source_hash = hashlib.file_digest(f,'sha256').hexdigest()
    evidence = {'source_zip':str(m.LUNA_TRACK_ZIP),'source_sha256':source_hash,
                'selection':"phase == 'S'; retain original 50–1,000 km tangent-height selection",
                'before':{'tracks':m.luna_track_count(full),'satellites':int(full['sat'].nunique()),'points':len(full)},
                'after':{'tracks':461,'satellites':38,'points':139809},
                'dates_utc':dates,'phases_in_original_figure':phases,
                'accepted_coordinates_and_times_unchanged':True,'original_satellite_colours_preserved':True}
    (output_dir/'lunar_surface_selection.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    return evidence

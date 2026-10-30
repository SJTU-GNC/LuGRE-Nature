"""Fig. 2 R3 polar tracks: original WGS84 science, consistent export styling.

Import has no file-writing or figure-export side effects. ``draw_polar`` expects
normalized figure bounds and returns (axes, ScalarMappable, audit dictionary).
The revision restores the original nonuniform amplitude bins with a dedicated colourbar.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import types

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import cartopy.io.shapereader as shpreader

PROJECT = Path('D:/Reference/Paper/2026/Nature_LuGRE')
FIGURE = PROJECT / 'Figure'
PROGRAM = PROJECT / 'Program'
POLAR_SCRIPT = PROGRAM / 'plot_lugre_polar_abs_nonuniform_polarcoords.py'
POLAR_BASE_SCRIPT = PROGRAM / 'plot_lugre_ionosphere_tangent_tracks_by_phase.py'
TRACK_SCRIPT = FIGURE / 'main_text_panel_png_exports_20260630/redraw_fig1_source_panels_no_labels.py'
POINTS = Path('E:\\LUGRE\\figures_R3\\LuGRE_E23_h50_R3_20261028\\data\\updated_points.csv')
SUMMARY = FIGURE / 'main_figures_lugre_reference_update_20260622/Fig1_replace_1_OP76G01_replace_6_OP38E25_new_reference_ecef_summary.csv'
TRACK_QA = FIGURE / 'Fig2_polar_track_panels_wgs84_update_20260702/data/fig2_selected_tracks_wgs84_height_latlon_QA.csv'
SOURCE_FILES = (POLAR_SCRIPT, POLAR_BASE_SCRIPT, TRACK_SCRIPT, POINTS, SUMMARY, TRACK_QA)
COLORBAR_TICKS = [0, 0.1, 0.5, 1, 2, 4]
COLORBAR_TICKLABELS = ['0', '0.1', '0.5', '1', '2', r'$\geq4$']


def _constants(path, names):
    values = {}
    for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != set(names):
        raise RuntimeError(f'Missing source constants: {set(names) - set(values)}')
    return values


def _functions(path, names, environment):
    nodes = [node for node in ast.parse(path.read_text(encoding='utf-8-sig')).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.ImportFrom) and node.module == '__future__')]
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if found != set(names):
        raise RuntimeError(f'Missing source functions: {set(names) - found}')
    module = types.ModuleType(path.stem)
    module.__dict__.update(environment)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=1)
def _load():
    base = types.SimpleNamespace(**_constants(POLAR_BASE_SCRIPT,
        {'TRACK_KEYS', 'CN0_RESID_COL', 'CN0_ABS_RESID_COL'}))
    env = dict(np=np, pd=pd, mpl=mpl, plt=plt, LineCollection=LineCollection,
               shpreader=shpreader, base=base)
    env.update(_constants(POLAR_SCRIPT,
        {'POLAR_LAT_DEG', 'MAP_LAT_MIN', 'CMAX_DBHZ', 'BOUNDARIES'}))
    polar = _functions(POLAR_SCRIPT, {'wrap_lon', 'iter_lines', 'polar_radius',
        'setup_axis', 'kept_runs', 'split_geo_piece', 'draw_land', 'draw_coastlines',
        'split_track', 'add_tracks'}, env)
    selected = _functions(TRACK_SCRIPT, {'split_track_hemi',
        'draw_selected_track_on_polar', 'add_native_label'},
        dict(np=np, pd=pd, LineCollection=LineCollection, polar_map=polar))
    points = pd.read_csv(POINTS, parse_dates=['time_utc'])
    points = points[points.phase.eq('S') & points.lugre_h_tan_km.between(50, 1000)].copy()
    assert points.phase.eq('S').all()
    summary = pd.read_csv(SUMMARY, parse_dates=['time_start', 'time_end'])
    qa = pd.read_csv(TRACK_QA)
    cases = {}
    for hemisphere, display_label, expected_n in [('north', 1, 591), ('south', 4, 930)]:
        row = next(summary[summary.display_label.eq(display_label)].itertuples(index=False))
        assert row.phase == 'S', 'Highlighted cases must be lunar-surface tracks'
        group = points[points.phase.eq(row.phase)
                       & points.op_id.eq(row.group_uid.split('|')[1])
                       & points.sat.eq(row.sat)
                       & points.time_utc.between(row.time_start, row.time_end)].copy()
        expected = qa[qa.display_label.eq(display_label)].iloc[0]
        assert len(group) == expected.n_rows_wgs84_shell == expected_n
        assert np.isclose(group.lugre_h_tan_km.min(), expected.h_wgs84_min_km, atol=1e-10)
        assert np.isclose(group.lugre_h_tan_km.max(), expected.h_wgs84_max_km, atol=1e-10)
        cases[hemisphere] = (row, group)
    return polar, selected, points, cases


def selected_cases():
    """Return validated north/south measured cases for adjacent profile panels."""
    return {k: (row, group.copy()) for k, (row, group) in _load()[3].items()}


def draw_polar(fig, bounds, hemisphere, case_label, *, cmap_name='viridis'):
    """Draw one source-grounded polar map at publication size, without exporting.

    Parameters
    ----------
    bounds : [left, bottom, width, height] in figure fractions.
    hemisphere : 'north' or 'south'.
    case_label : Adjacent measured-profile panel letter, e.g. 'h' or 'k'.
    """
    if hemisphere not in ('north', 'south'):
        raise ValueError('hemisphere must be north or south')
    polar, selected, points, cases = _load()
    cmap = mpl.colormaps[cmap_name].resampled(len(polar.BOUNDARIES)-1)
    norm = mpl.colors.BoundaryNorm(polar.BOUNDARIES,cmap.N,clip=True)
    ax = fig.add_axes(bounds, projection='polar', facecolor='white')
    polar.setup_axis(ax, hemisphere, '')
    polar.draw_coastlines(ax, hemisphere)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * 0.45)
        if line.get_zorder() == 0.5:
            line.set_color('#A5ABB2')
    ax.grid(True, color='#9AA3AD', linewidth=0.35, alpha=0.60)
    ax.tick_params(axis='x', labelsize=6.5, pad=-11.0, colors='#4C555F')
    ax.tick_params(axis='y', labelsize=6.3, pad=0, colors='#4C555F')
    for text in ax.get_xticklabels():
        text.set_fontfamily('Arial')
        text.set_fontweight('normal')
        text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.90, pad=0.05))
    for text in list(ax.texts):
        if text.get_text() in ('90°N','90°S'):
            text.remove()
    ax.set_yticklabels([])
    suffix='N' if hemisphere=='north' else 'S'
    for latitude,radius in [(90,0),(80,10),(70,20),(60,30)]:
        label=ax.annotate(f'{latitude}°{suffix}',xy=(np.deg2rad(202.5),radius),xycoords='data',
            xytext=(-1,-2),textcoords='offset points',ha='center',va='center',fontsize=6.3,
            fontfamily='Arial',fontweight='normal',color='#4C555F',zorder=8,annotation_clip=False)
        label.set_path_effects([pe.withStroke(linewidth=.9,foreground='white')])
    ax.spines['polar'].set_linewidth(0.60)
    ax.spines['polar'].set_edgecolor('#343B43')
    map_points = points.loc[points.signal_name.isin(['GPS L1', 'Galileo E1']) & np.isfinite(points[polar.base.CN0_ABS_RESID_COL])].copy()
    assert set(map_points.signal_name.unique()) <= {'GPS L1', 'Galileo E1'}
    n_tracks = polar.add_tracks(ax, map_points, hemisphere, cmap, norm, linewidth=0.88)
    row, full_group = cases[hemisphere]
    group = full_group.loc[full_group.signal_name.isin(['GPS L1', 'Galileo E1'])].copy()
    assert len(group) == (319 if hemisphere == 'north' else 543)
    assert group.signal_name.eq('GPS L1' if hemisphere == 'north' else 'Galileo E1').all()
    first = len(ax.collections)
    selected.draw_selected_track_on_polar(ax, group, hemisphere, cmap, norm, 4.0)
    for collection in ax.collections[first:]:
        collection.set_linewidth(collection.get_linewidth() * 0.82)
    selected.add_native_label(ax, group, hemisphere, case_label)
    label = ax.texts[-1]
    label.set_fontsize(6.8)
    label.set_fontfamily('Arial')
    label.get_bbox_patch().set_linewidth(0.6)
    audit = dict(revision='R3_L1_E1_v21', hemisphere=hemisphere, background_tracks=n_tracks,
                 source_points_rows_in_50_1000_km=len(points),
                 map_frequency_selection=['GPS L1','Galileo E1'],
                 map_source_rows=len(map_points),
                 map_selected_signal_counts=group.signal_name.value_counts().to_dict(),
                 original_dual_frequency_case_rows=len(full_group),
                 phase_selection='S', source_population='LuGRE lunar-surface operations',
                 all_background_and_highlight_samples_phase_S=True,
                 selected_case=f'{row.op} {row.sat}', selected_case_rows=len(group),
                 selected_case_panel=case_label, selected_other_cases_removed=True,
                 time_start=str(group.time_utc.min()), time_end=str(group.time_utc.max()),
                 original_boundaries=list(polar.BOUNDARIES), cmap=cmap_name,
                 normalization='Original nonuniform amplitude BoundaryNorm, clip=True; independent polar colourbar',
                 tick_style='Arial normal, longitude 6.5 pt, latitude 6.3 pt, softened labels and white halo; latitude labels at202.5deg between180 and135W',
                 polar_latitude_threshold=float(polar.POLAR_LAT_DEG),
                 source_values_unchanged=False, source_paths=[str(p) for p in SOURCE_FILES])
    return ax, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), audit


def source_manifest():
    return [{'path': str(p), 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCE_FILES]

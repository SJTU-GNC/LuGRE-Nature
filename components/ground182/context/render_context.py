"""Reflow station legends and electron-density units in the frozen six-panel figure.

Run via reproduce_with_surface_only.py with the existing Python runtime.  --validate-only reads the small station
CSV and optional snapshot JSON without importing the plotting stack or rendering.  The frozen
entry point, original source files, source data and existing figure are untouched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys

LOCAL_ROOT = Path(__file__).resolve().parent
ROOT = LOCAL_ROOT
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DEFAULT_CSV = LOCAL_ROOT / "data/Ground_contributing143_stations_R3.csv"
DEFAULT_OUTPUT = LOCAL_ROOT
DEFAULT_THINNED_OUTPUT = RESULTS / "ground_1hz_thinned"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT / "inventory_provenance.json"
V5_PATH = ROOT / "code" / "create_global_plus_polar_zoom_station_map_v5_fig3_consistent_time_matched.py"
BASE_PATH = ROOT / "code" / "create_global_plus_polar_zoom_station_map.py"
BASELINE_SUMMARY = RESULTS / "panel_summary_R3.csv"
REQUESTED_DATES = ("2025-03-03", "2025-03-04", "2025-03-14", "2025-03-15", "2025-03-16")
REQUIRED_COLUMNS = ("station", "lat", "lon", "processed_files", "cn0_rows", "unit_verified_rows", "dates")
COUNT_KEY = "1 Hz C/N0 station"
ORANGE = "#ef8a23"
STATION_LEGEND = (
    ("GNSS/ROTI station", "Polar GNSS", "5/15/30 s"),
    ("CHAIN/ISMR station", "GNSS scintillation", "50 Hz"),
    (COUNT_KEY, "GNSS CORS", "1 Hz"),
)
POLAR_LIMIT = 60.0
THINNING_NOTE = "5\N{DEGREE SIGN} \N{MULTIPLICATION SIGN} 5\N{DEGREE SIGN} thinning in non-polar Europe and Australia; other regions unchanged."
sys.dont_write_bytecode = True


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _nonnegative_integer(value: str, name: str, line: int) -> int:
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"CSV row {line}: invalid {name}") from error
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"CSV row {line}: {name} must be a nonnegative integer")
    return int(number)


def _archive_dates(value: str, line: int) -> list[str]:
    text = str(value).strip()
    values = json.loads(text) if text.startswith("[") else re.split(r"[;,|\s]+", text)
    if not isinstance(values, list):
        raise ValueError(f"CSV row {line}: dates must be a list or separated ISO dates")
    dates = sorted({str(item).strip() for item in values if str(item).strip()})
    if not dates:
        raise ValueError(f"CSV row {line}: dates cannot be empty")
    for value in dates:
        if date.fromisoformat(value).isoformat() != value or value not in REQUESTED_DATES:
            raise ValueError(f"CSV row {line}: archive date outside the five requested days: {value}")
    return dates


def _snapshot_evidence(station_csv: Path, provenance_path: Path | None) -> dict:
    path = provenance_path if provenance_path is not None else DEFAULT_PROVENANCE
    evidence = {"inventory_provenance": str(path.resolve()), "snapshot_started_utc": None}
    if not path.is_file():
        return evidence
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(record, dict):
        raise ValueError("inventory_provenance.json must contain a JSON object")
    value = record.get("snapshot_started_utc")
    if value:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("inventory_provenance.json snapshot_started_utc must have an explicit UTC offset")
        evidence["snapshot_started_utc"] = timestamp.astimezone(timezone.utc).isoformat()
    evidence["inventory_provenance_sha256"] = sha256_file(path)
    return evidence


def load_ground_stations(path: Path, provenance_path: Path | None = None) -> tuple[list[dict], dict]:
    """Require one explicit station ID per row; never infer four-character aliases."""
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
        raise ValueError(f"Station CSV must contain: {', '.join(REQUIRED_COLUMNS)}")
    stations, seen, excluded, missing_position = [], set(), [], []
    input_rows = 0
    for line, record in enumerate(reader, start=2):
        input_rows += 1
        station = str(record["station"] or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{4,9}", station):
            raise ValueError(f"CSV row {line}: invalid station ID {station!r}")
        if station in seen:
            raise ValueError(f"CSV row {line}: duplicate station {station}; export one aggregated row per full ID")
        seen.add(station)
        counts = {name: _nonnegative_integer(record[name], name, line)
                  for name in ("processed_files", "cn0_rows", "unit_verified_rows")}
        if counts["unit_verified_rows"] > counts["cn0_rows"]:
            raise ValueError(f"CSV row {line}: unit_verified_rows exceeds cn0_rows")
        dates = _archive_dates(record["dates"], line)
        if (counts["processed_files"] == 0 or counts["cn0_rows"] == 0
                or counts["unit_verified_rows"] == 0):
            excluded.append(station)
            continue
        if str(record.get("coordinate_valid", "")).strip().lower() in {"false", "0", "no"}:
            missing_position.append(station)
            continue
        try:
            lat, lon = float(record["lat"]), float(record["lon"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV row {line}: missing or invalid station coordinates without an explicit coordinate_valid=false flag") from error
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"CSV row {line}: station coordinates outside geographic bounds")
        stations.append({**record, "station": station, "lat": lat, "lon": lon,
                         **counts, "dates": dates})
    stations.sort(key=lambda item: item["station"])
    if not stations:
        raise ValueError("No station has completed processed files, positive C/N0 rows, and at least one verified dB-Hz observation")
    north = [s["station"] for s in stations if s["lat"] >= POLAR_LIMIT]
    south = [s["station"] for s in stations if s["lat"] <= -POLAR_LIMIT]
    report = {
        "csv": str(path.resolve()), "csv_sha256": hashlib.sha256(raw).hexdigest(),
        "input_columns": reader.fieldnames, "input_rows": input_rows,
        "certified_stations_before_coordinate_filter": len(stations) + len(missing_position),
        "eligible_unique_stations": len(stations), "north_unique_stations": len(north),
        "south_unique_stations": len(south),
        "mid_low_latitude_unique_stations": len(stations) - len(north) - len(south),
        "station_ids": [s["station"] for s in stations], "north_station_ids": north,
        "south_station_ids": south, "excluded_without_verified_processed_cn0": excluded,
        "excluded_missing_position": sorted(missing_position),
        "processed_files": sum(s["processed_files"] for s in stations),
        "cn0_rows": sum(s["cn0_rows"] for s in stations),
        "unit_verified_rows": sum(s["unit_verified_rows"] for s in stations),
        "stations_with_unverified_unit_rows": sum(s["unit_verified_rows"] < s["cn0_rows"] for s in stations),
        "actual_archive_dates": sorted({d for s in stations for d in s["dates"]}),
        "requested_archive_dates": list(REQUESTED_DATES), "snapshot_partial": True,
        **_snapshot_evidence(path, provenance_path),
        "admission": "processed_files > 0 AND cn0_rows > 0 AND unit_verified_rows > 0",
        "counting": "Unique full station IDs across dates/runs; geographic latitude >=60 or <=-60, inclusive.",
        "category_overlap": "The new C/N0 sites may overlap historical GNSS/ROTI or ISMR sites; category counts must not be summed as unique sites.",
        "unit_scope": "Every plotted station has at least one unit-verified dB-Hz C/N0 observation. cn0_rows still includes retained raw-unit rows; the station marker does not assert that all rows have verified units or that all five dates are complete.",
    }
    return stations, report


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_selection_provenance(path: Path | None, stations: list[dict], input_report: dict) -> dict | None:
    """Link an externally selected CSV to its full input without resampling it."""
    if path is None:
        return None
    raw = path.read_bytes()
    record = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(record, dict) or not isinstance(record.get("sampling"), dict):
        raise ValueError("Selection provenance must contain a sampling object")
    sampling = record["sampling"]
    if (sampling.get("grid_degrees") != 5 or sampling.get("non_polar_only") is not True
            or set(sampling.get("regions", [])) != {"Europe", "Australia"}):
        raise ValueError("The thinning caption requires 5-degree, non-polar Europe/Australia selection provenance")
    if record.get("selected_csv_sha256") != input_report["csv_sha256"]:
        raise ValueError("Selected station CSV does not match selection_provenance.json SHA256")
    declared = record.get("selected_station_ids")
    if not isinstance(declared, list) or len(declared) != len(set(declared)):
        raise ValueError("Selection provenance needs unique selected_station_ids")
    if set(declared) != set(input_report["station_ids"]):
        raise ValueError("Selection station IDs do not match the actual mapped CSV stations")
    if record.get("count_after") != input_report["eligible_unique_stations"]:
        raise ValueError("Selection count_after does not match the actual mapped station count")
    full_stations, full_report = load_ground_stations(DEFAULT_CSV, Path(input_report["inventory_provenance"]))
    if record.get("input_csv_sha256") != full_report["csv_sha256"]:
        raise ValueError("The full source CSV differs from selection provenance input_csv_sha256")
    if record.get("count_before") != full_report["eligible_unique_stations"]:
        raise ValueError("Selection count_before differs from the eligible full station inventory")
    source_by_id = {s["station"]: s for s in full_stations}
    for station in stations:
        original = source_by_id.get(station["station"])
        if original is None or (station["lat"], station["lon"]) != (original["lat"], original["lon"]):
            raise ValueError(f"Selected station is new or its coordinates changed: {station['station']}")
    polar_ids = set(full_report["north_station_ids"]) | set(full_report["south_station_ids"])
    if not polar_ids.issubset(set(input_report["station_ids"])):
        raise ValueError("Non-polar thinning must retain every original polar station")
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "sampling": sampling, "count_before": record["count_before"], "count_after": record["count_after"],
        "per_region": record.get("per_region", {}), "selected_station_ids": declared,
        "input_csv": str(DEFAULT_CSV), "input_csv_sha256": full_report["csv_sha256"],
        "selected_csv_sha256": input_report["csv_sha256"],
        "all_polar_stations_retained": True, "selected_station_coordinates_unchanged": True,
        "caption": THINNING_NOTE,
    }


def _patch_ground_overlay(original, stations: list[dict], input_report: dict,
                          selection_report: dict | None = None) -> dict:
    """Only wrap the three station panels and their legends, never c/e/f."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy

    old_polar = original.draw_polar_panel
    old_global = original.draw_global_panel_nature
    old_handles = original.station_legend_handles
    state = {"artists": {}, "axes": {}, "counts": {}, "legends": {}, "full_counts": {}}

    def draw_legends(ax, legend_handles, legend_call, *, loc, anchor, fontsize,
                     handlelength=0.95, handletextpad=0.28, columnspacing=0.85):
        panel = next(key for key, value in state["axes"].items() if value is ax)
        station_handles = [h for h in legend_handles if str(h.get_gid()).startswith("station-legend:")]
        context_handles = [h for h in legend_handles if h not in station_handles]
        for handle in context_handles:
            text = handle.get_label()
            handle.set_label(text.replace("ISR radars", "ISR").replace("SuperDARN radars", "SuperDARN"))
        cors = [h for h in station_handles if h.get_gid() == "station-legend:" + COUNT_KEY]
        other_stations = [h for h in station_handles if h not in cors]
        if panel == "a":
            coverage = [h for h in context_handles if h.get_label() in ("ISR range", "SuperDARN fan")]
            radars = [h for h in context_handles if h not in coverage]
            rows = [other_stations, cors + radars, coverage]
        else:
            rows = [other_stations, cors + context_handles]
        fontsize = 5.2
        common = dict(loc=loc, frameon=False, fontsize=fontsize,
                      handlelength=0.8, handletextpad=0.22,
                      columnspacing=0.4, borderaxespad=0.0,
                      markerscale=0.75)
        legends = []
        for row_index, row in enumerate(rows):
            if not row:
                continue
            kwargs = {}
            if row_index:
                kwargs["bbox_transform"] = offset_copy(ax.transAxes, fig=ax.figure,
                                                       y=-row_index * 10.5, units="points")
            legend = legend_call(handles=row, ncol=len(row), bbox_to_anchor=anchor,
                                 **common, **kwargs)
            if row_index < len(rows)-1:
                ax.add_artist(legend)
            legends.append(legend)
        state["legends"][panel] = legends
        return legends[-1]

    def overlay(ax, panel, subset, size):
        artist = ax.scatter([s["lon"] for s in subset], [s["lat"] for s in subset],
                            s=size, marker="^", facecolors="none", edgecolors=ORANGE,
                            linewidths=0.45 if panel == "d" else 0.60, alpha=1.0, transform=original.ccrs.PlateCarree(),
                            zorder=14, label=f"1 Hz C/N0 sites ({len(subset)})")
        artist.set_gid(f"processed-ground-1hz-cn0-panel-{panel}")
        state["artists"][panel] = artist
        state["axes"][panel] = ax
        state["counts"][panel] = len(subset)
        return len(subset)

    def polar(old, ax, hemisphere, *args, **kwargs):
        counts = dict(old_polar(old, ax, hemisphere, *args, **kwargs))
        if hemisphere not in ("north", "south"):
            raise ValueError(f"Unknown polar hemisphere: {hemisphere}")
        subset = [s for s in stations if s["lat"] >= POLAR_LIMIT] if hemisphere == "north" else [s for s in stations if s["lat"] <= -POLAR_LIMIT]
        counts[COUNT_KEY] = overlay(ax, "a" if hemisphere == "north" else "b", subset, 26.0)
        state["full_counts"]["a" if hemisphere == "north" else "b"] = counts
        return counts

    def global_panel(old, ax, *args, **kwargs):
        counts = dict(old_global(old, ax, *args, **kwargs))
        counts[COUNT_KEY] = overlay(ax, "d", stations, 8.0)
        state["full_counts"]["d"] = counts
        # This one axis bypasses add_station_legend in the frozen builder.
        original_axis_legend = ax.legend

        def global_legend(*legend_args, **legend_kwargs):
            return draw_legends(
                ax, legend_kwargs["handles"], original_axis_legend,
                loc=legend_kwargs["loc"], anchor=legend_kwargs["bbox_to_anchor"],
                fontsize=legend_kwargs["fontsize"],
                handlelength=legend_kwargs["handlelength"],
                handletextpad=legend_kwargs["handletextpad"])

        ax.legend = global_legend
        return counts

    def handles(old, counts, include_context=False):
        result = list(old_handles(old, counts, include_context=include_context))
        if COUNT_KEY in counts:
            result.append(Line2D([0], [0], marker="^", color="none", markerfacecolor="none",
                                 markeredgecolor=ORANGE, markeredgewidth=0.60, markersize=5.0,
                                 label=f"1 Hz C/N0 sites ({int(counts[COUNT_KEY])})"))
        station_handles = []
        prefixes = ("Polar GNSS receivers (", "GNSS scintillation receivers (", "1 Hz C/N0 sites (")
        for (key, name, sampling), prefix in zip(STATION_LEGEND, prefixes):
            count = int(counts.get(key, 0) or 0)
            if count:
                handle = next(h for h in result if h.get_label().startswith(prefix))
                handle.set_label(f"{name} ({sampling}; n={count})")
                handle.set_gid("station-legend:" + key)
                station_handles.append(handle)
        return station_handles + [h for h in result if h not in station_handles]

    def station_legend(ax, old, counts, **kwargs):
        return draw_legends(ax, handles(old, counts, kwargs.get("include_context", False)),
                            ax.legend, loc=kwargs["loc"], anchor=kwargs["anchor"],
                            fontsize=kwargs.get("fontsize", 4.6))

    original.draw_polar_panel = polar
    original.draw_global_panel_nature = global_panel
    original.station_legend_handles = handles
    original.add_station_legend = station_legend
    return state


def _patch_panel_header_alignment(original):
    """Align each labelled panel's letter and title on a shared baseline."""
    state = {"letters": {}, "titles": {}}
    old_label = original.add_panel_label_fig
    old_title = original.add_external_title

    def add_panel_label(fig, ax, label, *args, **kwargs):
        result = old_label(fig, ax, label, *args, **kwargs)
        artist = fig.texts[-1]
        artist.set_gid("panel-letter-" + label)
        state["letters"][ax] = (label, artist)
        return result

    def add_title(fig, ax, text, *args, **kwargs):
        result = old_title(fig, ax, text, *args, **kwargs)
        title_artist = fig.texts[-1]
        label, letter_artist = state["letters"][ax]
        if label in "abcdef":
            title_artist.set_verticalalignment("baseline")
            letter_artist.set_verticalalignment("baseline")
            x, _ = letter_artist.get_position()
            letter_artist.set_position((x, title_artist.get_position()[1]))
            state["titles"][label] = (letter_artist, title_artist)
        return result

    original.add_panel_label_fig = add_panel_label
    original.add_external_title = add_title
    return state


def _check_panel_summary(summary, baseline, input_report: dict) -> dict:
    import pandas as pd

    if list(summary["panel"]) != ["a", "b", "c", "d", "e", "f"]:
        raise ValueError("The extended figure must retain exactly the original six panels")
    actual, expected = summary.set_index("panel"), baseline.set_index("panel")
    preserved = {}
    correction = {}
    surface_expected = {'Luna tangent tracks':461, 'Luna satellites':38, 'Luna track points':139809}
    for panel in "abcdef":
        checked = 0
        for field in expected.columns:
            old, new = expected.at[panel, field], actual.at[panel, field]
            if pd.isna(old) and pd.isna(new):
                continue
            if panel == 'e' and field in (*surface_expected, 'Luna track segments'):
                if field in surface_expected and int(new) != surface_expected[field]:
                    raise ValueError(f'Incorrect surface-only count: {field}={new}')
                if field == 'Luna track segments' and not 0 < int(new) <= int(old):
                    raise ValueError('Unexpected number of surface drawing segments')
                correction[field] = {'before':int(old),'after':int(new)}
                continue
            if pd.isna(old) or pd.isna(new) or old != new:
                raise ValueError(f"Frozen panel statistic changed: {panel}/{field}: {old!r} -> {new!r}")
            checked += 1
        preserved[panel] = {"all_original_summary_fields_unchanged": panel != "e", "nonnull_fields_checked": checked}
    wanted = {"a": input_report["north_unique_stations"], "b": input_report["south_unique_stations"],
              "d": input_report["eligible_unique_stations"]}
    for panel, count in wanted.items():
        if int(actual.at[panel, COUNT_KEY]) != count:
            raise ValueError(f"Incorrect unique 1 Hz station count in panel {panel}")
    return {"original_panels": preserved, "new_unique_station_counts": wanted, "authorized_surface_only_correction": correction}


def render(station_csv: Path, output_dir: Path, preview_width: int = 2200,
           inventory_provenance: Path | None = None,
           selection_provenance: Path | None = None) -> dict:
    if selection_provenance is not None and output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError("A thinned figure must not overwrite the full ground_1hz output directory")
    stations, input_report = load_ground_stations(station_csv, inventory_provenance)
    selection_report = load_selection_provenance(selection_provenance, stations, input_report)
    if not BASELINE_SUMMARY.is_file():
        raise FileNotFoundError(f"Frozen panel summary is required for count QA: {BASELINE_SUMMARY}")
    frozen_paths = (ROOT / "reproduce_figure.py", BASE_PATH, V5_PATH, BASELINE_SUMMARY)
    frozen_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in frozen_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mpl_cache"))
    import cartopy
    import matplotlib
    import numpy as np
    import pandas as pd
    import PIL
    from PIL import Image

    cartopy.config["pre_existing_data_dir"] = DATA / "cartopy"
    cartopy.config["data_dir"] = output_dir / "cartopy_cache"
    cartopy.config["cache_dir"] = output_dir / "cartopy_tmp"
    v5 = _load_module("ground_1hz_frozen_v5", V5_PATH)
    v5.SOURCE_TABLE = DATA / "extended_data_fig7_leoro_panel_c_sources_20260617.csv"
    table = v5.load_source_table()
    specs, statuses = v5.mission_specs_from_table(table)
    missing = [mission for mission in v5.MISSION_ORDER if mission not in statuses]
    if missing:
        raise ValueError(f"Missing frozen panel-c mission statuses: {missing}")
    original = v5.load_original_module()
    original.OUT_DIR = output_dir
    original.OUT_NATURE_PNG = output_dir / "Extended_Data_Observing_context.png"
    original.OUT_NATURE_PDF = output_dir / "not_exported.pdf"  # The frozen builder exports PNG only.
    original.OUT_TRACK_SUMMARY_CSV = output_dir / "panel_summary.csv"
    original.load_ground_roti_stations = lambda old: pd.read_csv(DATA / "global_plus_polar_gnss_roti_stations.csv", low_memory=False)
    original.load_updated_ismr_stations = lambda old: pd.read_csv(DATA / "global_plus_polar_eswua_chain_ismr_stations.csv", low_memory=False)
    original.load_current_superdarn_radars = lambda old: pd.read_csv(DATA / "global_plus_polar_current_superdarn_radars.csv", low_memory=False)
    v5.patch_panel_titles(original)
    v5.patch_orbit_panel(original, specs, statuses)
    v5.patch_verified_occultation_tracks(original)
    from surface_only_scope import install_surface_scope
    surface_selection = install_surface_scope(original, output_dir)
    header_state = _patch_panel_header_alignment(original)
    original_orbit_panel = original.draw_ro_orbit_height_panel
    def orbit_panel_slash_units(ax, *args, **kwargs):
        counts = original_orbit_panel(ax, *args, **kwargs)
        ax.set_xlabel(r"$N_e$ ($10^{12}/\mathrm{m}^{3}$)", fontsize=5.9, labelpad=1.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.48)
        return counts
    original.draw_ro_orbit_height_panel = orbit_panel_slash_units
    untouched_names = ("draw_ro_orbit_height_panel", "draw_luna_track_panel_nature", "draw_occultation_track_panel_nature")
    untouched_functions = {name: getattr(original, name) for name in untouched_names}
    state = _patch_ground_overlay(original, stations, input_report, selection_report)
    from surface_only_layout import install_layout
    layout_state = install_layout(original)
    png, _ = original.build_nature_redraw()
    summary = pd.read_csv(original.OUT_TRACK_SUMMARY_CSV)
    checks = _check_panel_summary(summary, pd.read_csv(BASELINE_SUMMARY), input_report)
    for name, function in untouched_functions.items():
        if getattr(original, name) is not function:
            raise ValueError(f"Unintended change to frozen c/e/f renderer: {name}")
    marker_checks = {}
    for panel, count in checks["new_unique_station_counts"].items():
        plotted = len(state["artists"][panel].get_offsets())
        if plotted != count:
            raise ValueError(f"Scatter/count disagreement in panel {panel}: {plotted} != {count}")
        marker_checks[panel] = {"plotted_station_offsets": plotted, "unique_station_count": count}
    # Validate the actual rendered legend text, including counts and units.
    rendered_figure = state["axes"]["a"].figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(rendered_figure.canvas, "get_renderer"):
        FigureCanvasAgg(rendered_figure)
    rendered_figure.canvas.draw()
    renderer = rendered_figure.canvas.get_renderer()
    figure_text = "\n".join(t.get_text() for t in rendered_figure.texts)
    footer_tokens = ("snapshot started", "Requested archive dates:", "Downloaded-source selection")
    if any(token in figure_text for token in footer_tokens):
        raise ValueError("A removed figure-wide footer line remains")
    legend_checks = {}
    indexed_summary = summary.set_index("panel")
    for panel, legends in state["legends"].items():
        actual_rows = [[text.get_text() for text in legend.get_texts()] for legend in legends]
        station_labels = [f"{name} ({sampling}; n={int(indexed_summary.at[panel, key])})"
                          for key, name, sampling in STATION_LEGEND
                          if int(indexed_summary.at[panel, key]) > 0]
        counts = state["full_counts"][panel]
        context_labels = []
        for key, name in [("ISR radar", "ISR"), ("Coherent radar", "SuperDARN")]:
            if int(counts.get(key, 0)):
                context_labels.append(f"{name} ({int(counts[key])})")
        if panel in ("a", "b"):
            if int(counts.get("ISR coverage", 0)):
                context_labels.append("ISR range")
            if int(counts.get("Coherent coverage", 0)):
                context_labels.append("SuperDARN fan")
        if panel == "a":
            coverage = [label for label in context_labels if label in ("ISR range", "SuperDARN fan")]
            radars = [label for label in context_labels if label not in coverage]
            expected_rows = [station_labels[:2], station_labels[2:] + radars, coverage]
        else:
            expected_rows = [station_labels[:2], station_labels[2:] + context_labels]
        expected_rows = [row for row in expected_rows if row]
        if any("\n" in label for row in actual_rows for label in row):
            raise ValueError("A station/context legend entry wraps onto two lines")
        if actual_rows != expected_rows:
            raise ValueError(f"Legend row/count mismatch in panel {panel}: {actual_rows}")
        if any(legend not in state["axes"][panel].get_children() for legend in legends):
            raise ValueError(f"A legend row is no longer attached to panel {panel}")
        boxes = [legend.get_window_extent(renderer) for legend in legends]
        if any(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i+1,len(boxes))):
            raise ValueError(f"Station/context legends overlap in panel {panel}")
        if any(box.x0 < 0 or box.y0 < 0 or box.x1 > rendered_figure.bbox.x1
               or box.y1 > rendered_figure.bbox.y1 for box in boxes):
            raise ValueError(f"Station legend is outside the figure in panel {panel}")
        column_bounds = {"a": (0.0, 60/180), "b": (60/180, 122/180), "d": (0.0, 61/180)}
        left, right = column_bounds[panel]
        if any(box.x0 < rendered_figure.bbox.width * left or box.x1 > rendered_figure.bbox.width * right for box in boxes):
            raise ValueError(f"Legend exceeds allocated column in panel {panel}")
        legend_checks[panel] = {"rows": actual_rows,
                                "row_widths_px": [box.width for box in boxes],
                                "all_entries_single_line": True,
                                "legend_rows_do_not_overlap": True,
                                "within_allocated_column": True}
    a_boxes = [legend.get_window_extent(renderer) for legend in state["legends"]["a"]]
    b_boxes = [legend.get_window_extent(renderer) for legend in state["legends"]["b"]]
    if any(a.overlaps(b) for a in a_boxes for b in b_boxes):
        raise ValueError("Northern and southern legends overlap")
    expected_unit = r"$N_e$ ($10^{12}/\mathrm{m}^{3}$)"
    if not any(ax.get_xlabel() == expected_unit for ax in rendered_figure.axes):
        raise ValueError("Electron-density slash unit was not retained")
    header_checks = {}
    if set(header_state["titles"]) != set("abcdef"):
        raise ValueError("A titled panel is missing its baseline alignment")
    for label, (letter, title) in header_state["titles"].items():
        letter_y = letter.get_transform().transform(letter.get_position())[1]
        title_y = title.get_transform().transform(title.get_position())[1]
        if abs(letter_y - title_y) > 0.01 or letter.get_va() != "baseline" or title.get_va() != "baseline":
            raise ValueError(f"Panel {label} letter/title baselines differ")
        if letter.get_window_extent(renderer).overlaps(title.get_window_extent(renderer)):
            raise ValueError(f"Panel {label} letter overlaps its title")
        header_checks[label] = {"same_baseline": True, "letter_title_do_not_overlap": True}
    orbit_axis = next(ax for ax in rendered_figure.axes if ax.get_xlabel() == expected_unit)
    if not all(spine.get_visible() for spine in orbit_axis.spines.values()):
        raise ValueError("The orbit/electron-density panel lacks a full frame")
    orbit_text = {t.get_text(): t for t in orbit_axis.texts}
    params = [t for t in orbit_axis.texts if t.get_text().startswith('$h$')]
    if len(params) != 4 or any(t.get_bbox_patch() is not None for t in params):
        raise ValueError('All four h/i annotations must have no background boxes')
    orbit_annotations = [t for t in orbit_axis.texts if t.get_text()]
    collisions = [(a.get_text(),b.get_text()) for i,a in enumerate(orbit_annotations)
                  for b in orbit_annotations[i+1:]
                  if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if collisions:
        raise ValueError(f'Orbit-panel text overlaps: {collisions}')
    from matplotlib.offsetbox import AnnotationBbox
    orbit_icons = [a for a in orbit_axis.artists if isinstance(a, AnnotationBbox)]
    text_image_collisions = [(t.get_text(),list(icon.xy)) for t in orbit_annotations for icon in orbit_icons
                             if t.get_window_extent(renderer).overlaps(icon.get_window_extent(renderer))]
    if text_image_collisions:
        raise ValueError(f'Orbit-panel text/image overlap: {text_image_collisions}')
    proximity = {}
    for label,(letter,title) in header_state['titles'].items():
        ax = layout_state['axes'][label]
        pos = ax.get_position()
        dy_mm = (letter.get_position()[1]-pos.y1)*135
        dx_mm = (letter.get_position()[0]-pos.x0)*180
        expected_dy = 4.0 if label in 'abc' else 2.5
        if abs(dy_mm-expected_dy)>1e-6 or abs(dx_mm)>1e-6:
            raise ValueError(f'Panel {label} is not close to its plotting area')
        if title.get_rotation() != 0:
            raise ValueError(f'Panel {label} title is not horizontal')
        proximity[label] = {'letter_x_offset_mm':round(dx_mm,3),'title_baseline_above_plot_mm':round(dy_mm,3)}
    orbit_annotation_checks = {'parameter_backgrounds_removed':True,'all_annotation_text_nonoverlapping':True,'annotation_text_and_images_nonoverlapping':True,
                               'MetOp_name_position':list(orbit_text['MetOp'].get_position()),
                               'TSX_name_position':list(orbit_text['TSX'].get_position())}
    from matplotlib.text import Text
    ordinary_text = [t for t in rendered_figure.findobj(Text) if t.get_text() and t.get_visible()
                     and not str(t.get_gid()).startswith("panel-letter-")]
    if any(t.get_fontsize() < 5.0-1e-6 or t.get_fontsize() > 7.0+1e-6 for t in ordinary_text):
        raise ValueError("Some text is outside the final-size 5–7 pt range")
    letters = {t.get_text():t for t in rendered_figure.texts if str(t.get_gid()).startswith("panel-letter-")}
    if any(abs(t.get_fontsize()-8.0)>1e-6 or t.get_fontweight()!='bold' for t in letters.values()):
        raise ValueError("Panel letters are not 8 pt bold")
    top_baselines = [letters[k].get_position()[1] for k in 'abc']
    plot_tops = [layout_state['axes'][k].get_position().y1 for k in 'abc']
    if max(top_baselines)-min(top_baselines)>1e-6 or max(plot_tops)-min(plot_tops)>1e-6:
        raise ValueError("Top-row letters or plot boundaries are not aligned")
    clipped_text = []
    for text in ordinary_text:
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.x1 > rendered_figure.bbox.width+0.5 or box.y0 < -0.5 or box.y1 > rendered_figure.bbox.height+0.5:
            clipped_text.append(text.get_text())
    if clipped_text:
        raise ValueError(f"Text extends beyond the fixed publication canvas: {clipped_text}")
    typography_checks = {"all_visible_text_within_canvas":True,"all_legend_rows_attached":True,"canvas_mm":[180,135],"ordinary_font_range_pt":[min(t.get_fontsize() for t in ordinary_text),max(t.get_fontsize() for t in ordinary_text)],
                         "legend_and_note_pt":5.2,"axis_titles_pt":6,"panel_titles_pt":7,"panel_letters_pt":8,
                         "top_row_letters_aligned":True,"top_row_plot_tops_aligned":True,
                         "font":"Arial","e_note_two_lines":True,"f_legend_two_columns":True}
    with Image.open(png) as image:
        size, dpi = list(image.size), list(image.info.get("dpi", ()))
        if len(dpi) != 2 or any(abs(float(d) - 600) > 0.1 for d in dpi):
            raise ValueError(f"Expected 600 dpi PNG, got {dpi}")
    for path in frozen_paths:
        if sha256_file(path) != frozen_hashes[str(path.relative_to(ROOT))]:
            raise ValueError(f"Frozen input changed while rendering: {path}")
    if sha256_file(station_csv) != input_report["csv_sha256"]:
        raise ValueError("Station CSV changed while rendering; rerun from one immutable snapshot")
    provenance_hash = input_report.get("inventory_provenance_sha256")
    if provenance_hash and sha256_file(Path(input_report["inventory_provenance"])) != provenance_hash:
        raise ValueError("Snapshot metadata changed while rendering; rerun from one immutable snapshot")
    if selection_report is not None:
        if sha256_file(Path(selection_report["path"])) != selection_report["sha256"]:
            raise ValueError("Selection provenance changed while rendering")
        if sha256_file(DEFAULT_CSV) != selection_report["input_csv_sha256"]:
            raise ValueError("The full source station CSV changed while rendering")
    qa = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point_sha256": sha256_file(Path(__file__)),
        "purpose": "Map only the 143 Ground station IDs contributing accepted epochs to current Fig.3/4; preserve the approved six-panel observing-context renderers and styling.",
        "input": input_report, "selection": selection_report, "lunar_surface_selection": surface_selection,
        "marker": {"shape": "upward triangle", "fill": "none", "edge_color": ORANGE,
            "reason": "Unfilled markers preserve visibility of co-located historical purple circles and green ISMR squares."},
        "count_checks": checks, "marker_checks": marker_checks,
        "legend_checks": legend_checks, "header_checks": header_checks, "header_proximity_mm": proximity, "orbit_annotation_checks": orbit_annotation_checks, "typography_checks": typography_checks, "panel_c_four_spines_visible": True, "footer_lines_removed": True,
        "sampling_annotations": {name: sampling for _, name, sampling in STATION_LEGEND},
        "data_renderers_preserved": list(untouched_names), "panel_c_display_change": "electron density unit 10^12/m^3 and four-sided frame", "polar_legend_counts": "Separate a/b legends, each with its own n", "frozen_file_sha256": frozen_hashes,
        "software": {"executable": sys.executable, "python": sys.version,
            **{m.__name__: m.__version__ for m in (matplotlib, cartopy, np, pd, PIL)}},
        "outputs": {"png": str(png), "size_px": size, "dpi": dpi,
            "preview_exported": False,
            "panel_summary": str(original.OUT_TRACK_SUMMARY_CSV), "pdf_exported": False},
        "visual_review": "Count/artist/export checks passed; inspect the high-resolution PNG for legend spacing before delivery.",
    }
    qa_path = output_dir / "ground_1hz_QA.json"
    temporary = qa_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(qa_path)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to results/ground_1hz; with selection provenance, results/ground_1hz_thinned")
    parser.add_argument("--inventory-provenance", type=Path,
                        help="Snapshot metadata JSON; defaults to results/ground_1hz/inventory_provenance.json")
    parser.add_argument("--selection-provenance", type=Path,
                        help="Externally selected station IDs, CSV hashes, and 5-degree Europe/Australia sampling parameters")
    parser.add_argument("--preview-width", type=int, default=2200)
    parser.add_argument("--validate-only", action="store_true", help="Check the station CSV/counts without plotting or writing outputs")
    args = parser.parse_args()
    if not 1200 <= args.preview_width <= 4000:
        parser.error("preview-width must be between 1200 and 4000 pixels")
    output_dir = args.output_dir or (DEFAULT_THINNED_OUTPUT if args.selection_provenance else DEFAULT_OUTPUT)
    if args.selection_provenance and output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("Selected figures cannot overwrite the full ground_1hz output directory")
    if (args.stations_csv.resolve() != DEFAULT_CSV.resolve() and args.output_dir is None
            and args.selection_provenance is None):
        parser.error("A custom station CSV requires --output-dir or --selection-provenance to preserve the full figure")
    if args.validate_only:
        stations, report = load_ground_stations(args.stations_csv, args.inventory_provenance)
        if args.selection_provenance:
            report["selection"] = load_selection_provenance(args.selection_provenance, stations, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        qa = render(args.stations_csv, output_dir, args.preview_width, args.inventory_provenance,
                    args.selection_provenance)
        print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit("Run reproduce_with_surface_only.py for the separate-legend figure.")

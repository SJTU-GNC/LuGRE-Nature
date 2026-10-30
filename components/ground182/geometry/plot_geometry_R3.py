"""Render the two R3 geometry figures from explicitly supplied plotting tables."""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DATA = HERE / "plot_data"
OUTPUT = HERE / "figures"
DATA_STATUS = "FINAL_SHARED_GROUND_SELECTION_GEOMETRY"
os.environ["MPLCONFIGDIR"] = str(HERE / "mplconfig")
sys.dont_write_bytecode = True
# Dependencies are installed through requirements.txt; no external project paths.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

COLORS = {"Ground": "#636C76", "LEO-RO": "#B9781A", "LuGRE": "#AA2031"}
SYSTEMS = ("Ground", "LEO-RO", "LuGRE")
BANDS = ("Equatorial", "Mid-latitude", "Polar")
ANGLE_METRICS = ("ray_B_acute_deg", "scan_Bperp_acute_deg", "magnetic_meridian_acute_deg")
ANGLE_METRIC_ALIASES = {"screen_projected_scan_B_deg": "scan_Bperp_acute_deg"}
SPEED_METRICS = ("point_speed_mps", "scan_speed_mps", "fresnel_crossing_time_s")
ANGLE_LABELS = (
    r"Ray–$\mathbf{B}$ angle, $\theta_{\rm ray}$ (°)",
    r"Scan–$\mathbf{B}_\perp$ angle, $\theta_{\rm scan}$ (°)",
    "Ray azimuth to magnetic meridian,\n" + r"$\theta_{\rm mer}$ (°)",
)
SPEED_LABELS = (
    r"Tangent-point speed, $v_{\rm p}$ (m/s)",
    r"Scan speed, $v_{\rm scan}$ (m/s)",
    "Geometric Fresnel crossing time\n" + r"$\tau_{\rm F,geom}=r_1/v_{\rm scan}$ (s)",
)
MAIN_XLIMS = ((2000., 7000.), (2000., 7000.), (0., .75))
MAIN_YLIMS = ((0., 35.), (0., 30.), (0., 20.))
GROUND_XLIMS = ((0., 220.), (0., 140.), (0., 10.))
GROUND_YLIMS = ((0., 25.0), (0., 25.0), (0., 30.))
DPI = 600
WIDTH_MM = 180.
SPEED_WIDTH_MM = 200.
SPEED_HEIGHT_MM = 210.
INSET_LABELS = (r"IPP speed, $v_{\rm p}$ (m/s)", r"Scan speed, $v_{\rm scan}$ (m/s)", "Geometric Fresnel crossing time\n"+r"$\tau_{\rm F,geom}$ (s)")

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7., "axes.titlesize": 7.5,
    "axes.labelsize": 7., "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 7., "axes.linewidth": .65,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "text.color": "#161616", "axes.labelcolor": "#161616",
    "xtick.color": "#161616", "ytick.color": "#161616",
    "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
})
try:
    FONT = font_manager.findfont("Arial", fallback_to_default=False)
except ValueError:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    FONT = font_manager.findfont("DejaVu Sans", fallback_to_default=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()


def normalize_angle_metrics(curves):
    """Resolve declared spelling aliases only; reject conflicting CDF values."""
    original = curves.copy()
    normalized = curves.copy()
    alias_counts = {old: int(original.metric.eq(old).sum()) for old in ANGLE_METRIC_ALIASES}
    normalized["metric"] = normalized["metric"].replace(ANGLE_METRIC_ALIASES)
    keys = ["metric", "system", "latitude_band", "angle_deg"]
    exact_duplicates = int(normalized.duplicated(keep="first").sum())
    normalized = normalized.drop_duplicates(keep="first")
    conflicts = normalized.duplicated(keys, keep=False)
    if conflicts.any():
        examples = normalized.loc[conflicts, keys + ["cumulative_fraction"]].head(8)
        raise ValueError("Conflicting duplicate angle metrics after alias normalization: "
                         + examples.to_json(orient="records"))
    return normalized, {
        "canonical_scan_metric": "scan_Bperp_acute_deg",
        "explicit_aliases": ANGLE_METRIC_ALIASES,
        "alias_row_counts": alias_counts,
        "identical_duplicate_rows_removed": exact_duplicates,
        "conflicting_duplicate_metrics": False,
        "input_rows": len(original), "normalized_rows": len(normalized),
        "numeric_values_changed": False,
    }


def read_frozen():
    paths = [DATA / "geometry_angle_CDF_curves_R3.csv", DATA / "geometry_speed_bins_R3.csv",
             DATA / "geometry_speed_support_R3.csv"]
    fingerprints = {p.name: sha(p) for p in paths}
    manifest_path=DATA / "GEOMETRY_FREEZE.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        for name,digest in fingerprints.items():
            expected=manifest.get("redraw_input_sha256",{}).get("data/"+name)
            if expected is not None: assert expected==digest,name
    frames = [pd.read_csv(p, float_precision="round_trip") for p in paths]
    frames[0], alias_qa = normalize_angle_metrics(frames[0])
    return (*frames, fingerprints, alias_qa)


def new_grid(height_mm, width_mm=WIDTH_MM, speed=False):
    fig, axes = plt.subplots(3, 3, figsize=(width_mm/25.4, height_mm/25.4), dpi=DPI)
    if speed:
        fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.92, wspace=.37, hspace=.62)
    else:
        fig.subplots_adjust(left=.105, right=.985, bottom=.093, top=.905, wspace=.27, hspace=.49)
    for i, ax in enumerate(axes.flat):
        ax.set_title(BANDS[i % 3], pad=5.5)
        ax.text(-.14, 1.045, chr(97+i), transform=ax.transAxes,
                fontsize=8., weight="bold", ha="left", va="bottom", clip_on=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#242424")
            spine.set_linewidth(.65)
        ax.tick_params(pad=2.2)
    return fig, axes


def verify_text_inside(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad = []
    labels = []
    for ax in [child for main in fig.axes for child in [main,*main.child_axes]]:
        labels.extend([ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts])
        labels.extend(ax.get_xticklabels()+ax.get_yticklabels())
    for text in labels:
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            bad.append(text.get_text())
    assert not bad, f"Text outside canvas: {bad}"
    # Check adjacent main-panel x labels for overlap, including two-line labels.
    overlaps = []
    main_axes = fig.axes[:9]
    for r in range(3):
        for c in range(2):
            a = main_axes[3*r+c].xaxis.label.get_window_extent(renderer)
            b = main_axes[3*r+c+1].xaxis.label.get_window_extent(renderer)
            if a.overlaps(b):
                overlaps.append((r,c))
    assert not overlaps, f"Adjacent x labels overlap: {overlaps}"
    return {"all_text_inside_canvas": True, "adjacent_main_xlabels_overlap": False}


def image_info(path, height_mm, width_mm=WIDTH_MM):
    with Image.open(path) as image:
        return {"file": path.name, "pixels": list(image.size), "dpi_metadata": list(image.info["dpi"]),
                "nominal_width_mm": width_mm, "nominal_height_mm": height_mm,
                "sha256": sha(path)}


def plot_angles(curves):
    fig, axes = new_grid(177.)
    rows = []
    for r, metric in enumerate(ANGLE_METRICS):
        for c, band in enumerate(BANDS):
            ax = axes[r,c]
            for system in SYSTEMS:
                z = curves.loc[(curves.metric == metric) & (curves.latitude_band == band)
                               & (curves.system == system)].sort_values("angle_deg")
                x = z.angle_deg.to_numpy(); y = z.cumulative_fraction.to_numpy()
                assert len(z) == 901 and np.all(np.diff(x)>0)
                assert np.all(np.diff(y)>=-1e-14) and y.min()>=0 and y.max()<=1
                line, = ax.plot(x, y, color=COLORS[system], lw=1.05, ls="-")
                assert np.array_equal(line.get_xdata(), x)
                assert np.array_equal(line.get_ydata(), y)
                rows.append({"metric":metric,"latitude_band":band,"system":system,"n":len(z),
                             "x_array_equal_source":True,"y_array_equal_source":True,
                             "x_sha256":array_sha(x),"y_sha256":array_sha(y)})
            ax.set_xlim(0,90); ax.set_ylim(0,1)
            ax.set_xticks([0,30,60,90]); ax.set_yticks([0,.5,1])
            ax.set_yticklabels(["0.0","0.5","1.0"])
            ax.set_xlabel(ANGLE_LABELS[r], labelpad=4)
            if c == 0: ax.set_ylabel("Cumulative fraction", labelpad=4.5)
    handles = [Line2D([],[],color=COLORS[s],lw=1.3,label=s,ls="-") for s in SYSTEMS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.545,.988),ncol=3,
               frameon=False,handlelength=2.25,columnspacing=1.8,handletextpad=.65)
    qa_text = verify_text_inside(fig)
    path=OUTPUT/"Extended_Data_Geometry_angles.png"
    fig.savefig(path,dpi=DPI)
    plt.close(fig)
    assert len(rows)==27
    return {"curves":rows,"curve_count":len(rows),"text_checks":qa_text,"image":image_info(path,177.)}


def hist_group(bins, metric, band, system):
    z=bins.loc[(bins.metric==metric)&(bins.latitude_band==band)&(bins.system==system)].sort_values("bin_left")
    assert len(z)>0
    left=z.bin_left.to_numpy(); right=z.bin_right.to_numpy()
    assert np.array_equal(right[:-1],left[1:])
    edges=np.r_[left,right[-1]]
    values=z.weighted_proportion_percent.to_numpy()
    counts=z.weighted_count.to_numpy()
    assert np.all(values>=0) and np.all(counts>=0)
    nz=np.flatnonzero(counts>0)
    assert len(nz)>0
    return z, edges, values, counts, (float(left[nz[0]]),float(right[nz[-1]]))


def plot_stairs(ax, edges, values, color):
    artist=ax.stairs(values,edges,fill=True,facecolor=to_rgba(color,.20),edgecolor=color,
                     linewidth=.85,linestyle="-",baseline=0)
    actual=artist.get_data()
    assert np.array_equal(actual.values,values)
    assert np.array_equal(actual.edges,edges)
    return artist


def plot_speeds(bins, support):
    fig, axes=new_grid(SPEED_HEIGHT_MM,SPEED_WIDTH_MM,speed=True)
    rows=[]; no_occlusion=[]; inset_pairs=[]
    for r,metric in enumerate(SPEED_METRICS):
        for c,band in enumerate(BANDS):
            ax=axes[r,c]
            ax.set_xlim(*MAIN_XLIMS[r]);ax.set_ylim(*MAIN_YLIMS[r])
            ax.set_xlabel(SPEED_LABELS[r],labelpad=4)
            ax.set_ylabel("Weighted proportion (%)",labelpad=4.5)
            if r<2:
                ax.set_xticks([2000,4000,6000])
                ax.set_yticks([0,10,20,30])
            else:
                ax.set_xticks([0,.25,.5,.75]);ax.set_xticklabels(["0","0.25","0.50","0.75"])
                ax.set_yticks([0,5,10,15,20])
            # Upper-right in speed rows; upper-left in the time row. Main data
            # beneath the inset rectangle are verified below, without increasing y limits.
            rect=(.53,.66,.42,.28) if r<2 else (.25,.66,.42,.28)
            inset=ax.inset_axes(rect)
            inset_pairs.append((ax,inset,metric,band))
            inset.set_facecolor("white")
            inset.set_zorder(5)
            inset.set_xlim(*GROUND_XLIMS[r]);inset.set_ylim(*GROUND_YLIMS[r])
            inset.set_title("Ground",fontsize=6.2,pad=1.7)
            inset.set_xlabel(INSET_LABELS[r],fontsize=5.7,labelpad=1.4)
            inset.set_ylabel("Weighted\nproportion (%)",fontsize=5.7,labelpad=1.2)
            inset.tick_params(labelsize=5.8,length=1.8,width=.5,pad=1)
            for spine in inset.spines.values():
                spine.set_visible(True);spine.set_color("#363636");spine.set_linewidth(.55)
            inset.set_xticks(([0,100,200],[0,60,120],[0,5,10])[r])
            inset.set_yticks([0., 10., 20.] if r < 2 else [0., 15., 30.])
            assert tuple(inset.get_ylim()) == GROUND_YLIMS[r]
            x0=MAIN_XLIMS[r][0]+rect[0]*(MAIN_XLIMS[r][1]-MAIN_XLIMS[r][0])
            x1=MAIN_XLIMS[r][0]+(rect[0]+rect[2])*(MAIN_XLIMS[r][1]-MAIN_XLIMS[r][0])
            y0=rect[1]*MAIN_YLIMS[r][1]
            max_under=0.
            for system in SYSTEMS:
                z,edges,values,counts,nz=hist_group(bins,metric,band,system)
                target=inset if system=="Ground" else ax
                plot_stairs(target,edges,values,COLORS[system])
                xlim=GROUND_XLIMS[r] if system=="Ground" else MAIN_XLIMS[r]
                ylim=GROUND_YLIMS[r] if system=="Ground" else MAIN_YLIMS[r]
                assert xlim[0]<=nz[0] and xlim[1]>=nz[1], (metric,band,system,nz,xlim)
                assert ylim[1]>values.max(), (metric,band,system,"y clipping")
                sup=support.loc[(support.metric==metric)&(support.latitude_band==band)&(support.system==system)]
                assert len(sup)==1
                assert np.isclose(counts.sum(),float(sup.valid_weight.iloc[0]),rtol=0,atol=1e-7)
                assert np.isclose(values.sum(),100.,rtol=0,atol=1e-10)
                if system!="Ground":
                    mask=(edges[:-1]<x1)&(edges[1:]>x0)
                    max_under=max(max_under,float(values[mask].max()) if mask.any() else 0.)
                rows.append({"metric":metric,"latitude_band":band,"system":system,
                             "n_bins":len(z),"edges_array_equal_source":True,"proportions_array_equal_source":True,
                             "edge_sha256":array_sha(edges),"proportions_sha256":array_sha(values),
                             "weighted_count_sum":float(counts.sum()),"proportion_sum":float(values.sum()),
                             "source_full_edge_range":[float(edges[0]),float(edges[-1])],
                             "source_nonzero_bin_range":list(nz),"view_xlim":list(xlim),"view_ylim":list(ylim),
                             "nonzero_bins_clipped":0,
                             "empty_outer_range_hidden":[max(0.,xlim[0]-float(edges[0])),max(0.,float(edges[-1])-xlim[1])]})
            assert max_under<y0, (metric,band,"inset occlusion",max_under,y0)
            no_occlusion.append({"metric":metric,"latitude_band":band,"inset_rect_axes_fraction":list(rect),
                                 "maximum_main_histogram_under_inset_percent":max_under,
                                 "inset_bottom_data_y_percent":y0,"main_data_occlusion":False})
    handles=[Patch(facecolor=to_rgba(COLORS[s],.25),edgecolor=COLORS[s],linewidth=.85,
                   label="Ground (inset)" if s=="Ground" else s) for s in SYSTEMS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.545,.988),ncol=3,
               frameon=False,handlelength=1.75,columnspacing=1.8,handletextpad=.65)
    qa_text=verify_text_inside(fig)
    renderer=fig.canvas.get_renderer()
    labelled_inset_checks=[]
    for ax,inset,metric,band in inset_pairs:
        bbox=inset.get_tightbbox(renderer)
        extent=ax.transData.inverted().transform(bbox.get_points())
        x0,y0=extent[0]; x1,y1=extent[1]
        main_ylim=ax.get_ylim();main_xlim=ax.get_xlim()
        assert x0>=main_xlim[0] and x1<=main_xlim[1], (metric,band,"inset labels outside parent horizontally",extent)
        max_under=0.
        for system in ["LEO-RO","LuGRE"]:
            _,edges,values,_,_=hist_group(bins,metric,band,system)
            mask=(edges[:-1]<x1)&(edges[1:]>x0)
            if mask.any():max_under=max(max_under,float(values[mask].max()))
        assert max_under<y0, (metric,band,"labelled inset covers main data",max_under,y0)
        labelled_inset_checks.append({"metric":metric,"latitude_band":band,
             "full_bbox_in_main_data":extent.tolist(),"main_hist_max_under_full_bbox":max_under,
             "lowest_labelled_inset_y":float(y0),"main_data_occlusion_including_labels":False})
    path=OUTPUT/"Extended_Data_Scan_speed_Fresnel_time.png"
    fig.savefig(path,dpi=DPI)
    plt.close(fig)
    assert len(rows)==27
    return {"histograms":rows,"histogram_count":len(rows),"inset_checks":no_occlusion,
            "labelled_inset_checks":labelled_inset_checks,"text_checks":qa_text,"image":image_info(path,SPEED_HEIGHT_MM,SPEED_WIDTH_MM)}


def main(speed_only=False):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    curves,bins,support,hashes,alias_qa=read_frozen()
    if speed_only:
        previous=json.loads((OUTPUT/"plot_QA.json").read_text(encoding="utf-8"))
        angles=previous["cdf"]
        assert sha(OUTPUT/angles["image"]["file"])==angles["image"]["sha256"]
    else:
        angles=plot_angles(curves)
    speeds=plot_speeds(bins,support)
    for name,digest in hashes.items():assert sha(DATA/name)==digest
    qa={"status":DATA_STATUS, "source_package_modified":False,"source_csv_sha256":hashes,
        "angle_metric_aliases":alias_qa,
        "backend":"Python/matplotlib", "matplotlib_version":mpl.__version__,"font":FONT,
        "method_variables":{"ray_angle":"theta_ray","projected_scan_angle":"theta_scan",
                            "ray_azimuth_to_magnetic_meridian":"theta_mer","point_speed":"v_p",
                            "transverse_scan_speed":"v_scan","fresnel_time":"tau_F,geom = r1/v_scan"},
        "colors":COLORS,"all_lines_solid":True,"footers_added":False,
        "scientific_recomputation":False,"scientific_rebinning":False,
        "scientific_grouping_review":{"status":"DATA_STATUS_SUPPLIED_BY_CALLER","reason":"Plotting code displays supplied tables and does not establish per-epoch scientific eligibility."},
        "presentation_arrays_status":"PASS", "visual_review":{"status":"PENDING_LAYOUT_REVIEW"},
        "data_directory":str(DATA),"speed_axis_labels":list(SPEED_LABELS),"inset_x_labels":list(INSET_LABELS),"inset_y_label":"Weighted proportion (%)","angle_png_redrawn":not speed_only,
        "cdf":angles,"speed":speeds}
    (OUTPUT/"plot_QA.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":qa["status"],"presentation_arrays_status":"PASS","CDF_curves":27,"histograms":27,
                      "outputs":[angles["image"],speeds["image"]]},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-only",action="store_true",help="Retain the angle PNG and redraw only the speed figure.")
    parser.add_argument("--data-dir",type=Path,default=DATA)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--data-status",default=DATA_STATUS)
    args=parser.parse_args()
    DATA=args.data_dir.resolve(); OUTPUT=args.output_dir.resolve(); DATA_STATUS=args.data_status
    main(args.speed_only)

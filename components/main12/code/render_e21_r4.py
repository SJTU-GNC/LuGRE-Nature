"""Stack the E21 time panels over a horizontal world map.

The accepted observations and polynomial fits are unchanged. Lobe annotations
describe reference-pattern sectors, and geomagnetic bands use the manuscript's
fixed centred-dipole pole. Only a600dpi PNG is exported for review.
"""
from pathlib import Path
from datetime import timezone
import os, sys, ast, json, hashlib
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
COMPONENT=Path(__file__).resolve().parents[1]
COMMON=COMPONENT/'data/r4/e21'
R=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'e21'
R.mkdir(parents=True,exist_ok=True)
(R/'mplconfig').mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(R / 'mplconfig')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import shapefile

DATA=COMMON
PRIOR=COMMON
PATTERN=COMMON/'reference_full_range_summary_R3.csv'
BASE=COMMON
TEMPLATE=COMPONENT/'code/e21_style.py'
HELPER=TEMPLATE
OUT=R/'figures';OUT.mkdir(exist_ok=True)
COAST=COMPONENT/'assets/cartopy/shapefiles/natural_earth/physical/ne_110m_coastline.shp'
BANDS=COMMON/'fixed_pole_magnetic_band_geographic_boundaries_0p25deg.csv'
LOBE_INTERVALS=COMMON/'E21_reference_lobe_time_intervals.csv'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def extract(path, names, scope):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

scope = dict(np=np, pd=pd, mpl=mpl, plt=plt, MultipleLocator=MultipleLocator,
             FuncFormatter=FuncFormatter, PANEL_LABEL_X=-.07, PANEL_HEADER_Y=1.026,
             MAP_PANEL_LABEL_X=-.180)
extract(TEMPLATE, {'set_panel_header', 'time_segments', 'map_extent'}, scope)
extract(HELPER, {'style_nature_axis', 'format_lon', 'format_lat', 'draw_nature_map_base'}, scope)
pieces = scope['time_segments']
panel_labels=[]
panel_titles=[]
ALIGN_X_MM=4.5
def header(ax,label,title,label_x=None):
    label_area=TextArea(label,textprops={'fontsize':8,'fontweight':'bold','color':'#111111'})
    title_area=TextArea(title,textprops={'fontsize':7,'color':'#111111'})
    packed=HPacker(children=[label_area,title_area],align='baseline',pad=0,sep=3.)
    group=AnnotationBbox(packed,(0,1),xycoords='axes fraction',xybox=(0,1.5),
                         boxcoords='offset points',box_alignment=(0,0),
                         frameon=False,pad=0,annotation_clip=False)
    group.set_clip_on(False)
    ax.add_artist(group)
    panel_labels.append(label_area._text)
    panel_titles.append(title_area._text)

contract={'revision':'R4','export':'600 dpi PNG','source':'Measured E21 C/N0 with archived independent fourth-order arc fits and calculated reference geometry.'}
save(R/'FIGURE_CONTRACT_R4.json',contract)

raw_path = DATA / 'E21_OP38_all_native_observations_with_geometry.csv'
g_path = DATA / 'E21_OP38_first_pass_1s_geometry_not_measurements.csv'
raw = pd.read_csv(raw_path)
raw['time_dt'] = pd.to_datetime(raw.time_utc, utc=True)
raw = raw[raw.time_dt.lt(pd.Timestamp('2025-03-03 15:00', tz='UTC'))].sort_values('time_dt').copy()
res = raw[raw.fit_degree.eq(4) & raw.fit_eligible & np.isfinite(raw.e23_residual_db)].copy()
assert len(raw) == 6586 and len(res) == 6542
assert raw.h_tan_km.max() <= 6000
assert not raw.duplicated(['signal_id', 'epoch_key']).any()
assert np.max(abs(res.cn0_dbhz-res.quartic_background_dbhz-res.e23_residual_db)) < 1e-12
actual_start, actual_end = raw.time_dt.min(), raw.time_dt.max()
start = pd.Timestamp('2025-03-03 11:30', tz='UTC')
end = pd.Timestamp('2025-03-03 14:00', tz='UTC')
geometry = pd.read_csv(g_path)
geometry['time_dt'] = pd.to_datetime(geometry.time_utc, utc=True)
coarse_path = DATA / 'E21_OP38_full_30s_geometry_not_measurements.csv'
coarse = pd.read_csv(coarse_path)
coarse['time_dt'] = pd.to_datetime(coarse.time_utc, utc=True)
coarse = coarse[coarse.time_dt.between(start, end) & ~coarse.time_dt.between(geometry.time_dt.min(), geometry.time_dt.max())]
geometry = pd.concat([geometry, coarse], ignore_index=True).sort_values('time_dt').drop_duplicates('time_dt').reset_index(drop=True)
geometry = geometry[geometry.time_dt.between(start, end)].copy()
assert geometry.closest_segment_fraction.between(0, 1).all()
ledger_path = BASE / 'data/all_LuGRE_E23_fit_ledger.json'
ledger = {x['arc_id']: x for x in json.loads(ledger_path.read_text())}
quantile_path = PRIOR / 'quantile_check/first_pass_signed_60s_quantiles.csv'
quant = pd.read_csv(quantile_path)
pattern = pd.read_csv(PATTERN)
bands = pd.read_csv(BANDS)
lobe_intervals = pd.read_csv(LOBE_INTERVALS)

palette = {2: '#2b5fb3', 3: '#c93f3b'}
quantile_palette = {2: '#112D59', 3: '#7F1C21'}
markers = {2: 's', 3: 'D'}
bridge = '#7b818c'
model_ls=(0,(3.2,2.2))
geometry_dots_ls=(0,(.01,3.7))
geometry_dot_lw=1.2
ion = '#008A78'
ion_alpha = .12
blocked = '#CED1D5'

def spans(g, mask):
    ix = np.flatnonzero(np.asarray(mask, bool))
    if not len(ix):
        return []
    cuts = np.r_[0, np.where(np.diff(ix)>1)[0]+1, len(ix)]
    return [(g.time_dt.iloc[ix[a]], g.time_dt.iloc[ix[b-1]]+pd.Timedelta(seconds=1)) for a, b in zip(cuts[:-1], cuts[1:])]

ion_spans = spans(geometry, geometry.h_tan_km.between(50, 1000))
blocked_spans = spans(geometry, geometry.earth_occulted)
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 7., 'axes.titlesize': 7.,
                    'axes.labelsize': 7., 'legend.fontsize': 7., 'xtick.labelsize':7.,'ytick.labelsize':7.,'savefig.dpi': 600,
                    'axes.spines.top': True, 'axes.spines.right': True})
DRAW_SCALE=180./183.
W_MM,H_MM=180.,256.5*DRAW_SCALE
LEFT_MM,WIDTH_MM=17.*DRAW_SCALE,163.*DRAW_SCALE
fig = plt.figure(figsize=(W_MM/25.4,H_MM/25.4),facecolor='white')
def box(bottom_mm,height_mm):
    return [LEFT_MM/W_MM,bottom_mm*DRAW_SCALE/H_MM,WIDTH_MM/W_MM,height_mm*DRAW_SCALE/H_MM]
aa=fig.add_axes(box(205.5,37))
ab=fig.add_axes(box(159.5,37),sharex=aa)
ac=fig.add_axes(box(113.5,37),sharex=aa)
ae=fig.add_axes(box(64.5,40),sharex=aa)
ad=fig.add_axes(box(11,37))

# Same fixed centred-dipole definition as the main Methods and other R3 maps.
pole_lat,pole_lon=np.deg2rad([80.65,-72.68])
map_lon=np.linspace(-180,180,721)
map_lat=np.linspace(-90,90,361)
xx,yy=np.meshgrid(map_lon,map_lat)
mlat=np.rad2deg(np.arcsin(np.clip(np.sin(np.deg2rad(yy))*np.sin(pole_lat)+np.cos(np.deg2rad(yy))*np.cos(pole_lat)*np.cos(np.deg2rad(xx)-pole_lon),-1,1)))
ad.set_facecolor('white')
for shp in shapefile.Reader(str(COAST)).shapes():
    xy=np.asarray(shp.points,float)
    stops=list(shp.parts)+[len(xy)]
    for a,b in zip(stops[:-1],stops[1:]):
        pts=xy[a:b]
        cuts=np.r_[0,np.where(abs(np.diff(pts[:,0]))>180)[0]+1,len(pts)]
        for i,j in zip(cuts[:-1],cuts[1:]):
            if j-i>1:
                ad.plot(pts[i:j,0],pts[i:j,1],color='#A7B0BA',lw=.38,zorder=-3)
for target in [-60,-20,20,60]:
    c=f'geographic_lat_at_mlat_{target}deg'
    ad.plot(bands.longitude_deg,bands[c],color='#8F99A6',lw=.65,ls=':',zorder=-1)
    lat=float(np.interp(132.,bands.longitude_deg,bands[c]))
    ad.text(132.,lat+2.5,rf'$\lambda_m={target:+d}^\circ$',color='#444C58',fontsize=6.5,ha='left',va='bottom',zorder=7)
def geographic_lat_at(lon,mlat):
    A=np.sin(pole_lat)
    B=np.cos(pole_lat)*np.cos(np.deg2rad(lon)-pole_lon)
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(mlat))/np.hypot(A,B))-np.arctan2(B,A))

for ax in [aa, ab, ac, ae]:
    for lo, hi in ion_spans:
        ax.axvspan(lo, hi, facecolor=ion, edgecolor='none', alpha=ion_alpha, zorder=-1)
    for lo, hi in blocked_spans:
        ax.axvspan(lo, hi, facecolor=blocked, edgecolor='none', alpha=.55, zorder=-.5)

# Panel a retains long grey dashes for its polynomial fit.
# Panels b-e use separated grey dots for their background/geometry guides.
reception_geometry = geometry[geometry.time_dt.between(actual_start,actual_end)].copy()
ac.plot(reception_geometry.time_dt, reception_geometry.h_tan_km, color=bridge, lw=geometry_dot_lw, alpha=.96, ls=geometry_dots_ls, dash_capstyle='round', zorder=2)
# Use existing calculated positions spaced along the displayed map path.
# These dots are a guide through the unobserved gap, not extra measurements.
map_xy_mm=np.column_stack((reception_geometry.tp_lon_deg.to_numpy()*WIDTH_MM/360.,
                           reception_geometry.tp_lat_deg.to_numpy()*ad.get_position().height*H_MM/180.))
map_path_mm=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(map_xy_mm,axis=0),axis=1))]
map_dot_indices=np.unique(np.searchsorted(map_path_mm,np.arange(0.,map_path_mm[-1]+1e-8,1.5)))
map_dots=reception_geometry.iloc[map_dot_indices]
ad.plot(map_dots.tp_lon_deg, map_dots.tp_lat_deg, color=bridge, ls='none', marker='o', markersize=1.2, markeredgewidth=0, alpha=.96, zorder=2)
# A broad pale-green path halo marks exactly the same height interval as the
# time-panel spans; it does not label an entire geographic latitude band.
for p in pieces(geometry[geometry.h_tan_km.between(50,1000)],90):
    ad.plot(p.tp_lon_deg,p.tp_lat_deg,color=ion,lw=12.,alpha=ion_alpha,solid_capstyle='round',zorder=1.5)
handles = []
pattern_rows = []
for signal in [2, 3]:
    q = raw[raw.signal_id.eq(signal)].sort_values('time_dt')
    rr = res[res.signal_id.eq(signal)].sort_values('time_dt')
    color = palette[signal]
    marker = markers[signal]
    aa.scatter(q.time_dt, q.cn0_dbhz, s=5.2, marker=marker, facecolor=color, edgecolor='none', alpha=.58, zorder=3)
    ab.scatter(rr.time_dt, rr.e23_residual_db, s=5.2, marker=marker, facecolor=color, edgecolor='none', alpha=.64, zorder=3)
    for p in pieces(rr, 90):
        aa.plot(p.time_dt, p.quartic_background_dbhz, color=bridge, lw=1., alpha=.9, ls=model_ls, zorder=4)
    for aid, z in rr.groupby('e23_local_arc_id'):
        meta = ledger[aid]
        t = np.arange(z.rx_gps_seconds.min(), z.rx_gps_seconds.max()+.01, 1.)
        t_obs = z.rx_gps_seconds.to_numpy(float)
        j = np.searchsorted(t_obs, t, side='right')
        left = t_obs[np.maximum(j-1, 0)]
        right = t_obs[np.minimum(j, len(t_obs)-1)]
        gap = right-left > 90
        model = np.polyval(meta['coefficients'], (t-meta['center_gps_seconds'])/meta['normalization_scale_s'])
        dt = pd.Timestamp('1980-01-06', tz='UTC') + pd.to_timedelta(t-18, unit='s')
        dense = pd.DataFrame({'time_dt': dt, 'model': model, 'gap': gap})
        for p in pieces(dense[dense.gap], 90):
            aa.plot(p.time_dt, p.model, color=bridge, lw=.95, alpha=.84, ls=model_ls, zorder=4)
    # Complete actual reception tracks, not only the ionospheric shell.
    for p in pieces(q, 90):
        ac.plot(p.time_dt, p.h_tan_km, color=color, lw=1.1, alpha=.78, zorder=5)
        ad.plot(p.tp_lon_deg, p.tp_lat_deg, color=color, lw=1.8 if signal==2 else 1.15, alpha=.72 if signal==2 else .88, zorder=5)
    reference = pattern[pattern.signal.eq('Galileo E1' if signal==2 else 'Galileo E5a')]
    gain = np.interp(geometry.tx_off_boresight_deg, reference.theta_deg, reference.median_delta_eirp_db, left=np.nan, right=np.nan)
    ae.plot(geometry.time_dt, gain, color=bridge, lw=geometry_dot_lw, ls=geometry_dots_ls, dash_capstyle='round', alpha=.96, zorder=2)
    for p in pieces(q, 90):
        y = np.interp(p.tx_off_boresight_deg, reference.theta_deg, reference.median_delta_eirp_db, left=np.nan, right=np.nan)
        ae.plot(p.time_dt, y, color=color, lw=1.05, alpha=.9, zorder=5)
    pattern_rows.extend({'signal_id':signal, 'time_utc':str(t), 'alpha_deg':float(a), 'reference_delta_EIRP_db':float(y)} for t,a,y in zip(geometry.time_dt,geometry.tx_off_boresight_deg,gain))
    shell = q[q.h_tan_km.between(50, 1000)]
    for p in pieces(shell, 90):
        ac.plot(p.time_dt, p.h_tan_km, color=color, lw=1.65, alpha=.9, zorder=6)
        ad.plot(p.tp_lon_deg, p.tp_lat_deg, color=color, lw=2.35 if signal==2 else 1.35, alpha=.72 if signal==2 else .88, zorder=6)
    first, last = q.iloc[0], q.iloc[-1]
    ad.scatter(first.tp_lon_deg, first.tp_lat_deg, s=15, marker='o', facecolor='white', edgecolor=color, lw=.8, zorder=12)
    ad.scatter(last.tp_lon_deg, last.tp_lat_deg, s=17, marker='>', facecolor=color, edgecolor='white', lw=.45, zorder=12)
    handles.append(Line2D([0], [0], marker=marker, ls='none', color=color, markeredgecolor='none', markersize=4., label='E21 Galileo '+('E1' if signal==2 else 'E5a')))

header(aa, 'a', r'Raw receiver-reported $C/N_0^k$ and fourth-order polynomial fit')
aa.set_ylabel(r'$C/N_0^k$ (dB-Hz)')
aa.set_ylim(15, 40)
aa.set_yticks([15, 20, 25, 30, 35, 40])
aa.text(pd.Timestamp('2025-03-03 12:45', tz='UTC'), 38.2, 'Earth occultation', ha='center', va='center', fontsize=7., color='#222222')
header(ab, 'b', r'$\delta C/N_0^k=C/N_0^k-\widehat{C/N_0^k}$')
ab.set_ylabel(r'$\delta C/N_0^k$ (dB)')
first_piece_end = raw[raw.time_dt.lt(pd.Timestamp('2025-03-03 12:45',tz='UTC'))].time_dt.max()
second_piece_start = raw[raw.time_dt.gt(pd.Timestamp('2025-03-03 12:45',tz='UTC'))].time_dt.min()
ab.plot([first_piece_end,second_piece_start],[0,0], color=bridge, lw=geometry_dot_lw, ls=geometry_dots_ls, dash_capstyle='round', alpha=.96, zorder=2)
ab.set_ylim(-2, 2)
ab.set_yticks([-2,-1,0,1,2])
header(ac, 'c', 'Tangent height above the WGS84 ellipsoid')
ac.set_ylabel('Tangent height (km)')
ac.set_ylim(-650, 6000)
ac.set_yticks([0, 1000, 3000, 6000])
ac.axhline(1000, c=ion, ls=':', lw=.45,alpha=.30,zorder=1)
header(ae,'d','Reference transmit pattern along the E21 path')
ae.set_ylabel('Relative EIRP (dB)')
ae.set_ylim(-30,3)
ae.set_yticks([-20,-10,0])
ae.set_xlabel('UTC on 3 March 2025 (hh:mm:ss)')
# Reference first-minimum sectors are not claims of received sidelobe data.
lobe_arrows=[]
main_lobe_labels=[]
for signal,ymain in [(2,-25.0),(3,-28.0)]:
    color=palette[signal]
    intervals=lobe_intervals[lobe_intervals.signal_id.eq(signal)]
    for _,row in intervals.iterrows():
        lo=pd.to_datetime(row.start_utc,utc=True)
        hi=pd.to_datetime(row.end_utc,utc=True)
        main=row.reference_region.startswith('Reference main')
    main=intervals[intervals.reference_region.str.startswith('Reference main')].iloc[0]
    ts=[pd.to_datetime(main.start_utc,utc=True),pd.to_datetime(main.end_utc,utc=True)]
    for t in ts:
        ae.axvline(t,color=color,lw=.65,ls=':',alpha=.75,zorder=1)
    lo,hi=ts
    center=lo+(hi-lo)/2
    ae.annotate('',xy=(hi,ymain),xytext=(lo,ymain),
                arrowprops={'arrowstyle':'<->','lw':.75,'color':color,'mutation_scale':7,'shrinkA':0,'shrinkB':0},zorder=10)
    # Compact white backgrounds keep the text clear while marking the range.
    label=ae.text(center,ymain,'E1 main lobe' if signal==2 else 'E5a main lobe',
                  fontsize=6.5,color='#111111',ha='center',va='center',zorder=12,
                  bbox={'facecolor':'white','edgecolor':'none','pad':.35})
    main_lobe_labels.append(label)
    lobe_arrows.append({'signal_id':signal,'region':'main lobe full range',
                        'tail_utc':str(lo),'head_utc':str(hi),'y_db':ymain,
                        'arrowstyle':'<->','label_y_db':ymain})
    arrow_segments=[
        ('left sidelobe',lo,start+pd.Timedelta(minutes=.2),ymain),
        ('right sidelobe',hi,end-pd.Timedelta(minutes=.2),ymain)]
    for region,tail,head,yarrow in arrow_segments:
        ae.annotate('',xy=(head,yarrow),xytext=(tail,yarrow),
                    arrowprops={'arrowstyle':'-|>','lw':.75,'color':color,'mutation_scale':4.8,'shrinkA':0,'shrinkB':0},zorder=10)
        ae.text(tail+(head-tail)/2,yarrow,'Sidelobe',color='#111111',fontsize=6.5,ha='center',va='center',zorder=12,
                bbox={'facecolor':'white','edgecolor':'none','pad':0.})
        lobe_arrows.append({'signal_id':signal,'region':region,'tail_utc':str(tail),'head_utc':str(head),'y_db':yarrow,'label_y_db':yarrow})
for ax in [aa, ab, ac, ae]:
    scope['style_nature_axis'](ax)
    ax.tick_params(axis='both',which='major',labelsize=7.)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=timezone.utc))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=timezone.utc))
    ax.grid(True, axis='x', color='#e0e4e9', lw=.42, alpha=.62)
plt.setp(aa.get_xticklabels(), visible=False)
plt.setp(ab.get_xticklabels(), visible=False)
plt.setp(ac.get_xticklabels(), visible=False)
ad.set_xlim(-180,180)
ad.set_ylim(-90,90)
ad.set_xticks(np.arange(-180,181,60))
ad.set_yticks(np.arange(-90,91,30))
def lon_degrees(v,pos):
    return f'{abs(v):.0f}°'+('W' if v<0 else 'E' if v>0 else '')
def lat_degrees(v,pos):
    return f'{abs(v):.0f}°'+('S' if v<0 else 'N' if v>0 else '')
ad.xaxis.set_major_formatter(FuncFormatter(lon_degrees))
ad.yaxis.set_major_formatter(FuncFormatter(lat_degrees))
scope['style_nature_axis'](ad)
ad.tick_params(axis='both',which='major',labelsize=7.)
ad.grid(True,color='#D9DEE5',lw=.35,alpha=.65,zorder=-5)
ad.set_xlabel('Longitude (°)',fontsize=7.,labelpad=2.)
ad.set_ylabel('Latitude (°)',fontsize=7.,labelpad=2.)
header(ad, 'e', 'Tangent-point track and geomagnetic-latitude bands', label_x=-.07)
direction_rows=[]
for label,t0,t1 in [('Ingress','2025-03-03 12:24+00:00','2025-03-03 12:30+00:00'),('Egress','2025-03-03 13:00+00:00','2025-03-03 13:12+00:00')]:
    p0=geometry.iloc[np.argmin(abs(geometry.time_dt-pd.Timestamp(t0)))]
    p1=geometry.iloc[np.argmin(abs(geometry.time_dt-pd.Timestamp(t1)))]
    # Offset each arrow to run parallel to the path without hiding either signal.
    xy0=(float(p0.tp_lon_deg)+12.,float(p0.tp_lat_deg))
    xy1=(float(p1.tp_lon_deg)+12.,float(p1.tp_lat_deg))
    ad.annotate('',xy=xy1,xytext=xy0,arrowprops={'arrowstyle':'-|>','lw':.8,'color':'#48515D','mutation_scale':6.5,'shrinkA':0,'shrinkB':0},zorder=15)
    ad.text((xy0[0]+xy1[0])/2+9.,(xy0[1]+xy1[1])/2,label,ha='left',va='center',fontsize=7.,color='#222222',zorder=15)
    arrow_mm=float(np.linalg.norm((ad.transData.transform(xy1)-ad.transData.transform(xy0))/fig.dpi*25.4))
    direction_rows.append({'label':label,'time_from_utc':t0,'time_to_utc':t1,'tail_lon_lat':xy0,'head_lon_lat':xy1,'arrow_length_mm':arrow_mm,'shrinkA':0,'shrinkB':0,'mutation_scale':6.5})
handles.extend([
    Line2D([0], [0], color=bridge, lw=.95, ls=model_ls, label='Fourth-order polynomial fit'),
    Patch(facecolor=ion, alpha=ion_alpha, label='50–1,000 km'),
])
legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=4,fontsize=7.,frameon=False,handlelength=1.8,columnspacing=1.2,borderaxespad=0.)
for ax in [aa,ab,ac,ae,ad]:
    ax.yaxis.set_label_coords((ALIGN_X_MM-LEFT_MM)/WIDTH_MM,.5)
    ax.yaxis.label.set_fontsize(7.)
    ax.yaxis.label.set_horizontalalignment('center')
    ax.yaxis.label.set_verticalalignment('center')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.6)
        spine.set_edgecolor('#262626')
    ax.tick_params(which='both',width=.6)
ae.get_xticklabels()[-1].set_ha('right')
ad.get_xticklabels()[-1].set_ha('right')
for txt in fig.findobj(mpl.text.Text):
    txt.set_fontsize(txt.get_fontsize()+2.)
path = OUT / 'Extended_Data_CN0_detrending_OP38_E21.png'
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
alignment=[]
for ax,letter,title in zip([aa,ab,ac,ae,ad],panel_labels,panel_titles):
    lb=letter.get_window_extent(renderer)
    tb=title.get_window_extent(renderer)
    yb=ax.yaxis.label.get_window_extent(renderer)
    alignment.append({'letter':letter.get_text(),'title':title.get_text(),
                      'letter_center_x_mm':float((lb.x0+lb.x1)/2/fig.bbox.width*W_MM),
                      'ylabel_center_x_mm':float((yb.x0+yb.x1)/2/fig.bbox.width*W_MM),
                      'letter_title_gap_pt':float((tb.x0-lb.x1)/fig.dpi*72),
                      'letter_bbox_px':list(lb.bounds),'title_bbox_px':list(tb.bounds)})
fonts=[{'text':t.get_text(),'pt':float(t.get_fontsize())} for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
outside=[]
for txt in fig.findobj(mpl.text.Text):
    if not txt.get_visible() or not txt.get_text():continue
    bb=txt.get_window_extent(renderer)
    if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:
        outside.append(txt.get_text())
fig.savefig(path,dpi=600,facecolor='white')
plt.close(fig)

raw.to_csv(R / 'E21_all_actual_first_pass.csv', index=False)
pd.DataFrame(pattern_rows).to_csv(R/'E21_reference_EIRP_geometry_only.csv',index=False)
qa = {
    'status': 'REPRODUCED_R4_E21',
    'figure': str(path), 'raw_n': len(raw), 'qualified_quartic_n': len(res),
    'per_signal_raw_n': {str(k): int(v) for k, v in raw.groupby('signal_id').size().items()},
    'reference_sidelobe_count': int(sum((raw[raw.signal_id.eq(s)].tx_off_boresight_deg>edge).sum() for s, edge in [(2,24),(3,28)])),
    'raw_range_dbhz': [float(raw.cn0_dbhz.min()), float(raw.cn0_dbhz.max())],
    'residual_range_db': [float(res.e23_residual_db.min()), float(res.e23_residual_db.max())],
    'actual_height_range_km': [float(raw.h_tan_km.min()), float(raw.h_tan_km.max())],
    'actual_footprint_lon_range_deg': [float(raw.tp_lon_deg.min()), float(raw.tp_lon_deg.max())],
    'actual_footprint_lat_range_deg': [float(raw.tp_lat_deg.min()), float(raw.tp_lat_deg.max())],
    'axes_a_dbhz': [15,40], 'axes_b_db': [-2,2], 'axes_c_km': [-650,6000], 'axes_d_EIRP_db':[-30,3],
    'd_y_ticks':[float(v) for v in ae.get_yticks()],
    'raw_display_n':sum(len(c.get_offsets()) for c in aa.collections),
    'detrended_display_available_fit_n':sum(len(c.get_offsets()) for c in ab.collections),
    'CN0_display_threshold_applied':False,
    'raw_points_below25_n':int(raw.cn0_dbhz.lt(25).sum()),
    'raw_points_below20_n':int(raw.cn0_dbhz.lt(20).sum()),
    'no_existing_fit_raw_n':int(len(raw)-len(res)),
    'no_existing_fit_reason': '44 actual measurements below50km have no archived fit; shown in a,c,d,e without inventing a residual',
    'production_scientific_population_unchanged':True,
    'a_all_raw_values_inside_display':bool(raw.cn0_dbhz.between(15,40).all()),
    'b_display_zoom_only_no_quality_exclusion':True,
    'b_existing_fit_values_outside_display_n':int((~res.e23_residual_db.between(-2,2)).sum()),
    'b_outside_display_by_signal':{str(k):int((~z.e23_residual_db.between(-2,2)).sum()) for k,z in res.groupby('signal_id')},
    'unfiltered_values_preserved_in_screening_figure':True,
    'occultation_spans_utc': [[str(a),str(b)] for a,b in blocked_spans],
    'ionosphere_spans_utc': [[str(a),str(b)] for a,b in ion_spans],
    'time_range_utc': [str(start),str(end)],
    'source_hashes': {str(p): sha(p) for p in [raw_path,g_path,coarse_path,ledger_path,quantile_path,PATTERN,TEMPLATE,HELPER,COAST,BANDS,LOBE_INTERVALS]},
    'layout':'One aligned column, a-d time panels over horizontal world map e, as requested; previous colors/markers/frame/header style preserved.',
    'all_axes_left_and_width_mm':[LEFT_MM,WIDTH_MM],
    'map_height_mm':37.*DRAW_SCALE,
    'map_height_previous_mm':81.5,
    'd_height_mm':40.*DRAW_SCALE,
    'a_b_c_height_mm':37.*DRAW_SCALE,
    'map_height_unchanged_since_selected_raw_display':False,
    'all_five_panel_heights_equal':False,
    'd_height_increment_since_equal_version_mm':3.*DRAW_SCALE,
    'font_pt':{'ordinary':9.,'secondary':8.5,'panel_letters':10.},
    'global_font_increment_pt':2.,
    'font_increment_since_previous_candidate_pt':1.,
    'font_values_pt':sorted(set(t['pt'] for t in fonts)),
    'font_text_above7pt':[t for t in fonts if t['pt']>7],
    'font_text_above8pt':[t for t in fonts if t['pt']>8],
    'font_text_above9pt':[t for t in fonts if t['pt']>9],
    'panel_letter_and_ylabel_alignment':alignment,
    'ylabel_alignment_reference_x_mm':ALIGN_X_MM,
    'panel_headers':'Letters grouped with titles above axes,3pt gap and common text baseline; not aligned with y-axis labels',
    'plotting_axes_positions':[list(ax.get_position().bounds) for ax in [aa,ab,ac,ae,ad]],
    'magnetic_band_fills':False,
    'geographic_units':'Degrees shown in tick labels and axis labels',
    'UTC_format':'hh:mm:ss',
    'legend_items':[t.get_text() for t in legend.get_texts()],
    'legend_rows':1,
    'reference_lobe_annotation':'For each frequency, the central double-headed main-lobe arrow and outward sidelobe arrows share one lower row below the reference curves; all labels centered inline.',
    'reference_lobe_arrows':lobe_arrows,
    'lobe_label_background':'white',
    'lobe_arrow_line_style':'solid',
    'fit_dash_colour':bridge,
    'dashed_lines_reserved_for':'Panel a fourth-order polynomial fit',
    'b_c_d_grey_guide_style':{'line_pattern':[.01,3.7],'linewidth_pt':1.2,'capstyle':'round','point_period_mm':3.71*1.2/72.*25.4},
    'c_d_reference_geometry_line_style':'Clearly separated round-point line',
    'e_calculated_track_style':{'marker':'o','markersize_pt':1.2,'nominal_path_spacing_mm':1.5,'joining_line':False,'displayed_point_n':len(map_dots)},
    'magnetic_and_lobe_boundary_line_style':'dotted',
    'b_solid_zero_lines_removed':True,
    'b_remaining_line_count':len(ab.lines),
    'b_line_styles':[line.get_linestyle() for line in ab.lines],
    'figure_mm':[W_MM,H_MM],
    'c_e_geometry_display_interval_utc':[str(reception_geometry.time_dt.min()),str(reception_geometry.time_dt.max())],
    'actual_reception_interval_utc':[str(actual_start),str(actual_end)],
    'c_e_geometry_outside_actual_reception_n':int((~reception_geometry.time_dt.between(actual_start,actual_end)).sum()),
    'c_e_end_extrapolations_removed':True,
    'd_full_reference_geometry_interval_unchanged':True,
    'd_e_spacing_increment_mm':3.*DRAW_SCALE,
    'd_e_axes_gap_mm':16.5*DRAW_SCALE,
    'approved_baseline_physical_heights_mm':{'a':24.*DRAW_SCALE,'b':24.*DRAW_SCALE,'c':24.*DRAW_SCALE,'d':40.*DRAW_SCALE,'e':55.*DRAW_SCALE},
    'new_physical_heights_mm':{k:(40. if k=='d' else 37.)*DRAW_SCALE for k in 'abcde'},
    'uniform_frame_pt':.6,
    'text_outside_canvas':outside,
    'reference_lobe_interval_source':str(LOBE_INTERVALS),
    'magnetic_bands':{'pole_lat_deg':80.65,'pole_lon_deg':-72.68,'boundaries_deg':[-60,-20,20,60],'latitude_axis_is_geographic':True},
    'no_horizontal_height_shading':True,
    'height_highlight_ED3_match':{'colour':ion,'alpha':ion_alpha,'map_style':'Path-local halo at calculated50-1000km positions, not a latitude-band assignment'},
    'direction_arrows':direction_rows,
    'main_P05_P95_removed':True,
    'main_low_CN0_removed':False,
    'b_gap_guide':'Dotted zero-reference background guide; not an interpolated residual measurement.',
    'EIRP_basis':'Galileo reference median directionality evaluated at the calculated alpha, relative to boresight; not measured instantaneous E21 transmitter power.',
    'original_display_limits_expanded_for_full_data': True,
    'new_exclusions_or_refits': False,
    'display_mask_removed': 'Previous25dB-Hz and fit-eligibility display masks; raw measurement files and archived fits unchanged'
}
assert qa['a_all_raw_values_inside_display'] and qa['raw_display_n']==6586 and qa['detrended_display_available_fit_n']==6542 and qa['reference_sidelobe_count']==0
assert sha(raw_path)=='10a582f00292b954be7e8bffa6d2743692f8c18031a5662426bb194407bec802'
save(R / 'FIGURE_QA_R4.json', qa)
print(json.dumps({'status':qa['status'],'figure':str(path),'text_outside_canvas':outside,'direction_arrows':direction_rows,'d_y_ticks':qa['d_y_ticks']}, ensure_ascii=False))

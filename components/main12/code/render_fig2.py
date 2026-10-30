"""Recompose source-backed Fig. 2 panels at publication size; PNG export only."""
from pathlib import Path
import os, sys, json, math, hashlib, shutil
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
COMPONENT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'fig2'
ROOT.mkdir(parents=True,exist_ok=True)
PROJECT = COMPONENT
DATA = COMPONENT/'data/fig2'
os.environ['MPLCONFIGDIR'] = str(ROOT/'mplconfig')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, NullLocator, FuncFormatter
import cartopy, cartopy.crs as ccrs, cartopy.feature as cfeature
from PIL import Image
cartopy.config['data_dir'] = COMPONENT/'assets/cartopy'
cartopy.config['pre_existing_data_dir'] = COMPONENT/'assets/cartopy'
from polar_helper import draw_polar

W,H=183.,170.
DPI=600
C={'Equatorial':'#238BA3','Mid-lat':'#7957C5','Polar':'#AA2031'}
F={'Equatorial':'#E2F1F4','Mid-lat':'#EEE8F7','Polar':'#F4E1E4'}
ORDER=['Equatorial','Mid-lat','Polar']
MARK={'Equatorial':'o','Mid-lat':'s','Polar':'^'}
TEXT='#22262B'
LAYER_COLOURS={'D':'#F6F0E4','E':'#E9F1F7','F':'#F0ECF7','Topside':'#F6EBEF'}
LAYER_ALPHA=.24
SRC=PROJECT/'R3_main_figures_20260920/source_data'
LOCAL=DATA
paths={'amplitude':DATA/'spatial_amplitude_cells.csv','roti':DATA/'spatial_roti_cells.csv','psd':DATA/'fig3_panel_a_psd_no_time_interpolation.csv','paired':DATA/'panel_c_paired_frequency_segments.csv','paired_geometry':DATA/'paired_geometry.csv','events':DATA/'fig3_panel_b_event_level_points.csv','height':DATA/'fig3_panel_d_altitude_layer_heatmap.csv','north':DATA/'OP76_G01_measured_samples.csv','south':DATA/'OP38_E21_measured_samples.csv'}

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
hashes={k:sha(p) for k,p in paths.items()}
data={k:pd.read_csv(p) for k,p in paths.items()}
assert len(data['amplitude'])>0 and len(data['roti'])==455
assert len(data['paired'])>0 and len(data['events'])>0
assert data['height'].events.sum()==len(data['events'])
assert data['paired'].n_measured_pairs.ge(180).all()

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
 'font.size':7.0,'axes.labelsize':7.0,'axes.titlesize':7.2,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
 'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.55,'axes.edgecolor':TEXT,
 'text.color':TEXT,'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT,
 'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.5,'ytick.major.width':.5,
 'xtick.minor.size':1.2,'ytick.minor.size':1.2,'xtick.minor.width':.35,'ytick.minor.width':.35,
 'xtick.major.pad':1.5,'ytick.major.pad':1.5,'legend.frameon':False,'legend.fontsize':6.3,
 'figure.facecolor':'white','savefig.facecolor':'white','figure.dpi':DPI})
fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=DPI)
axes={};checks={};headers=[];case_legends={};panel_letters={};map_colourbars={}
PANEL_LABEL={'a':'a','b':'g','c':'j','d':'d','e':'h','f':'k','g':'b','j':'c','h':'e','k':'f','i':'i'}
def rect(x,y,w,h):return [x/W,y/H,w/W,h/H]
def add(key,x,y,w,h,**kwargs):
    ax=fig.add_axes(rect(x,y,w,h),**kwargs);axes[key]=ax;return ax
def heading(x,y,key,title):
    panel_letters[key]=fig.text(x/W,y/H,PANEL_LABEL[key],fontsize=8.5,weight='bold',va='bottom')
    headers.append(panel_letters[key])
    headers.append(fig.text((x+4.0)/W,y/H,title,fontsize=7.2,va='bottom'))
def clean(ax,grid='y'):
    for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.55);sp.set_color(TEXT)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis=grid,which='major',color='#E1E5E8',lw=.35)
def guides(ax):
    lon,lat=np.meshgrid(np.arange(-180,181),np.arange(-90,91))
    mlat=np.rad2deg(np.arcsin(np.clip(np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(80.65))+
        np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(80.65))*np.cos(np.deg2rad(lon+72.68)),-1,1)))
    ax.contour(lon,lat,mlat,levels=[-60,-20,0,20,60],transform=ccrs.PlateCarree(),colors='#7C858D',linewidths=.38,linestyles='--',zorder=7)
    for m,ly in [(60,62.5),(20,23.5),(0,2.5),(-20,-22.5),(-60,-62.5)]:
        ax.text(-165,ly,rf'$\lambda_m={m}^\circ$',transform=ccrs.PlateCarree(),fontsize=5.8,color='#515B64',
          bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.08),zorder=8)
def global_map(key,y,table,vmax,ticks,title):
    ax=add(key,6.5,y,56.8,29.5,projection=ccrs.Robinson());ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#F5F6F7',edgecolor='none')
    ax.gridlines(crs=ccrs.PlateCarree(),linewidth=.2,color='#D8DDE2',xlocs=range(-180,181,60),ylocs=range(-60,61,30))
    grid=np.full((36,72),np.nan)
    for r in table.itertuples(): grid[int(r.lat_bin),int(r.lon_bin)]=r.metric
    cmap=mpl.colormaps['viridis'].copy();cmap.set_bad((1,1,1,0))
    m=ax.pcolormesh(np.arange(-180,185,5),np.arange(-90,95,5),np.ma.masked_invalid(grid),
      transform=ccrs.PlateCarree(),cmap=cmap,vmin=0,vmax=vmax,zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#8F99A3',linewidth=.3,zorder=5)
    guides(ax);ax.spines['geo'].set_linewidth(.5);ax.spines['geo'].set_edgecolor('#89929B')
    cb_y=130.2 if key=='a' else 84
    cb=fig.colorbar(m,cax=fig.add_axes(rect(12.15,cb_y,45.5,1.3)),orientation='horizontal')
    map_colourbars[key]=cb
    cb.set_ticks(ticks,labels=[str(t) if t!=vmax else f'≥{t}' for t in ticks]);cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=6.0,length=1.5,pad=1)
    heading(2.5,118 if key=='d' else 166,key,title)
    checks[key]=dict(cells=len(table),limits=[0,vmax],source_values_unchanged=True,
                    colourbar_orientation='horizontal',map_aspect_preserved=True,title_orientation='horizontal above')

global_map('a',134.5,data['amplitude'],4,[0,1,2,3,4],r'$|\delta C/N_0|$ (dB)')
global_map('d',86.4,data['roti'],30,[0,10,20,30],'ROTI (TECU/min)')

for key,bottom,pair,title in [('b',48,'L1 / E1',r'$\delta C/N_0$ power spectrum (L1/E1)'),('c',7.7,'L5 / E5a',r'$\delta C/N_0$ power spectrum (L5/E5a)')]:
    ax=add(key,10.5,bottom,45.5,27.0 if key=='b' else 27.5)
    for (signal,lat),q in data['psd'][data['psd'].asd_pair.eq(pair)].groupby(['signal_name','lat_group']):
        q=q.sort_values('frequency_hz')
        line,=ax.plot(q.frequency_hz,q.median_psd_dbhz2_per_hz,color=C[lat],lw=.9,ls='-' if signal.startswith('GPS') else (0,(3,1.8)))
        np.testing.assert_array_equal(line.get_ydata(),q.median_psd_dbhz2_per_hz)
    ax.set(xscale='log',yscale='log',xlim=(.004,.5),ylim=(.001,2),xticks=[.01,.1,.5],yticks=[.001,.01,.1,1],xlabel='Frequency (Hz)')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r'PSD (dB$^2$/Hz)',labelpad=1)
    ax.set_xlabel('Frequency (Hz)',labelpad=1)
    clean(ax,'both');heading(2.5,bottom+29,key,title)
latitude_handles=[Line2D([],[],color=C[g],lw=1.1,label={'Mid-lat':'Mid-latitude'}.get(g,g)) for g in ORDER]
system_handles=[Line2D([],[],color=TEXT,lw=.9,label='GPS'),Line2D([],[],color=TEXT,lw=.9,ls=(0,(3,1.8)),label='Galileo')]
# Legend columns are filled top-to-bottom, preserving latitude labels in the top row.
handles=[latitude_handles[0],system_handles[0],latitude_handles[1],system_handles[1],latitude_handles[2]]
spectral_legend=axes['b'].legend(handles=handles,loc='lower left',bbox_to_anchor=(.012,.035),ncol=3,
  handlelength=1.2,columnspacing=.7,handletextpad=.25,fontsize=5.9,borderaxespad=0,borderpad=.1,labelspacing=.38,
  frameon=True,facecolor='white',edgecolor='none',framealpha=.92)
checks['psd']={'curves':12,'spectral_points':len(data['psd']),'all_values_unchanged':True}

ax=add('e',74,47.5,45.5,27.5)
p=data['paired'];xcol='high_width_60sample_block_median_db';ycol='low_width_60sample_block_median_db'
p=p.drop(columns=['mlat_median','median_latitude_band','latitude_bands_present'],errors='ignore').merge(data['paired_geometry'][['joint_run_id','mlat_median','median_latitude_band','latitude_bands_present']],on='joint_run_id',how='left',validate='1:1')
assert p.mlat_median.notna().all() and len(p)==len(data['paired'])
p['display_latitude_band']=np.where(p.mlat_median.abs()<20,'Equatorial',np.where(p.mlat_median.abs()<60,'Mid-latitude','Polar'))
assert p.display_latitude_band.eq(p.median_latitude_band).all()
assert p.display_latitude_band.isin(['Equatorial','Mid-latitude','Polar']).all()
for g in ['Equatorial','Mid-latitude','Polar']:
    k='Mid-lat' if g=='Mid-latitude' else g;q=p[p.display_latitude_band.eq(g)]
    scatter=ax.scatter(q[xcol],q[ycol],s=11 if k=='Polar' else 9,marker=MARK[k],color=C[k],alpha=.85,edgecolors='white',lw=.25,zorder=3)
    np.testing.assert_array_equal(scatter.get_offsets(),q[[xcol,ycol]].to_numpy())
ax.plot([.1,10],[.1,10],ls=':',lw=.6,color='#89919B',zorder=1)
ax.set(xscale='log',yscale='log',xlim=(.1,10),ylim=(.1,10),xticks=[.1,.3,1,3,10],yticks=[.1,.3,1,3,10],
       xlabel='L1/E1 width (dB)',ylabel='L5/E5a width (dB)')
for axis in [ax.xaxis,ax.yaxis]:axis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));axis.set_minor_formatter(NullFormatter())
ax.set_xlabel('L1/E1 width (dB)',labelpad=1);ax.set_ylabel('L5/E5a width (dB)',labelpad=1)
leg=[Line2D([],[],ls='none',marker=MARK[g],mfc=C[g],mec='none',ms=3.1,label=label)
   for g,label in [('Equatorial','Equatorial'),('Mid-lat','Mid-latitude'),('Polar','Polar')]]
ax.legend(handles=leg,loc='upper left',labelspacing=.18,handletextpad=.15,handlelength=.7,borderpad=.1,fontsize=5.9)
clean(ax,None);heading(69,77,'e','Paired-frequency response')
p.to_csv(ROOT/'Fig2_R3_paired_display_source_v21.csv',index=False)
checks['paired']={'segments':len(p),'blocks':int(p.n_exact_60sample_blocks.sum()),'all_scatter_coordinates_unchanged':True,
 'display_latitude_rule':'absolute segment median centered-dipole magnetic latitude: <20 equatorial,20-60 midlatitude,>=60 polar',
 'boundary_crossing_segments_retained':int(p.latitude_band.eq('Mixed').sum()),'display_group_counts':p.display_latitude_band.value_counts().to_dict(),
 'interpretation':'Representative segment colour, not a claim that every sample stays in that latitude band'}

ax=add('f',74,7.7,45.5,27.5)
vals=[data['events'].loc[data['events'].lat_group.eq(g),'signed_p95_p05_width_db'].to_numpy() for g in ORDER]
parts=ax.violinplot(vals,positions=np.arange(3),widths=.68,showmeans=False,showextrema=False)
rng=np.random.default_rng(2)
for i,(g,v,body) in enumerate(zip(ORDER,vals,parts['bodies'])):
    body.set_facecolor(F[g]);body.set_edgecolor(C[g]);body.set_alpha(.8);body.set_linewidth(.6)
    ax.scatter(i+rng.normal(0,.04,len(v)),v,s=3,color=C[g],alpha=.23,lw=0)
    q1,med,q3=np.quantile(v,[.25,.5,.75]);ax.plot([i,i],[q1,q3],color=TEXT,lw=.8);ax.plot([i-.13,i+.13],[med,med],color=TEXT,lw=.9)
ax.set(xticks=[0,1,2],xticklabels=['Equatorial','Mid-latitude','Polar'],ylim=(0,11),yticks=[0,2,4,6,8,10],ylabel='P95–P05 width (dB)')
ax.set_ylabel('P95–P05 width (dB)',labelpad=1.5)
clean(ax);heading(69,36.7,'f','Event fluctuation widths')
ax.grid(False,axis='y',which='both')
checks['violin']={'event_windows':len(data['events']),'group_counts':dict(zip(ORDER,map(len,vals))),'maximum_db':float(max(v.max() for v in vals)),'horizontal_gridlines_removed_by_user_request':True,'full_axes_frame_and_category_ticks_retained':True}

polar_info=[];polar_mappable=None
for key,left,hemi,case in [('g',76.25,'north','e'),('j',131.25,'south','f')]:
    ax,polar_mappable,polar_audit=draw_polar(fig,rect(left,123,41,41),hemi,case)
    axes[key]=ax;polar_info.append(polar_audit)
    heading(left-7.25,166,key,f'{hemi.capitalize()} polar tracks '+r'$|\delta C/N_0|$ (dB)')
cb=fig.colorbar(polar_mappable,cax=fig.add_axes(rect(177,132,1.35,23.1)),spacing='uniform')
cb.set_ticks([0,.1,.5,1,2,4],labels=['0','0.1','0.5','1','2','≥4'])
cb.ax.tick_params(labelsize=6.0,length=1.6,pad=1);cb.outline.set_linewidth(.4)
checks['polar_colourbar']={'panels':['b','c'],'normalization':'original nonuniform BoundaryNorm','colormap':'viridis','independent_of_panel_a':True,'polar_diameter_mm':41,'track_linewidth_pt':.88,'height_mm':23.1,'location':'right of c','label':'absolute delta C/N0 (dB) in both horizontal polar track titles; numeric colourbar ticks only'}

for key,left,which,title in [('h',74,'north','North: OP76 G01\n15 Mar 2025'),('k',129,'south','South: OP38 E21\n3 Mar 2025')]:
    y=88
    ax=add(key,left,y,45.5,26)
    table=data[which]
    for layer,lo,hi in [('D',50,90),('E',90,150),('F',150,600),('Topside',600,1000)]:
        ax.axvspan(lo,hi,color=LAYER_COLOURS[layer],alpha=LAYER_ALPHA,lw=0,zorder=0)
        ax.text(math.sqrt(lo*hi),1.025,layer,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=5.8,color='#5D6671')
    for boundary in [90,150,600]:ax.axvline(boundary,color='#E6E8EB',lw=.3,zorder=1)
    for signal,q in table.groupby('freq_group',sort=False):
        color='#AA2031' if signal.endswith(('L5','E5a')) else '#245C92'
        s=ax.scatter(q.lugre_h_tan_km,q.cn0_dbhz,color=color,s=1.6,alpha=.9,lw=0,label=signal.replace('GPS ','').replace('Galileo ',''))
        np.testing.assert_array_equal(s.get_offsets(),q[['lugre_h_tan_km','cn0_dbhz']].to_numpy())
    ax.set(xscale='log',xlim=(50,1000),ylim=(22,43),xticks=[50,90,150,300,600,1000],xticklabels=['50','90','150','300','600','1,000'],yticks=[25,30,35,40])
    ax.xaxis.set_minor_formatter(NullFormatter());ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Tangent height (km)',labelpad=1)
    side='left' if key=='h' else 'right'
    ax.yaxis.set_ticks_position(side);ax.yaxis.set_label_position(side)
    ax.tick_params(axis='y',left=side=='left',labelleft=side=='left',right=side=='right',labelright=side=='right',pad=2)
    ax.set_ylabel(r'$C/N_0$ (dB-Hz)',rotation=90,va='center',ha='center')
    ax.yaxis.set_label_coords(-.14 if side=='left' else 1.14,.5)
    case_legends[key]=ax.legend(loc='upper left',ncol=2,handlelength=.6,handletextpad=.2,columnspacing=.8,markerscale=1.8,fontsize=6.2)
    clean(ax)
    panel_letters[key]=fig.text((left-5)/W,118/H,PANEL_LABEL[key],fontsize=8.5,fontweight='bold',va='bottom')
    fig.text(left/W,118/H,title.replace('\n',' | '),fontsize=6.5,ha='left',va='bottom')
    checks[key]={'measured_samples':len(table),'all_sample_coordinates_unchanged':True,'signal_counts':table.freq_group.value_counts().to_dict()}
    checks[key].update(display_panel=PANEL_LABEL[key],title_orientation='horizontal above panel',y_axis=side,operation_phase='OP76' if key=='h' else 'OP38')

ax=add('i',129,7.7,45.5,67.3)
layers=['Topside','F2','F1','E','D'];alt=data['height']
HEIGHT_GROUP_OFFSET=.22
BAR_ALPHA=.84
height_display=[];height_bars=[];height_value_labels=[]
for i,layer in enumerate(layers):
    colour_layer='F' if layer in ('F1','F2') else layer
    ax.axhspan(-.9 if i==0 else i-.5,4.55 if i==4 else i+.5,
               color=LAYER_COLOURS[colour_layer],alpha=LAYER_ALPHA,lw=0,zorder=0)
    for guide in [20,40,60]:
        ax.vlines(guide,i-.39,i+.39,color='#D9DFE4',lw=.35,linestyle=(0,(1.5,2.5)),zorder=1)
    for j,g in enumerate(ORDER):
        r=alt[alt.height_layer.eq(layer)&alt.lat_group.eq(g)].iloc[0]
        yy=i+(j-1)*HEIGHT_GROUP_OFFSET
        if r.total_windows:
            value=float(r.event_fraction_percent)
            bar=ax.barh(yy,value,height=.18,left=0,color=C[g],alpha=BAR_ALPHA,edgecolor='none',zorder=2).patches[0]
            assert bar.get_width()==value
            height_bars.append(bar)
            height_value_labels.append(ax.text(80,yy,f'{value:.0f}%',
                        va='center',ha='right',fontsize=6.3,color=C[g]))
        else:
            ax.text(3,yy,'No polar data' if g=='Polar' else 'No data',va='center',ha='left',fontsize=6.0,color='#68727C')
        height_display.append(dict(height_layer=layer,lat_group=g,events=int(r.events),total_windows=int(r.total_windows),event_fraction_percent=float(r.event_fraction_percent)))
for boundary in [.5,1.5,2.5,3.5]:ax.axhline(boundary,color='#BAC4CE',lw=.45,zorder=1)
ax.set(xticks=[0,20,40,60],xlim=(-3,83),ylim=(4.55,-.9),yticks=range(len(layers)),yticklabels=layers)
ax.yaxis.tick_right()
ax.set_xlabel('Event windows (%)',labelpad=2)
clean(ax,None)
ax.xaxis.grid(False)
ax.vlines(0,-.5,4.55,color='#DEE3E8',lw=.4,zorder=0)
ax.tick_params(axis='y',left=False,labelleft=False,right=True,labelright=True,length=2,pad=1.5,labelsize=6.2)
ax.yaxis.get_major_ticks()[0].set_pad(4.0)
topside_label=ax.get_yticklabels()[0]
topside_label.set_rotation(90)
topside_label.set_rotation_mode('anchor')
topside_label.set_ha('center')
topside_label.set_va('center')
height_legend=ax.legend(handles=[Patch(facecolor=C[g],edgecolor='none',alpha=BAR_ALPHA,label={'Mid-lat':'Mid-latitude'}.get(g,g)) for g in ORDER],
 loc='upper center',bbox_to_anchor=(.5,.991),ncol=3,fontsize=5.9,columnspacing=.6,handlelength=1.2,handleheight=.5,handletextpad=.25,borderpad=.15,borderaxespad=.15,
 frameon=False,facecolor='none',edgecolor='none')
heading(124,77,'i','Occurrence by tangent height')
pd.DataFrame(height_display).to_csv(ROOT/'Fig2_R3_height_occurrence_source_v21.csv',index=False)
checks['height']={'valid_windows':int(alt.total_windows.sum()),'events':int(alt.events.sum()),'categories':15,'all_counts_preserved':True,'display':'grouped horizontal bars from zero to exact observed percentages; aligned right-side percentage column; faint 20/40/60-percent guides within each height group; right-side layer names with Topside rotated90 degrees; shared faint layer backgrounds; transparent one-row inset legend over continuous Topside background; clear full-width height boundaries','empty_polar_D':'No polar data',
                  'layer_colours':LAYER_COLOURS,'layer_alpha':LAYER_ALPHA,'F1_F2_share_F_colour':True,'categorical_group_offset':HEIGHT_GROUP_OFFSET,
                  'bars_equal_original_percentages':True,'sampled_categories_drawn':len(height_bars),'zero_fraction_label_retained':True,'unsampled_category_has_no_bar':True}

checks['new_residual_display_ranges']={
      'event_widths_above_11_db':int((data['events'].signed_p95_p05_width_db>11).sum()),
      'paired_points_outside_0p1_to_10_db':int((~p[['high_width_60sample_block_median_db','low_width_60sample_block_median_db']].apply(lambda z:z.between(.1,10)).all(axis=1)).sum()),
      'mapping_colour_saturation_at_4_db_is_display_only':True}
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
pos={k:ax.get_position().bounds for k,ax in axes.items()}
checks['alignment']={
 'right_column_case_and_occurrence_aligned':bool(np.allclose(pos['k'][0::2],pos['i'][0::2])),
 'middle_column_left_right_aligned':bool(all(np.allclose(pos['h'][0::2],pos[k][0::2]) for k in ['e','f'])),
 'left_column_spectral_axes_aligned':bool(np.allclose(pos['b'][0::2],pos['c'][0::2])),
 'lower_first_row_top_aligned':bool(all(np.isclose(pos[k][1]+pos[k][3],pos['i'][1]+pos['i'][3]) for k in ['b','e'])),
 'i_bottom_aligned_with_j_k':bool(all(np.isclose(pos['i'][1],pos[k][1]) for k in ['c','f'])),
 'polar_pair_aligned':bool(np.allclose(pos['g'][1::2],pos['j'][1::2])),
 'case_pair_aligned':bool(np.allclose(pos['h'][1::2],pos['k'][1::2])),
 'polar_case_centres_aligned':bool(all(np.isclose(pos[p][0]+pos[p][2]/2,pos[q][0]+pos[q][2]/2) for p,q in [('g','h'),('j','k')])),
 'lower_last_row_top_aligned':bool(np.isclose(pos['c'][1]+pos['c'][3],pos['f'][1]+pos['f'][3])),
 'row1_letters_aligned':len({panel_letters[k].get_position()[1] for k in ['a','g','j']})==1,
 'row2_case_letters_aligned':panel_letters['h'].get_position()[1]==panel_letters['k'].get_position()[1],
 'row2_d_e_f_letters_aligned':len({panel_letters[k].get_position()[1] for k in ['d','h','k']})==1,
 'd_header_above_fixed_map':bool(panel_letters['d'].get_position()[1]>pos['d'][1]+pos['d'][3]),
 'row3_letters_aligned':len({panel_letters[k].get_position()[1] for k in ['b','e','i']})==1,
 'row4_letters_aligned':panel_letters['c'].get_position()[1]==panel_letters['f'].get_position()[1]}
checks['alignment']['d_map_and_colourbar_shifted_down_together']=bool(np.isclose(map_colourbars['d'].ax.get_position().y0*H,84) and np.isclose(axes['d'].get_position(original=True).y0*H,86.4))
assert all(checks['alignment'].values()), {k:v for k,v in checks['alignment'].items() if not v}
checks['layout_spacing_mm']={'polar_circle_diameter':41,'panel_heights_mm':{'b_c':41,'e_f':26,'g':27,'j':27.5,'h_k':27.5,'i':67.3},'quantitative_panel_width_mm':45.5,
                            'j_k_shifted_down_from_v14_mm':1.8,'d_shifted_down_from_v14_mm':4,'a_map_and_colourbar_unchanged':True}
checks['panel_relocation_old_to_new']=PANEL_LABEL
for key in ['h','k']:
    labelbox=axes[key].yaxis.label.get_window_extent(renderer)
    overlaps=[t.get_text() for t in axes[key].get_yticklabels() if t.get_visible() and labelbox.overlaps(t.get_window_extent(renderer))]
    checks[f'{PANEL_LABEL[key]}_axis_title_tick_overlap']=overlaps
    assert not overlaps
for key,legend in case_legends.items():
    bb=legend.get_window_extent(renderer)
    q=data['north' if key=='h' else 'south']
    n_under=sum(int(bb.contains(*xy)) for xy in axes[key].transData.transform(q[['lugre_h_tan_km','cn0_dbhz']].to_numpy()))
    checks[key].update(legend_location='upper left',sample_points_under_legend=n_under)
    assert n_under==0
legend_box=spectral_legend.get_window_extent(renderer)
line_points_under_legend=sum(int(legend_box.contains(*xy)) for line in axes['b'].lines for xy in axes['b'].transData.transform(np.column_stack([line.get_xdata(),line.get_ydata()])))
checks['spectral_legend']={'location':'inside panel g, lower left, two rows','data_points_under_legend':line_points_under_legend}
assert line_points_under_legend==0
height_legend_box=height_legend.get_window_extent(renderer)
topside_bbox=topside_label.get_window_extent(renderer)
checks['height']['legend_frameless']=not height_legend.get_frame_on()
checks['height']['topside_label_rotated_deg']=float(topside_label.get_rotation())
checks['height']['topside_label_clear_of_axes']=bool(topside_bbox.x0>axes['i'].get_window_extent(renderer).x1)
checks['height']['topside_label_clear_of_tick']=bool(topside_bbox.x0>axes['i'].get_window_extent(renderer).x1+2*DPI/72)
checks['height']['topside_label_inside_canvas']=bool(topside_bbox.x1<fig.bbox.x1 and topside_bbox.y0>0 and topside_bbox.y1<fig.bbox.y1)
checks['height']['topside_label_overlaps_other_labels']=any(topside_bbox.overlaps(label.get_window_extent(renderer)) for label in axes['i'].get_yticklabels()[1:])
assert checks['height']['legend_frameless'] and checks['height']['topside_label_inside_canvas'] and checks['height']['topside_label_clear_of_axes'] and checks['height']['topside_label_clear_of_tick']
assert not checks['height']['topside_label_overlaps_other_labels']
height_points_under_legend=sum(int(height_legend_box.contains(*axes['i'].transData.transform(
    (row['event_fraction_percent'],layers.index(row['height_layer'])+(ORDER.index(row['lat_group'])-1)*HEIGHT_GROUP_OFFSET))))
    for row in height_display if row['total_windows'])
checks['height']['legend_points_obscured']=height_points_under_legend
assert height_points_under_legend==0
checks['height']['legend_bars_obscured']=sum(int(height_legend_box.overlaps(bar.get_window_extent(renderer))) for bar in height_bars)
assert checks['height']['legend_bars_obscured']==0
checks['height']['value_labels_overlap_bars']=sum(int(label.get_window_extent(renderer).overlaps(bar.get_window_extent(renderer))) for label in height_value_labels for bar in height_bars)
assert checks['height']['value_labels_overlap_bars']==0
critical_texts=list(fig.texts)+[t for ax in fig.axes for t in [ax.xaxis.label,ax.yaxis.label]]
for t in fig.findobj(mpl.text.Text):
    if not t.get_visible() or not t.get_text().strip():continue
    bb=t.get_window_extent(renderer)
    if bb.width and bb.height and (bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5):
        # Matplotlib generates unused, clipped off-range log ticks; they are not rendered.
        if t.get_clip_on() and t not in critical_texts:continue
        outside.append(t.get_text())
out=ROOT/'Fig2_R3_E23_h50_detrending.png'
fig.savefig(out,dpi=DPI,facecolor='white',pil_kwargs={'compress_level':6})
with Image.open(out) as im:
    size=im.size;dpi=im.info.get('dpi');mode=im.mode
    if mode=='RGBA':im.convert('RGB').save(out,dpi=(DPI,DPI),compress_level=6)
qa={'status':'PASS' if not outside else 'CHECK_TEXT','version':'R3_E23_method_h50_refit_fixed_cohorts','dimensions_mm':[W,H],'size_pixels':size,'dpi':dpi,
 'outside_text':outside,'source_values_recomputed':False,'frozen_current_source_tables':True,'panels':checks,'polar':polar_info,
 'source_hashes_unchanged':all(sha(paths[k])==v for k,v in hashes.items()),
 'four_sided_numeric_axes':all(all(ax.spines[x].get_visible() for x in ['left','right','top','bottom']) for key,ax in axes.items() if key in ['b','c','e','f','h','i','k']),
 'k_horizontal_gridlines_hidden':not any(line.get_visible() for line in axes['f'].get_ygridlines()),
 'exported_formats':['PNG'],'sha256':sha(out)}
assert qa['source_hashes_unchanged']
qa['earlier_selected_png_preserved']=True
qa['method_update']='Redraw current h50 E23-method source tables. Scientific statistics were recomputed in the original h50 processing recorded under provenance; this plotting entry point preserves those fixed cohorts and values.'
(ROOT/'Fig2_R3_E23_h50_detrending_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'Fig2_R3_E23_h50_detrending_source_manifest.json').write_text(json.dumps({k:{'path':str(p),'sha256':hashes[k]} for k,p in paths.items()},indent=2),encoding='utf8')
(ROOT/'Fig2_R3_E23_h50_detrending_panel_mapping.json').write_text(json.dumps(PANEL_LABEL,indent=2),encoding='utf8')
plt.close(fig)
print(json.dumps(qa,ensure_ascii=False,indent=2))

"""Numerical redraw of current Figure 5 from frozen case inputs.

The accepted reference was assembled in PowerPoint and later patched in place.
This renderer retains the original plotting functions and all source values,
and assembles editable Matplotlib axes instead of reusing reference pixels.
"""
from pathlib import Path
import importlib.util, json, sys, types, struct

ROOT=Path(__file__).resolve().parents[1]

def _natural_earth_shim():
    """Use the included Natural Earth vectors, without network downloads."""
    class Feature:
        def __init__(self,name):self.path=ROOT/'data/natural_earth'/name
        def with_scale(self,scale):
            if scale!='110m':raise ValueError(scale)
            return self
        def geometries(self):
            # Natural Earth coast/border SHP records are polylines. Read their
            # vertices directly; DBF attributes and external GIS libraries are
            # not required by the original map functions.
            with self.path.open('rb') as stream:
                stream.read(100)
                records=[]
                while True:
                    header=stream.read(8)
                    if not header:break
                    _,words=struct.unpack('>ii',header);body=stream.read(words*2)
                    kind=struct.unpack_from('<i',body)[0]
                    if kind==0:continue
                    if kind not in (3,5,13,15,23,25):raise ValueError(('Unsupported SHP type',kind))
                    nparts,npoints=struct.unpack_from('<ii',body,36)
                    parts=list(struct.unpack_from('<'+'i'*nparts,body,44))+[npoints]
                    points=list(struct.iter_unpack('<dd',body[44+nparts*4:44+nparts*4+npoints*16]))
                    records.append((parts,points))
            for parts,points in records:
                for a,b in zip(parts[:-1],parts[1:]):
                    yield types.SimpleNamespace(geom_type='LineString',coords=points[a:b])
    feature=types.ModuleType('cartopy.feature')
    feature.COASTLINE=Feature('ne_110m_coastline.shp')
    feature.BORDERS=Feature('ne_110m_admin_0_boundary_lines_land.shp')
    cartopy=types.ModuleType('cartopy');cartopy.feature=feature;cartopy.config={}
    sys.modules['cartopy']=cartopy;sys.modules['cartopy.feature']=feature

def _load():
    _natural_earth_shim()
    path=ROOT/'code/fig5_sources/velocity_source.py'
    spec=importlib.util.spec_from_file_location('portable_fig5_velocity',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def render(output):
    import numpy as np,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    source=_load();v37=source.v37;north=v37.north;south=v37.south
    current=pd.read_csv(ROOT/'data/Fig5_LuGRE_h50_current.csv',float_precision='round_trip')
    assert len(current)==1521
    assert current.hemisphere.value_counts().to_dict()=={'south':930,'north':591}
    assert current.fit_degree.eq(4).all() and current.fit_eligible.astype(bool).all()
    def profiles(case):
        hemi='north' if int(case['track_index'])==145 else 'south'
        frames=[]
        # Actual tracking segments stay separate in height/time curves. A
        # shared signal label must not join observations across tracking gaps.
        for (label,segment),d in current[current.hemisphere.eq(hemi)].groupby(['label','segment_id'],sort=False,dropna=False):
            d=d.copy();t=pd.to_datetime(d['time'],utc=True,format='mixed',errors='coerce')
            t=t.fillna(pd.to_datetime(d['_utc_s'],unit='s',utc=True))
            color=d['color'].dropna().iloc[0] if d['color'].notna().any() else ('#2F80ED' if ' E1' in label or ' L1' in label else '#22A766')
            marker='o' if ' E1' in label or ' L1' in label else '^'
            frames.append(pd.DataFrame({'time':t,'height':d.lugre_h_tan_km,'dcn0':d.e23_residual_db,
                'label':label,'color':color,'marker':marker}).sort_values('time'))
        return frames
    north.lugre_profiles=profiles;south.south_lugre_profiles=profiles
    # Suppress redundant cache exports by the historical loaders. Their data
    # are supplied directly and remain read-only inside this package.
    v37._ROLLING_POINT_CACHE=pd.read_csv(v37.ROLLING_POINTS_SOURCE)
    v37._ROLLING_POINT_CACHE['time_utc']=pd.to_datetime(v37._ROLLING_POINT_CACHE.time_utc,utc=True,format='mixed')
    v37._HIGHRATE_L1_CACHE=pd.read_csv(v37.HIGHRATE_POINTS_SOURCE)
    v37._HIGHRATE_L1_CACHE['time_utc']=pd.to_datetime(v37._HIGHRATE_L1_CACHE.time_utc,utc=True,format='mixed')
    cases=[v37.with_pfisr_site_context(north.load_case(145)),south.load_south_case(9)]
    corrected=pd.read_csv(ROOT/'data/Fig5_southern_FIR_current.csv')
    fan=cases[1]['fan'].merge(corrected[['beam','range_gate','mean_abs_velocity_ms']],on=['beam','range_gate'],how='left',validate='one_to_one',suffixes=('_old',''))
    assert fan.mean_abs_velocity_ms.notna().sum()==corrected.mean_abs_velocity_ms.notna().sum()
    cases[1]['fan']=fan
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.65,'mathtext.fontset':'dejavusans'})
    # More vertical space separates the source-rendered panels and their
    # legends. This changes assembly only, not any scientific axes or values.
    fig=plt.figure(figsize=(4203/600,8.0),dpi=600,facecolor='white')
    def clear_titles(ax):
        for loc in ('left','center','right'):
            ax.set_title('',loc=loc)
    def panel_letter(ax,value):
        ax.text(-.16,1.075,value,transform=ax.transAxes,fontsize=9,fontweight='bold')
    def compact_legend(ax):
        handles,labels=ax.get_legend_handles_labels()
        existing=ax.get_legend()
        # The source amplitude panels pass proxy handles directly to legend;
        # those handles do not appear in get_legend_handles_labels().
        if not handles and existing is not None:
            handles=list(getattr(existing,'legend_handles',getattr(existing,'legendHandles',[])))
            labels=[text.get_text() for text in existing.get_texts()]
        if existing is not None:existing.remove()
        if handles:
            ax.legend(handles,labels,loc='upper left',ncol=1,fontsize=4.8,
                frameon=False,handletextpad=.2,labelspacing=.12,
                borderaxespad=.25,handlelength=1.1,markerscale=.8)
    def compact_time_annotation(ax,hemi):
        for text in ax.texts:
            if 'OP76' in text.get_text() or 'OP38' in text.get_text():
                text.set_text('15 March 2025\nOP76' if hemi=='north' else '3 March 2025\nOP38')
                text.set_fontsize(4.9);text.set_linespacing(1.05)
                text.set_position((.97,.96));text.set_ha('right');text.set_va('top')
    map_axes=[]
    for hemi,case,x in zip(['north','south'],cases,[.035,.545]):
        ax=fig.add_axes([x,.565,.42,.36]);map_axes.append(ax)
        draw=north.draw_map if hemi=='north' else south.draw_south_map
        v37.draw_map_without_duplicate_colorbars(draw,ax,fig,case,'')
        clear_titles(ax)
        # The original map header is an ax.text artist in data coordinates,
        # separate from all three standard axes-title locations. Remove only
        # that OP/time header; geographic and station labels stay intact.
        for text in list(ax.texts):
            value=text.get_text()
            if '|' in value and ('OP76' in value or 'OP38' in value):
                text.remove()
        if hemi=='south':
            for text in ax.texts:
                if text.get_text().strip() in ('LEO-RO','LuGRE'):
                    text.set_ha('right')
    v37.add_shared_map_colorbars(fig,x=.493,y0=.655,w=.008)
    map_axes[0].text(-.01,1.055,'a',transform=map_axes[0].transAxes,fontsize=10,fontweight='bold')
    map_axes[1].text(-.01,1.055,'b',transform=map_axes[1].transAxes,fontsize=10,fontweight='bold')
    map_axes[0].set_title('OP76 G01: northern polar night\n15 March 2025, 16:24–16:29 UTC',fontsize=7.4,pad=13)
    map_axes[1].set_title('OP38 E21: southern polar day\n3 March 2025, 12:23–12:32 UTC',fontsize=7.4,pad=13)
    fig.legend(handles=north.map_legend_handles(),loc='center',bbox_to_anchor=(.50,.535),
        ncol=3,frameon=False,fontsize=5.4,handletextpad=.3,columnspacing=.9)
    # Four height/amplitude panels, retaining the actual current LuGRE samples
    # and the accepted rolling-median/high-rate LEO comparison data.
    lefts=[.068,.288,.568,.788];width=.168
    for i,(case,hemi,panel) in enumerate([(cases[0],'north','c'),(cases[1],'south','m')]):
        ax=fig.add_axes([lefts[2*i],.303,width,.18])
        (north.draw_height_time if hemi=='north' else south.draw_south_height_time)(ax,case,'')
        clear_titles(ax);compact_time_annotation(ax,hemi)
        ax.set_title('Tangent height',loc='left',fontsize=6.5,pad=7)
        ax.tick_params(labelsize=5.8);ax.set_ylabel('Tangent height (km)',fontsize=6.0,labelpad=3)
        ax.set_xlabel('UTC',fontsize=6.1,labelpad=3)
        if hemi=='south':
            lo,hi=ax.get_xlim();ax.set_xticks(np.linspace(lo,hi,4))
        panel_letter(ax,'c' if hemi=='north' else 'e')
        ax=fig.add_axes([lefts[2*i+1],.303,width,.18])
        v37.draw_dcn0_panel_with_highrate_background(ax,case,'',panel=panel,
            lugre_func=profiles,leoro_func=north.leoro_profiles if hemi=='north' else south.south_leoro_profiles)
        clear_titles(ax);compact_legend(ax)
        ax.set_title('Amplitude fluctuations',loc='left',fontsize=6.0,pad=7)
        ax.tick_params(labelsize=5.8);ax.set_ylabel('')
        ax.set_xlabel(r'$\delta C/N_0$ (dB)',fontsize=6.1,labelpad=3)
        panel_letter(ax,'d' if hemi=='north' else 'f')
    for i,(case,hemi,index) in enumerate([(cases[0],'north',145),(cases[1],'south',9)]):
        ax=fig.add_axes([lefts[2*i],.055,width,.15])
        v37.draw_yunyao_ne_panel(ax,index,'',include_pfisr_context=hemi=='north')
        clear_titles(ax);compact_legend(ax)
        ax.set_title('Electron density',loc='left',fontsize=6.5,pad=8)
        ax.tick_params(labelsize=5.8);ax.set_ylabel('Height (km)',fontsize=6.0,labelpad=3)
        ax.xaxis.label.set_fontsize(6.1);ax.xaxis.labelpad=3
        panel_letter(ax,'g' if hemi=='north' else 'i')
        ax=fig.add_axes([lefts[2*i+1],.055,width,.15],projection='polar')
        source.draw_velocity_fan(ax,source.velocity_case(case),hemi)
        clear_titles(ax)
        ax.set_title('LOS velocity',loc='center',fontsize=6.5,pad=19)
        ax.tick_params(labelsize=5.6)
        for child in ax.child_axes:
            child.tick_params(labelsize=4.5,pad=1,length=1.5)
            child.xaxis.label.set_fontsize(5.1)
            child.xaxis.labelpad=3
        ax.text(-.16,1.25,'h' if hemi=='north' else 'j',transform=ax.transAxes,fontsize=9,fontweight='bold')
    # The source code predates the accepted typography revision. Normalize
    # body labels while leaving the numerical artists and original axes intact.
    for artist in fig.findobj(Text):
        if artist.get_text() not in list('abcdefghij'):
            artist.set_fontweight('normal')
        artist.set_fontfamily('DejaVu Sans')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=600,facecolor='white')
    plt.close(fig)
    return {'figure':'Fig5','reproduction_class':'scientific_numerical_redraw',
        'accepted_full_composite_pixel_exact':False,'LuGRE_north_epochs':591,'LuGRE_south_epochs':930,
        'southern_velocity_statistic':'mean(abs(v_LOS))','private_YunYao_profiles':2,
        'note':'All scientific panels redrawn from included numerical inputs; PowerPoint assembly and font rasterization are not claimed pixel-exact.'}

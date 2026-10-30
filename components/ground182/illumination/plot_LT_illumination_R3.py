"""R3 final common selection fractional-reporting CN0 IPP SZA/magnetic-latitude scatter; every accepted window shown."""
from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('LUGRE_FIGURE_OUTPUT',str(ROOT.parent/'outputs'))).resolve();OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'nearspace'))
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from PIL import Image
import solar_geometry_R3 as solar

PLATFORMS=['Ground','LEO-RO','LuGRE']
BANDS=['Equatorial','Mid-latitude','Polar']
COLORS={'Equatorial':'#238BA3','Mid-latitude':'#7957C5','Polar':'#AA2031'}
MARKERS={'Equatorial':'o','Mid-latitude':'s','Polar':'^'}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def frame(ax):
    ax.tick_params(direction='in',top=True,right=True,length=2.6,width=.65,pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_linewidth(.65);spine.set_color('#343A40')

def dawn_limits(suns,latitudes,height=350):
    """Exact solar-centre/WGS84 crossing LT for each latitude and observed solar state."""
    if len(suns)>256:
        result=[dawn_limits(suns[i:i+256],latitudes,height) for i in range(0,len(suns),256)]
        return np.min([x[0] for x in result],axis=0),np.max([x[1] for x in result],axis=0)
    # The rotational ellipsoid permits rotation of every Sun vector to longitude zero.
    suns=np.c_[np.hypot(suns[:,0],suns[:,1]),np.zeros(len(suns)),suns[:,2]]
    latitude=np.deg2rad(np.asarray(latitudes))[None,:]
    rad=6378.137/np.sqrt(1-(1-(solar.AXES_KM[2]/solar.AXES_KM[0])**2)*np.sin(latitude)**2)
    rho=(rad+height)*np.cos(latitude)
    z=(rad*(solar.AXES_KM[2]/solar.AXES_KM[0])**2+height)*np.sin(latitude)
    shape=(len(suns),len(latitudes))
    def blocked(lt):
        lon=(lt-12)*np.pi/12
        xyz=np.stack([rho*np.cos(lon),rho*np.sin(lon),np.broadcast_to(z,shape)],axis=-1)
        u=suns[:,None,:]-xyz
        return solar.ray_blocked(xyz.reshape(-1,3),u.reshape(-1,3)).reshape(shape)
    lo=np.zeros(shape);hi=np.full(shape,12.)
    night=blocked(lo);noon=blocked(hi)
    for _ in range(30):
        mid=(lo+hi)/2
        shadow=blocked(mid)
        lo=np.where(shadow,mid,lo);hi=np.where(shadow,hi,mid)
    dawn=(lo+hi)/2
    dawn=np.where(~night,0,dawn)
    dawn=np.where(noon,12,dawn)
    return dawn.min(axis=0),dawn.max(axis=0)

def main():
    sources=[ROOT/'ground/Ground_solar_windows_R3.parquet',ROOT/'nearspace/nearspace_solar_windows_R3.parquet']
    g=pd.read_parquet(sources[0]);near=pd.read_parquet(sources[1])
    qa_source=json.loads((ROOT/'ground/Ground_expanded_solar_QA_R3.json').read_text(encoding='utf8'))
    n_ground=qa_source['Ground_windows']
    assert len(g)==n_ground
    expected_near=json.loads((ROOT.parent/'baseline_source_contract_R3.json').read_text(encoding='utf8'))['nearspace_window_counts']
    assert near.groupby('platform').size().to_dict()==expected_near
    data=pd.concat([g,near],ignore_index=True)
    assert data.groupby('platform').size().to_dict()=={'Ground':n_ground,**expected_near}
    assert np.isfinite(data[['plot_solar_local_time_h','plot_sza_deg','plot_latitude_deg','plot_height_km','width_db']]).all().all()
    assert data.width_db.ge(0).all() and data.n_points.between(30,60).all()
    ground_shadow_sza=90+np.degrees(np.arccos(6371/(6371+350)))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7.1,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'text.color':'#22272B','axes.labelcolor':'#22272B','figure.facecolor':'white','axes.facecolor':'white'})
    fig,axes=plt.subplots(3,2,figsize=(183/25.4,190/25.4))
    fig.subplots_adjust(left=.115,right=.975,bottom=.075,top=.905,wspace=.38,hspace=.42)
    band_legend=fig.legend(handles=[Line2D([],[],marker=MARKERS[b],color=COLORS[b],linestyle='-',markersize=3.6,
        linewidth=1.1,label=b) for b in BANDS],loc='center left',bbox_to_anchor=(.13,.966),borderaxespad=0,
        ncol=3,frameon=False,fontsize=7.4,handlelength=1.5,columnspacing=1.6,handletextpad=.4)
    norm=mpl.colors.Normalize(0,4,clip=False);cmap=mpl.colormaps['viridis'].with_extremes(over='#FDE725')
    paths={b:MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform()) for b,m in MARKERS.items()}
    hist_rows=[];scatter_checks=[];scatters=[]
    for row,platform in enumerate(PLATFORMS):
        q=data.loc[data.platform.eq(platform)]
        left,right=axes[row]
        ymax=0
        for band in BANDS:
            v=q.loc[q.latitude_band.eq(band),'plot_solar_local_time_h'].to_numpy()
            counts,_=np.histogram(v,bins=np.arange(25));proportion=counts/len(v)*100
            assert counts.sum()==len(v) and abs(proportion.sum()-100)<1e-10
            left.stairs(proportion,np.arange(25),color=COLORS[band],linewidth=1.2)
            ymax=max(ymax,float(proportion.max()))
            hist_rows.extend(dict(platform=platform,latitude_band=band,LT_bin_start_h=i,n_windows=int(c),proportion_pct=float(p)) for i,(c,p) in enumerate(zip(counts,proportion)))
        observed_peak=ymax
        ymax={'Ground':10,'LEO-RO':25,'LuGRE':60}[platform]
        if row==0:ground_ymax=float(ymax)
        assert observed_peak<ymax
        left.set(xlim=(0,24),ylim=(0,ymax),xticks=[0,6,12,18,24],
            xlabel='IPP solar local time (h)' if row==0 else 'Tangent solar local time (h)',ylabel='Proportion per 1-h bin (%)')
        ystep={'Ground':2,'LEO-RO':5,'LuGRE':10}[platform]
        left.set_yticks(np.arange(0,ymax+.001,ystep))
        left.grid(False,axis='both')
        ordered=q.sample(frac=1,random_state=20261027+row)
        if row==0:
            right.axvspan(ground_shadow_sza,180,color='#E3E6E9',lw=0,zorder=0)
            right.axvline(ground_shadow_sza,color='#89939D',lw=.7,ls='--',zorder=2)
            right.axvline(90,color='#B2B9C0',lw=.55,ls=':',zorder=1)
            xx=ordered.plot_sza_deg
            yy=ordered.representative_mlat_centered_dipole_deg
            assert np.isfinite(yy).all()
            magnetic_band=np.where(np.abs(yy)<20,'Equatorial',np.where(np.abs(yy)<60,'Mid-latitude','Polar'))
            assert np.array_equal(magnetic_band,ordered.latitude_band.to_numpy())
            a=right.scatter(xx,yy,c=ordered.width_db,cmap=cmap,norm=norm,s=.45,
                edgecolors='none',linewidths=0,alpha=1,zorder=3)
            a.set_paths([paths[b] for b in ordered.latitude_band])
            ordered[['window_id','source_unit','satellite','signal','latitude_band','n_points',
                'representative_utc_s','plot_sza_deg','representative_mlat_centered_dipole_deg',
                'width_db','representative_sunlit']].to_csv(OUT/'Ground_IPP_magnetic_SZA_scatter_points_R3.csv',index=False)
            right.set(xlim=(0,180),ylim=(-90,90),xticks=[0,30,60,90,120,150,180],yticks=[-90,-60,-30,0,30,60,90],
                yticklabels=['90°S','60°S','30°S','0°','30°N','60°N','90°N'],
                xlabel='IPP solar zenith angle (°)',ylabel='IPP geomagnetic latitude')
            for x,txt in [(.03,'Sunlit'),(.97,'Earth shadow')]:
                right.text(x,.97,txt,transform=right.transAxes,ha='left' if x<.1 else 'right',
                    va='top',fontsize=6.3,color='#58636E',zorder=6)
        else:
            top=300 if platform=='LEO-RO' else 1000
            heights=np.linspace(50,top,500)
            boundary=90+np.degrees(np.arccos(6371/(6371+heights)))
            right.fill_betweenx(heights,boundary,180,color='#E3E6E9',lw=0,zorder=0)
            right.plot(boundary,heights,color='#89939D',lw=.7,ls='--',zorder=2)
            right.axvline(90,color='#B2B9C0',lw=.55,ls=':',zorder=1)
            xx=ordered.plot_sza_deg;yy=ordered.plot_height_km
            a=right.scatter(xx,yy,c=ordered.width_db,cmap=cmap,norm=norm,s=6.2,
                edgecolors='white',linewidths=.13,alpha=1,zorder=3)
            a.set_paths([paths[b] for b in ordered.latitude_band])
            right.set(xlim=(0,180),ylim=(50,top),xticks=[0,30,60,90,120,150,180],
                yticks=[50,100,200,300] if top==300 else [50,300,600,1000],
                xlabel='Tangent solar zenith angle (°)',ylabel='Tangent height (km)')
            right.text(.03,.97,'Sunlit',transform=right.transAxes,ha='left',va='top',fontsize=6.3,color='#58636E',zorder=6)
            right.text(.97,.97,'Earth shadow',transform=right.transAxes,ha='right',va='top',fontsize=6.3,color='#58636E',zorder=6)
        assert np.array_equal(np.asarray(a.get_offsets()),np.c_[xx,yy])
        assert np.array_equal(np.asarray(a.get_array()),ordered.width_db.to_numpy())
        scatters.append(a)
        scatter_checks.append(dict(platform=platform,points=len(ordered),
            represented_windows=len(q),exact_coordinates=True,exact_widths=True,
            colour_statistic='Individual window W',
            size_statistic='Fixed marker size0.45pt2' if row==0 else 'Fixed marker size6.2pt2',
            above_colour_limit=int(ordered.width_db.gt(4).sum()),
            ordering='Fixed-seed shuffle independent of amplitude',all_windows=True))
        row_position=left.get_position()
        fig.text(.025,(row_position.y0+row_position.y1)/2,platform,rotation=90,
            ha='center',va='center',fontsize=8.2)
        for col,ax in enumerate([left,right]):
            frame(ax)
            ax.text(-.135,1.022,chr(97+2*row+col),transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=8.5)
    cbax=fig.add_axes([.735,.961,.20,.010])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal',ticks=[0,1,2,3,4],extend='max',extendfrac=.025)
    bar_label=fig.text(.715,.966,r'$\delta C/N_0$ width $W$ (dB)',fontsize=7.3,ha='right',va='center')
    cb.outline.set_linewidth(.65);cb.ax.tick_params(labelsize=7,length=2,width=.65,pad=2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    legend_bbox=band_legend.get_window_extent(renderer)
    label_bbox=bar_label.get_window_extent(renderer)
    assert not legend_bbox.overlaps(label_bbox), 'Band legend overlaps colorbar label'
    assert label_bbox.x1 < cbax.get_window_extent(renderer).x0, 'Colorbar label overlaps bar'
    assert len(axes[0,1].collections)==1 and len(axes[0,1].collections[0].get_offsets())==n_ground, 'Ground scatter count mismatch'
    outside=[]
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text():
            bb=t.get_window_extent(renderer)
            if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:outside.append(t.get_text())
    assert not outside,outside
    output=OUT/'Extended_Data_Local_time_illumination.png'
    fig.savefig(output,dpi=600,pad_inches=0,bbox_inches=None,pil_kwargs={'compress_level':7})
    plt.close(fig)
    hist=pd.DataFrame(hist_rows);hist.to_csv(OUT/'Solar_LT_hourly_counts_R3.csv',index=False)
    summary=[]
    for (platform,band),q in data.groupby(['platform','latitude_band']):
        summary.append(dict(platform=platform,latitude_band=band,windows=len(q),epochs=int(q.n_points.sum()),
            fully_sunlit=int(q.sunlit_fraction.eq(1).sum()),fully_shadow=int(q.sunlit_fraction.eq(0).sum()),
            mixed=int(q.sunlit_fraction.between(0,1,inclusive='neither').sum()),
            evening_18_24=int(q.plot_solar_local_time_h.ge(18).sum()),median_width_db=float(q.width_db.median())))
    pd.DataFrame(summary).to_csv(OUT/'Window_sampling_summary_R3.csv',index=False)
    with Image.open(output) as im:dimensions=im.size;dpi=im.info['dpi']
    qa=dict(status='PASS_DATA_AND_RENDER_CHECKS',sources={str(s):sha(s) for s in sources},
        output=str(output),output_sha256=sha(output),pixels=dimensions,dpi=dpi,
        accepted_window_count=len(data),points=sum(x['points'] for x in scatter_checks),
        right_column_represented_windows=sum(x['represented_windows'] for x in scatter_checks),histogram_count=int(hist.n_windows.sum()),
        scatter_checks=scatter_checks,text_outside_canvas=outside,ground_count_overlay=False,ground_grid_degrees=None,ground_scatter_points=n_ground,ground_subsampling=False,ground_coordinate_jitter=False,ground_alpha=1,ground_latitude_coordinate='Signed centered-dipole magnetic latitude, pole80.65N72.68W',ground_magnetic_band_mismatches=0,solar_coordinates_unchanged=True,
        right_column_x='Solar zenith angle, degrees;0-180 for all platforms',
        left_axis_limits_percent={'Ground':[0,ground_ymax],'LEO-RO':[0,25],'LuGRE':[0,60]},
        layout_revision='Vertical platform row labels; no top titles; Ground b exact signed centered-dipole magnetic latitude xSZA individual-window scatter; compact single-row legend and colorbar',
        left_x_tick_marks=True,left_x_tick_labels=True,left_horizontal_grid=False,left_axis_label='Proportion per 1-h bin (%)',colorbar_orientation='horizontal',colorbar_at_top=True,colorbar_width_fraction=.20,header_row_overlap=False,
        ground_boundary=f'Illustrative spherical Earth6371km at350km, SZA={ground_shadow_sza:.8f}deg; actual source illumination WGS84pernativeepoch',
        nearspace_boundary='Illustrative spherical Earth6371km; actual source illumination WGS84 per native epoch',
        new_refits=False,new_observation_selection=True,spatial_interpolation=False)
    assert qa['accepted_window_count']==qa['right_column_represented_windows']==qa['histogram_count']==qa['points']==(n_ground+len(near))
    (OUT/'overall_plot_QA_R3.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':qa['status'],'output':str(output),'points':qa['points'],'pixels':dimensions},indent=2),flush=True)

if __name__=='__main__':main()

"""Regenerate the three-platform amplitude/phase comparison from frozen tables."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def render(output):
    import numpy as np,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    lugre=pd.read_csv(ROOT/'data/joint/lugre_paired_minutes_selection.csv',float_precision='round_trip')
    lugre=lugre[lugre.selected]
    leo=pd.read_csv(ROOT/'data/joint/leo_profiles_selected_P95.csv.gz',float_precision='round_trip')
    ground=pd.read_csv(ROOT/'data/joint/ground_station_minutes_selected.csv.gz',float_precision='round_trip')
    assert len(lugre)==920 and len(leo)==1619
    bands=['Equatorial','Mid-latitude','Polar'];colors=['#17899C','#7442B6','#CF363F']
    missions=['PlanetiQ','Spire','TSX','MetOp'];mcolors=['#1f77b4','#2ca02c','#7b3294','#d62728'];markers=['o','^','s','D']
    density=LinearSegmentedColormap.from_list('event_dark_to_red',['#240039','#12306E','#1766A6','#229C8D','#71BF44','#F0A51D','#D6422F'])
    gcmap=LinearSegmentedColormap.from_list('ground_viridis_like',['#341052','#334F9B','#1E93B3','#28B98D','#A8D84B','#FFF06A'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.65,'mathtext.fontset':'dejavusans'})
    width,height=4251,4015
    fig=plt.figure(figsize=(width/600,height/600),dpi=600)
    xb=[(472,1583),(1724,2835),(2976,4087)];yb=[(283,1228),(1546,2491),(2809,3754)]
    axes=[];counts=[];hex_rows=[]
    for row in range(3):
        for col,band in enumerate(bands):
            x0,x1=xb[col];y0,y1=yb[row]
            ax=fig.add_axes([x0/width,(height-y1)/height,(x1-x0)/width,(y1-y0)/height]);axes.append(ax)
            ax.tick_params(direction='out',length=2,width=.6,pad=2)
            for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#33383d')
            if row==0:
                d=lugre[lugre.lat_group.eq(band)]
                shown=d[d.cn0_p95_abs_delta_db.between(0,6)&d.roti_1min_independent.between(0,60)]
                ax.scatter(shown.cn0_p95_abs_delta_db,shown.roti_1min_independent,s=8.5,color=colors[col],alpha=.9,edgecolors='white',linewidths=.25)
                ax.set(xlim=(0,6),ylim=(0,60),xticks=[0,2,4,6],yticks=[0,15,30,45,60],xlabel=r'$|\delta C/N_0|$ (dB)')
                ax.set_title(band,fontsize=8.2,fontweight='bold',pad=25)
                if col==0:ax.set_ylabel('ROTI (TECU/min)',labelpad=4)
                counts.append({'platform':'LuGRE','latitude_band':band,'source_n':len(d),'displayed_n':len(shown)})
            elif row==1:
                d=leo[leo.latitude_band_magnetic.eq(band)].copy()
                hb=ax.hexbin(d.s4_p95,d.sigma_phi_rad_p95,gridsize=42,extent=(0,1,0,1),mincnt=1,bins='log',cmap=density,linewidths=0,alpha=.82)
                dm=d.mission.astype(str).replace({'MetOp-B':'MetOp','MetOp-C':'MetOp','TerraSAR-X':'TSX'})
                for mission,color,marker in zip(missions,mcolors,markers):
                    p=d[dm.eq(mission)]
                    ax.scatter(p.s4_p95,p.sigma_phi_rad_p95,s=8,color=color,marker=marker,alpha=.62,linewidths=0,zorder=4)
                if col==0:
                    handles=[Line2D([],[],color='none',marker=m,markerfacecolor=c,markeredgewidth=0,label=n,markersize=3) for n,c,m in zip(missions,mcolors,markers)]
                    ax.legend(handles=handles,loc='upper left',ncol=4,frameon=False,fontsize=4.8,handletextpad=.15,columnspacing=.4,borderpad=.2)
                counts.append({'platform':'LEO-RO','latitude_band':band,'source_n':len(d),'displayed_n':len(d)})
                hex_rows += [dict(platform='LEO-RO',latitude_band=band,x=float(x),y=float(y),n=int(n)) for (x,y),n in zip(hb.get_offsets(),hb.get_array())]
            else:
                d=ground[ground.lat_group_magnetic_approx.eq(band)]
                d=d[d.s4.between(0,1)&d.sigma_phi.between(0,1)]
                hb=ax.hexbin(d.s4,d.sigma_phi,gridsize=42,extent=(0,1,0,1),mincnt=1,cmap=gcmap,norm=LogNorm(1,10000),linewidths=0,alpha=.98)
                counts.append({'platform':'Ground','latitude_band':band,'source_n':len(d),'displayed_n':len(d)})
                hex_rows += [dict(platform='Ground',latitude_band=band,x=float(x),y=float(y),n=int(n)) for (x,y),n in zip(hb.get_offsets(),hb.get_array())]
                if col==2:
                    cax=ax.inset_axes([.77,.35,.045,.43]);cb=fig.colorbar(hb,cax=cax,ticks=[1,10,100,1000,10000])
                    cb.ax.set_yticklabels(['1','10','100','1,000','10,000']);cb.ax.tick_params(labelsize=5.1,length=2,pad=1)
                    ax.text(.795,.83,'Station-minute\nobservations\nper bin',transform=ax.transAxes,ha='center',va='bottom',fontsize=5.4,linespacing=.9)
                    ax.text(.81,.24,'Log scale',transform=ax.transAxes,ha='center',fontsize=5.4)
            if row>0:
                ax.set(xlim=(0,1),ylim=(0,1),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6),xlabel=r'$S_4$')
                if col==0:ax.set_ylabel(r'$\sigma_\phi$ (rad)',labelpad=5)
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            if col>0:ax.tick_params(labelleft=False)
            ax.text(-.09,1.035,chr(97+row*3+col),transform=ax.transAxes,fontsize=9.4,fontweight='bold')
    for row,label in enumerate(['LuGRE','LEO-RO','Ground ISMR']):
        y0,y1=yb[row];fig.text(.016,1-(y0+y1)/2/height,label,rotation=90,ha='center',va='center',fontweight='bold',fontsize=8)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=600,facecolor='white');plt.close(fig)
    pd.DataFrame(hex_rows).to_csv(output.with_name('Joint_redrawn_hexbin_counts.csv'),index=False)
    return {'figure':'Extended_Data_Joint_observables','reproduction_class':'scientific_numerical_redraw',
        'accepted_full_composite_pixel_exact':False,'LuGRE_source_points':920,'LEO_profile_points':1619,
        'quantile':'P95; no renamed P90 values','counts':counts}

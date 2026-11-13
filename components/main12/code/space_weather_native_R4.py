"""Native matplotlib style refresh of frozen ED2/ED8 data; no scientific reselection."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, ast, types, json, hashlib
sys.dont_write_bytecode=True

COMPONENT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'space_weather'
ROOT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'mplconfig')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import NullLocator, FuncFormatter

OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
SPACE=COMPONENT/'data/space_weather'
FORMAL=COMPONENT/'reference'
INK='#262626';MUTED='#636C76'
mpl.rcParams.update({'font.family':'Arial','font.size':7.,'axes.labelsize':7.,'axes.titlesize':7.,
    'xtick.labelsize':7.,'ytick.labelsize':7.,'legend.fontsize':7.,'axes.spines.top':True,
    'axes.spines.right':True,'axes.linewidth':.6,'axes.edgecolor':INK,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def style(ax):
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color(INK);spine.set_linewidth(.6)
    ax.tick_params(direction='out',length=2.2,width=.6,labelsize=7,pad=2)

def export(fig,name,sources,science,original):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    outside=[]
    for text in texts:
        bounds=text.get_window_extent(renderer)
        if bounds.width and bounds.height and (bounds.x0<-.5 or bounds.y0<-.5 or bounds.x1>fig.bbox.width+.5 or bounds.y1>fig.bbox.height+.5):outside.append(text.get_text())
    assert not outside,outside
    path=OUT/name
    fig.savefig(path,dpi=600,facecolor='white')
    report={'revision':'R4','status':'READY_FOR_VISUAL_CHECK','figure':str(path),'sha256':sha(path),
        'physical_size_mm':[float(x*25.4) for x in fig.get_size_inches()],
        'pixels':[int(x) for x in fig.canvas.get_width_height()], 'dpi':600,
        'font_family':'Arial','font_values_pt':sorted({float(t.get_fontsize()) for t in texts}),
        'font_above_7pt':[{'text':t.get_text(),'fontsize':float(t.get_fontsize())} for t in texts if t.get_fontsize()>7],
        'text_outside_canvas':outside,'frame_width_pt':.6,'frame_colour':INK,
        'ordinary_title_and_rowlabel_weight':'normal','panel_letter_weight':'bold',
        'source_hashes':{str(p):sha(p) for p in sources},'science':science,
        'baseline_formal_png':str(original),'baseline_sha256':sha(original),
        'native_renderer':'Python/matplotlib; no PNG resampling or reuse as a plot background',
        'formal_png_unchanged':True}
    save(ROOT/(Path(name).stem+'_style_QA_R4.json'),report)
    plt.close(fig)
    return report

def render_space():
    sources=[SPACE/'unchanged_space_weather_context.csv',SPACE/'op_mean_all_latitude_and_polar_h50.csv',COMPONENT/'code/space_weather_native_R4.py']
    context=pd.read_csv(sources[0],parse_dates=['datetime_utc']).set_index('datetime_utc')
    summary=pd.read_csv(sources[1],parse_dates=['start','end','center'])
    assert len(summary)==6 and summary.all_windows.sum()==2244 and summary.polar_windows.sum()==693
    t0,t1=pd.Timestamp('2025-03-03'),pd.Timestamp('2025-03-17')
    context=context.loc[t0:t1]
    cols={'f107':'#8E5376','bz':'#275B9A','span':'#B03A42','pcn':'#2C7789',
          'ae':'#DE7E1E','au':'#008762','al':'#6A3D9A','grey':'#8A8E94','red':'#B9453A'}
    specs=[('f107','F10.7\n(sfu)',130,250,[150,200,250]),('bz','IMF Bz\n(nT)',-15,15,[-10,0,10]),
           ('kp','Kp',0,6.5,[0,3,5,6]),('pcn','PCN\n(mV/m)',-.5,7.5,[0,3,6]),
           ('auroral','AE/AU/AL\n(nT)',-1600,1800,[-1000,0,1000]),('lugre','Mean\nfluctuation\nwidth (dB)',0,3,[0,1,2,3])]
    # Same six-panel physical map as the accepted 1800x1810 drawing.
    fig=plt.figure(figsize=(180/25.4,(180*1810/1800)/25.4),dpi=600)
    spans=[];start=None;previous=None
    for time,active in context.kp.ge(5).items():
        if active and start is None:start=time
        if not active and start is not None and previous is not None:
            spans.append((start,previous+pd.Timedelta(hours=1)));start=None
        previous=time
    if start is not None:spans.append((start,previous+pd.Timedelta(hours=1)))
    axes=[]
    for i,(key,ylabel,low,high,ticks) in enumerate(specs):
        top=58+i*(205+72);bottom=top+205
        ax=fig.add_axes([310/1800,1-bottom/1810,(1545-310)/1800,205/1810])
        axes.append(ax);style(ax)
        ax.set(xlim=(t0,t1),ylim=(low,high),yticks=ticks)
        ax.set_facecolor('#FCFCFB')
        ax.set_xticks(pd.date_range(t0,t1,freq='2D'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(True,color='#E0E2E6',lw=.35,alpha=.7)
        if i<5:ax.tick_params(labelbottom=False)
        else:ax.tick_params(axis='x',pad=29)
        for lo,hi in spans:ax.axvspan(lo,hi,color=cols['span'],alpha=24/255,lw=0,zorder=0)
        ax.text(-122/(1545-310),1.01,chr(97+i),transform=ax.transAxes,fontsize=8,fontweight='bold',ha='left',va='top')
        ax.text(-72/(1545-310),.5,ylabel,transform=ax.transAxes,fontsize=7,ha='right',va='center',linespacing=1.08)
        if key in ('f107','bz','kp','pcn'):
            values=context[key].clip(low,high)
            ax.plot(context.index,values,color=cols.get(key,'#2D3035'),lw=.864,zorder=3)
        if key=='f107':ax.text(.015,.90,'solar radio flux / EUV background proxy',transform=ax.transAxes,fontsize=6.5,color=cols['f107'],va='top')
        elif key=='bz':
            values=context.bz.clip(low,high)
            negative=values[values.lt(0)]
            ax.fill([t0,*negative.index,t1],[0,*negative,0],color='#CADEF4',alpha=115/255,lw=0,zorder=1)
            ax.axhline(0,color='#969696',lw=.3,zorder=2)
            ax.text(.985,.90,'southward IMF favors coupling',transform=ax.transAxes,fontsize=6.5,color=cols['bz'],ha='right',va='top')
        elif key=='kp':
            ax.axhline(5,color=cols['span'],lw=.576,ls=(0,(4.5,3)),zorder=2)
            ax.text(.97,(5-low)/(high-low)+.035,'Kp = 5',transform=ax.transAxes,fontsize=6.5,color=cols['span'],ha='right')
        elif key=='pcn':ax.text(.015,.9,'polar-cap convection response',transform=ax.transAxes,fontsize=6.5,color=cols['pcn'],va='top')
        elif key=='auroral':
            ax.axhline(0,color='#919191',lw=.3)
            for column in ('ae','au','al'):ax.plot(context.index,context[column].clip(low,high),color=cols[column],lw=.864,label=column.upper())
            ax.legend(loc='lower left',bbox_to_anchor=(.0,1.03),ncol=3,frameon=False,fontsize=6.5,handlelength=1.5,columnspacing=1,borderpad=0,handletextpad=.4)
        elif key=='lugre':
            handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=cols['grey'],ms=3,label='All latitudes (including polar)'),
                     Line2D([],[],ls='',marker='o',color=cols['red'],ms=3,label='Polar'),
                     Line2D([],[],ls='',label='60-s windows')]
            ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.035),ncol=3,frameon=False,fontsize=6.5,handlelength=.7,columnspacing=1.5,borderpad=0,handletextpad=.4)
            ax.axhline(1.5,color=cols['span'],lw=.576,ls=(0,(4.5,3)))
            ax.text(pd.Timestamp('2025-03-07 12:00'),1.6,'1.5 dB reference level',fontsize=6.5,color=cols['span'])
            for _,row in summary.iterrows():
                ax.vlines(row.center,0,row.all_mean_width_db,color=cols['grey'],lw=1.44,alpha=.7)
                ax.plot(row.center,row.all_mean_width_db,'o',mfc='#FAFAFA',mec=cols['grey'],ms=4.6,mew=.576,zorder=4)
                if row.polar_windows>0:
                    ax.vlines(row.center,0,row.polar_mean_width_db,color=cols['red'],lw=.864)
                    ax.plot(row.center,row.polar_mean_width_db,'o',color=cols['red'],ms=3.74,mew=.3,zorder=5)
                shift={'77_0':-9,'78_1':35}.get(row.op_id,0)
                color=cols['red'] if row.polar_windows>0 else cols['grey']
                dx=shift/1800*180/25.4*72
                ax.annotate(row.display_label,xy=(row.center,0),xycoords=('data','axes fraction'),xytext=(dx,-5),textcoords='offset points',rotation=50,ha='right',va='top',fontsize=6.5,color=color,annotation_clip=False)
    return export(fig,'Extended_Data_Space_weather_index.png',sources,
        {'context_rows':len(context),'OP_rows':len(summary),'valid_windows':2244,'polar_windows':693,
         'Kp_active_spans':[[str(a),str(b)] for a,b in spans],
         'frozen_context_and_OP_means_reused':True,'scientific_selection_or_detrending_changed':False,
         'original_body_font_pt':7.776,'original_panel_letter_pt':9.984},FORMAL/'Extended_Data_Space_weather_index_R4_reference.png')

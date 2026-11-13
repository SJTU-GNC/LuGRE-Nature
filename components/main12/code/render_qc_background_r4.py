from pathlib import Path
import os,sys,json,hashlib,shutil
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
COMPONENT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'qc_background'
SOURCE_ROOT=ROOT
DATA=ROOT/'data';OUTDATA=ROOT/'plotted_data';QA=ROOT/'qa'
for pth in [ROOT/'mplconfig',OUTDATA,QA]:pth.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'mplconfig')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[DATA/'E21_first_pass_all_points_with_QC_flags.csv',DATA/'E21_first_pass_window_QC_ledger.csv',DATA/'highalt_retained_60s_windows.csv',DATA/'highalt_retained_daily_widths.csv',SOURCE_ROOT/'qa/background_QC_R4.json',DATA/'highalt_retained_width_summary.csv']
for p in inputs:assert p.exists(),p
hashes={str(p):sha(p) for p in inputs}
points=pd.read_csv(inputs[0]);wins=pd.read_csv(inputs[1]);high=pd.read_csv(inputs[2]);daily=pd.read_csv(inputs[3]);sourceqa=json.loads(inputs[4].read_text(encoding='utf-8'));summary=pd.read_csv(inputs[5])
COL={'GPS L1':'#78A6D5','GPS L5':'#E58B85','Galileo E1':'#2B5FB3','Galileo E5a':'#C93F3B'};MARK={'GPS L1':'o','GPS L5':'o','Galileo E1':'o','Galileo E5a':'o'};SIGS=list(COL);INK='#262626';GREY='#A4A8AD';GRID='#E5E8EC'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'text.color':INK,'axes.linewidth':.6,'axes.spines.top':True,'axes.spines.right':True,'mathtext.default':'regular','figure.facecolor':'white','savefig.facecolor':'white'})
fig=plt.figure(figsize=(210/25.4,200/25.4));width=84/210;xs=[16/210,110/210]
axs=[fig.add_axes([x,145/200,width,48/200]) for x in xs]+[fig.add_axes([x,79/200,width,48/200]) for x in xs]+[fig.add_axes([x,13/200,width,48/200]) for x in xs]
def style(ax,grid=False):
 for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.6);s.set_color(INK)
 ax.tick_params(direction='in',top=False,bottom=True,right=True,width=.6,length=2.6,pad=3)
 ax.minorticks_off()
 if grid:ax.grid(axis='y',color=GRID,lw=.45,zorder=0)
 ax.set_axisbelow(True)
def label(ax,letter,title=None):
 if title:ax.set_title(title,loc='left',x=.065,pad=7,fontweight='normal')
 ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(0,4 if letter in 'abcd' else 7),textcoords='offset points',fontsize=10,fontweight='bold',ha='left',va='baseline',annotation_clip=False)
def utc(gps):return pd.Timestamp('1980-01-06',tz='UTC')+pd.Timedelta(seconds=float(gps)-18)
overview=[];endpoint_callouts=[]
# Set each complete-pass display to its actual observed extent plus two minutes.
for j,(sig,segment,first_idx,name) in enumerate([(2,1,24,'Galileo E1'),(3,4,0,'Galileo E5a')]):
 ax=axs[j];style(ax);label(ax,'ab'[j])
 q=points[points.signal_id.eq(sig)].copy();q['dt']=pd.to_datetime(q.time_utc,utc=True)
 lo=q['dt'].min().floor('min')-pd.Timedelta(minutes=2)
 hi=q['dt'].max().ceil('min')+pd.Timedelta(minutes=2)
 finite=q[np.isfinite(q.e23_residual_db)].copy();good=finite[finite.analysis_qc_keep];bad=finite[~finite.analysis_qc_keep]
 ax.scatter(bad.dt,bad.e23_residual_db,s=2.3,c=GREY,alpha=.65,linewidths=0,zorder=3)
 ax.scatter(good.dt,good.e23_residual_db,s=1.4,c=COL[name],alpha=.65,lw=0,zorder=4)
 # No interior horizontal guide; keep the numeric zero tick.
 ax.set_xlim(lo,hi);ax.set_ylim(-13,4);ax.set_yticks([-12,-8,-4,0,4]);ax.set_ylabel(name+' '+r'$\delta C/N_0$ (dB)')
 ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]));ax.xaxis.set_minor_locator(mpl.ticker.NullLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
 ax.set_xlabel('UTC on 3 March 2025 (hh:mm)',labelpad=4)
 w=wins[wins.signal_id.eq(sig)&wins.segment_id.eq(segment)&wins.window_idx.eq(first_idx)].iloc[0]
 start=utc(w.window_start_gps_s);duration=180;end=start+pd.Timedelta(seconds=duration)
 ax.add_patch(Rectangle((mdates.date2num(start),-12.9),duration/86400,16.8,fill=False,ec='#505860',lw=.8,zorder=6))
 # Local callouts stay over each grey endpoint, away from the zoom box.
 low_early=bad[bad.dt.lt(pd.Timestamp('2025-03-03 12:15',tz='UTC'))&bad.reject_point_cn0_lt25]
 low_late=bad[bad.dt.gt(pd.Timestamp('2025-03-03 13:20',tz='UTC'))&bad.reject_point_cn0_lt25&bad.e23_residual_db.abs().lt(2)]
 targets=[]
 if j==0 and len(low_early):targets.append(('early',low_early.iloc[0]))
 if len(low_late):targets.append(('late',low_late.iloc[-1]))
 for side,p in targets:
  assert p.e23_residual_db<1.8
  ax.text(p['dt'],2.6,r'$C/N_0<25$ dB-Hz',ha='left' if side=='early' else 'right',va='center',fontsize=8.5,color='#686D73',bbox={'fc':'white','ec':'none','pad':.1,'alpha':.9},zorder=8)
  arrow_start=p['dt']+(hi-lo)*(2.5/84)*(1 if side=='early' else -1)
  assert (p['dt']<start and arrow_start<start) or (p['dt']>end and arrow_start>end)
  arrow_tip_y=float(p.e23_residual_db)+(.8 if j==0 else 0.)
  ax.annotate('',xy=(p['dt'],arrow_tip_y),xycoords='data',xytext=(arrow_start,1.8),textcoords='data',arrowprops={'arrowstyle':'->','lw':.7,'color':'#686D73'},zorder=7)
  endpoint_callouts.append({'signal':name,'side':side,'time_utc':str(p['dt']),'CN0_dbhz':float(p.cn0_dbhz),'residual_db':float(p.e23_residual_db),'label':'CN0<25dB-Hz','local_vertical_arrow':False,'local_diagonal_arrow':True,'arrow_horizontal_inset_mm':2.5,'arrow_start_time_utc':str(arrow_start),'label_y_db':2.6,'arrow_start_y_db':1.8,'arrow_display_tip_y_db':arrow_tip_y,'arrow_display_tip_lift_db':(.8 if j==0 else 0.)})
 detail='cd'[j]
 ax.annotate('Detail '+detail,xy=(end,-6.1),xycoords='data',xytext=(end+pd.Timedelta(minutes=10),-6.1),textcoords='data',ha='left',va='center',fontsize=8.5,color='#505860',arrowprops={'arrowstyle':'->','lw':.6,'color':'#505860'},bbox={'fc':'white','ec':'none','pad':.15,'alpha':.85},zorder=7)
 ax.legend(handles=[Line2D([],[],ls='',marker='o',ms=3,c=GREY,alpha=.65,label='Excluded from statistics')],loc='lower right',fontsize=8.5,frameon=False,borderaxespad=.45,handlelength=.7,handletextpad=.3)
 assert finite.dt.between(lo,hi).all()
 assert finite.e23_residual_db.between(-13,4).all()
 overview.append({'signal':name,'native_n':len(q),'finite_detrended_points':len(finite),'coloured_retained':len(good),'grey_circled_excluded':len(bad),'unavailable_residual_not_synthesized':len(q)-len(finite),'detail_panel':detail,'boxed_start_UTC':str(start),'boxed_end_UTC':str(end),'time_or_amplitude_clipped':0,'display_start_UTC':str(lo),'display_end_UTC':str(hi),'padding_minutes_each_end':2})
plotted=[];windowexamples=[];detail_condition_artists=[]
def condition_text(ax,mid,first,second):
 artists=[ax.text(mid,y,text,ha='center',va='baseline',fontsize=8.5,color='#686D73') for y,text in [(1.35,first),(.25,second)]]
 detail_condition_artists.append((ax,mid,artists))

for j,(sig,specs,name) in enumerate([(2,[(1,24),(1,25),(1,26)],'Galileo E1'),(3,[(4,0),(4,1),(4,2)],'Galileo E5a')]):
 ax=axs[2+j];style(ax);label(ax,'cd'[j])
 select=np.zeros(len(wins),dtype=bool)
 for seg,wi in specs:select |= (wins.signal_id.eq(sig)&wins.segment_id.eq(seg)&wins.window_idx.eq(wi)).to_numpy()
 w=wins[select].sort_values('window_start_gps_s').copy();assert len(w)==len(specs)
 q=points[points.window_uid.isin(w.window_uid)].copy();origin=float(w.window_start_gps_s.iloc[0]);duration=float(w.window_start_gps_s.iloc[-1]-origin+60);q['elapsed_s']=q.rx_gps_seconds-origin
 finite=q[np.isfinite(q.e23_residual_db)].copy();assert finite.elapsed_s.between(0,duration,inclusive='left').all()
 accepted=finite[finite.analysis_qc_keep];excluded=finite[~finite.analysis_qc_keep]
 boundaries=sorted(set((w.window_start_gps_s-origin).tolist()+(w.window_start_gps_s-origin+60).tolist()))
 for bound in boundaries:ax.axvline(bound,color='#B4B8BE',lw=.65,ls=(0,(1,2)),zorder=1)
 for row in w.itertuples():
  x=float(row.window_start_gps_s-origin);n=int(row.n_point_quality_eligible)
  if not row.window_qc_keep:ax.axvspan(x,x+60,facecolor='#F1F2F3',alpha=.70,zorder=0)
  shown=int(np.isfinite(q.loc[q.window_uid.eq(row.window_uid),'e23_residual_db']).sum())
  ax.text(x+30,3.25,rf'$N_{{\rm valid}}={n}$',ha='center',va='center',fontsize=8.5)
  assert not row.window_qc_keep or shown>=30
  if n<30:
   condition_text(ax,x+30,r'$N_{\rm valid}<30$',r'$C/N_0<25$ dB-Hz')
  elif row.window_qc_keep:
   for value in [row.pre_bias_p05,row.pre_bias_p95]:ax.hlines(value,x+2,x+58,color=INK,lw=.95,zorder=5)
   ax.fill_between([x,x+60],[row.pre_bias_p05]*2,[row.pre_bias_p95]*2,color=COL[name],alpha=.055,zorder=1)
   ax.text(x+13,row.pre_bias_p95,r'$P_{95}$',ha='center',va='center',fontsize=8.5,bbox={'fc':'white','ec':'none','pad':.1,'alpha':.9},zorder=7)
   ax.text(x+13,row.pre_bias_p05,r'$P_{05}$',ha='center',va='center',fontsize=8.5,bbox={'fc':'white','ec':'none','pad':.1,'alpha':.9},zorder=7)
   ax.annotate('',xy=(x+51,row.pre_bias_p95),xytext=(x+51,row.pre_bias_p05),arrowprops={'arrowstyle':'<->','lw':.65,'color':INK},zorder=6)
   ax.text(x+47,(row.pre_bias_p95+row.pre_bias_p05)/2,r'$W$',ha='right',va='center',fontsize=8.5,bbox={'fc':'white','ec':'none','pad':.1,'alpha':.85},zorder=7)
  elif row.reject_window_median_bias:
   condition_text(ax,x+30,r'$|\mathrm{median}(\delta C/N_0)|$',r'$>1$ dB')
  else:
   condition_text(ax,x+30,r'$N_{\rm valid}<30$',r'$C/N_0<25$ dB-Hz')
  windowexamples.append({'signal':name,'source_segment':int(row.segment_id),'window_idx':int(row.window_idx),'start_UTC':str(utc(row.window_start_gps_s)),'shown_count':shown,'CN0_screened_count':n,'n':n,'pre_bias_median_db':None if pd.isna(row.pre_bias_median) else float(row.pre_bias_median),'keep':bool(row.window_qc_keep),'status':row.window_status})
 if j==0:
  low=finite[finite.window_uid.eq(w.window_uid.iloc[0])&finite.reject_point_cn0_lt25]
  if len(low):
   p=low.iloc[0]
   ax.annotate(r'$C/N_0<25$ dB-Hz',xy=(p.elapsed_s,p.e23_residual_db),xytext=(8,-11.25),textcoords='data',ha='left',va='center',fontsize=8.5,color='#686D73',arrowprops={'arrowstyle':'->','lw':.6,'color':'#686D73'},bbox={'fc':'white','ec':'none','pad':.1,'alpha':.85},zorder=7)
 ax.scatter(excluded.elapsed_s,excluded.e23_residual_db,s=5,c=GREY,alpha=.65,lw=0,zorder=3)
 ax.scatter(accepted.elapsed_s,accepted.e23_residual_db,s=5,c=COL[name],alpha=.85,lw=0,zorder=4)
 # No interior horizontal guide; keep the numeric zero tick.
 ax.set_xlim(0,duration);ax.set_xticks(boundaries);ax.set_ylim(-13,4);ax.set_yticks([-12,-8,-4,0,4]);ax.set_ylabel(name+' '+r'$\delta C/N_0$ (dB)')
 ax.set_xlabel(f'Time since {utc(origin).strftime("%H:%M:%S")} UTC (s)',labelpad=4)
 assert not ((finite.e23_residual_db<-13)|(finite.e23_residual_db>4)).any()
 plotted.append({'signal':name,'finite_points':len(finite),'coloured_retained':len(accepted),'grey_excluded':len(excluded),'no_finite_residual_not_plotted':len(q)-len(finite),'time_or_y_clipped':0})

ax=axs[4];style(ax);label(ax,'e','Lunar-surface daily background (>1,000 km)')
days=['2025-03-03','2025-03-04','2025-03-14','2025-03-15','2025-03-16'];dayidx={x:i for i,x in enumerate(days)}
for j,sig in enumerate(SIGS):
 q=daily[daily.signal_name.eq(sig)].sort_values('utc_day');x=q.utc_day.map(dayidx).to_numpy(float)+(j-1.5)*.115
 ax.errorbar(x,q['median'],yerr=[q['median']-q.p05,q.p95-q['median']],fmt=MARK[sig],ms=3,c=COL[sig],lw=.8,capsize=1.6,capthick=.7)
ax.set_xticks(range(5),['3','4','14','15','16']);ax.set_xlim(-.55,4.55);ax.set_xlabel('UTC day (March 2025)');ax.set_ylabel(r'1-min width $W$ (dB)');ax.set_ylim(0,1.2);ax.set_yticks([0,.3,.6,.9,1.2])
daily_pooled=high.groupby('utc_day',as_index=False).agg(n_windows=('W_db','size'),median_W_db=('W_db','median'))
assert daily_pooled.n_windows.sum()==len(high)
daily_pooled.to_csv(OUTDATA/'daily_pooled_window_medians_R4.csv',index=False)
signal_legend=ax.legend(handles=[Line2D([],[],color=COL[s],marker=MARK[s],ms=3,lw=.8,label=s) for s in SIGS],ncol=4,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False,fontsize=8.5,columnspacing=.7,handletextpad=.3,handlelength=1.0,borderpad=.1,labelspacing=.2,borderaxespad=.3)
# Per-signal daily median points are retained; no pooled daily number is overlaid.
ax=axs[5];style(ax);label(ax,'f','All lunar-surface observing days (>1,000 km)')
for j,sig in enumerate(SIGS):
 z=summary[summary.signal_name.eq(sig)].iloc[0]
 ax.errorbar([j],[z['median']],yerr=[[z['median']-z.p05],[z.p95-z['median']]],fmt=MARK[sig],ms=4,c=COL[sig],lw=1,capsize=2,capthick=.8)
 ax.text(j+.16,float(z['median']),f"{z['median']:.2f}",ha='left',va='center',fontsize=8.5,color=INK)
ax.set_xlim(-.8,3.7);ax.set_ylim(0,1.2);ax.set_yticks([0,.3,.6,.9,1.2]);ax.set_xticks(range(4),['GPS L1','GPS L5','Gal. E1','Gal. E5a']);ax.set_ylabel(r'1-min width $W$ (dB)')
# Legend symbols are illustrative keys and add no data to the axes.
class IntervalKey:pass
class MedianKey:pass
class IntervalHandler(HandlerBase):
 def create_artists(self,legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans):
  x=-xdescent+width/2;y0=-ydescent+.08*height;y1=-ydescent+.92*height;cap=width*.20
  return [Line2D([x,x],[y0,y1],color=INK,lw=.8,transform=trans),Line2D([x-cap,x+cap],[y0,y0],color=INK,lw=.8,transform=trans),Line2D([x-cap,x+cap],[y1,y1],color=INK,lw=.8,transform=trans)]
class MedianHandler(HandlerBase):
 def create_artists(self,legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans):
  y=-ydescent+height/2
  return [Line2D([-xdescent+width*.5],[y],marker='o',ls='',ms=3.5,color=INK,transform=trans)]
stat_legend=ax.legend([IntervalKey(),MedianKey()],[r'$P_{05}$–$P_{95}$','Median'],handler_map={IntervalKey:IntervalHandler(),MedianKey:MedianHandler()},ncol=2,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False,fontsize=8.5,handlelength=1.6,handletextpad=.45,columnspacing=1.5,borderpad=.2,borderaxespad=.3)
# Preserve both independent legends in E: signal identity at top, statistical marks at bottom.
axs[4].add_artist(signal_legend)
e_stat_legend=axs[4].legend([IntervalKey(),MedianKey()],[r'$P_{05}$–$P_{95}$','Median'],handler_map={IntervalKey:IntervalHandler(),MedianKey:MedianHandler()},ncol=2,loc='lower center',bbox_to_anchor=(.5,.04),frameon=False,fontsize=8.5,handlelength=1.6,handletextpad=.45,columnspacing=1.5,borderpad=.2,borderaxespad=.3)
assert daily.p95.max()<1.2 and summary.p95.max()<1.2
median=float(high.W_db.median())
# Signal legend is inside panel e.
for column,ax in enumerate(axs):
 right_column=(column%2==1)
 ax.yaxis.set_label_position('right' if right_column else 'left')
 ax.yaxis.set_ticks_position('right' if right_column else 'left')
 ax.tick_params(axis='y',which='both',left=not right_column,right=right_column,labelleft=not right_column,labelright=right_column)
 ax.yaxis.label.set_rotation(90)
 ax.yaxis.labelpad=4
for index,ax in enumerate(axs):
 assert ax.yaxis.get_label_position()==('right' if index%2 else 'left')
 assert ax.yaxis.get_ticks_position()==('right' if index%2 else 'left')
 if index<4:assert '\n' not in ax.yaxis.label.get_text()
 assert ax.yaxis.label.get_rotation()==90
 assert not any(line.get_visible() for line in ax.get_ygridlines())
 assert len(ax.xaxis.get_minorticklocs())==0
fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height();outside=[]
for text in fig.findobj(Text):
 if not text.get_visible() or not text.get_text():continue
 b=text.get_window_extent(renderer)
 if b.x0<-.5 or b.y0<-.5 or b.x1>fw+.5 or b.y1>fh+.5:outside.append(text.get_text())
assert not outside,outside
endpoint_annotation_gaps=[]
for idx,ax in enumerate(axs[:2]):
 box=ax.patches[0].get_window_extent(renderer);ab=ax.get_window_extent(renderer)
 for t in ax.texts:
  if t.get_text()!=r'$C/N_0<25$ dB-Hz':continue
  tb=t.get_window_extent(renderer)
  assert tb.x0>=ab.x0 and tb.x1<=ab.x1
  assert tb.x1<box.x0 or tb.x0>box.x1
  gap=max(box.x0-tb.x1,tb.x0-box.x1)*25.4/fig.dpi
  endpoint_annotation_gaps.append({'panel':'ab'[idx],'gap_to_zoom_box_mm':gap})
ef_titles=[]
for ax in axs[4:]:
 t=ax._left_title;tb=t.get_window_extent(renderer);ab=ax.get_window_extent(renderer)
 assert tb.x0>=ab.x0 and tb.x1<=ab.x1,(t.get_text(),(tb.x1-ab.x1)*25.4/fig.dpi)
 ef_titles.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'width_mm':tb.width*25.4/fig.dpi,'right_margin_mm':(ab.x1-tb.x1)*25.4/fig.dpi})
legend_box=signal_legend.get_window_extent(renderer)
legend_axis_box=axs[4].get_window_extent(renderer)
legend_margins_mm={'left':(legend_box.x0-legend_axis_box.x0)*25.4/fig.dpi,'right':(legend_axis_box.x1-legend_box.x1)*25.4/fig.dpi}
assert min(legend_margins_mm.values())>1,legend_margins_mm
assert signal_legend._ncols==4
stat_box=stat_legend.get_window_extent(renderer);stat_ax=axs[5].get_window_extent(renderer)
assert stat_box.x0>stat_ax.x0 and stat_box.x1<stat_ax.x1 and stat_box.y1<stat_ax.y1
assert stat_box.y0>axs[5].transData.transform((0,float(summary.p95.max())))[1]+2
estat_box=e_stat_legend.get_window_extent(renderer)
eax_box=axs[4].get_window_extent(renderer)
assert estat_box.x0>eax_box.x0 and estat_box.x1<eax_box.x1 and estat_box.y0>eax_box.y0
estat_data_gap_mm=(axs[4].transData.transform((0,float(daily.p05.min())))[1]-estat_box.y1)*25.4/fig.dpi
signal_data_gap_mm=(legend_box.y0-axs[4].transData.transform((0,float(daily.p95.max())))[1])*25.4/fig.dpi
assert estat_data_gap_mm>1,estat_data_gap_mm
assert signal_data_gap_mm>.6,signal_data_gap_mm
right_axis_gaps=[]
for ax in axs[1::2]:
 label_box=ax.yaxis.label.get_window_extent(renderer)
 tick_boxes=[t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_visible()]
 gap_mm=(label_box.x0-max(b.x1 for b in tick_boxes))*25.4/fig.dpi
 assert gap_mm>=1.0,gap_mm
 right_axis_gaps.append(float(gap_mm))
print('Right-axis label gaps (mm):',right_axis_gaps,flush=True)
condition_margins=[]
for ax,mid,artists in detail_condition_artists:
 assert [note.get_position()[1] for note in artists]==[1.35,.25]
 left=ax.transData.transform((mid-30,0))[0];right=ax.transData.transform((mid+30,0))[0]
 for note in artists:
  box=note.get_window_extent(renderer)
  margins={'left_mm':float((box.x0-left)*25.4/fig.dpi),'right_mm':float((right-box.x1)*25.4/fig.dpi)}
  assert min(margins.values())>=1.2,margins
  condition_margins.append(margins)
for ax in axs[2:4]:
 for note in ax.texts:
  if note.get_text() not in (r'$P_{05}$',r'$P_{95}$'):continue
  assert note.get_verticalalignment()=='center'
  line_levels=[float(line.get_segments()[0][0,1]) for line in ax.collections if isinstance(line,mpl.collections.LineCollection) and line.get_segments()]
  assert any(abs(note.get_position()[1]-v)<1e-9 for v in line_levels)
print('Aligned condition baselines: [1.35,0.25] dB; conditions stay within each minute.',flush=True)
out=ROOT/'Extended_Data_Background_quantiles_QC.png';fig.savefig(out,dpi=600)
summary[['signal_name','n_windows','p05','median','p95']].to_csv(OUTDATA/'plotted_highalt_pooled_summary.csv',index=False)
pd.DataFrame(windowexamples).to_csv(OUTDATA/'plotted_E21_QC_example_windows.csv',index=False)
qa={'status':'REPRODUCED_R4_QC_BACKGROUND','input_hashes':hashes,'output':str(out),'output_sha256':sha(out),'figure_mm':[210,200],'dpi':600,'font_pt':{'ordinary':9,'secondary':8.5,'panel_letters':10},'a_b_overview':overview,'endpoint_callouts':endpoint_callouts,'c_d_details':plotted,'window_examples':windowexamples,'grey_definition':'Every finite existing residual with analysis_qc_keep=False, including entire rejected windows and individual point quality flags. No missing residual invented.','quantile_definition':'P05/P95 from point-quality-eligible actual seconds of retained windows; horizontal lines disconnected across windows; W is P95-P05. No P05/P95 tail point quality removal.','e_f_same_population':True,'N_lt60_annotation_removed':True,'formal_minimum_valid_seconds':30,'count_label_definition':'N_valid counts actual1Hz samples after point-level C/N0 quality screening, before the whole-window median test. Formal minimum30 unchanged; all N<60 annotations removed.','highalt_retained_windows':len(high),'highalt_median_W_db':median,'f_summary':'Per-signal median(W) and P05/P95(W) over the same accepted windows','f_all_retained_widths_included':True,'text_outside_canvas':outside,'new_fits':False,'D_manuscript_or_existing_figures_modified':False,'style_only_data_and_QC_unchanged':True,'excluded_marker':'consistent filled grey circles','shared_condition_key':'removed; explicit median and sample-count/CN0 conditions placed inside each detailed window','overview_zoom_arrows':'short horizontal arrows ending on the existing zoom boxes','PDF_generated':False,'daily_pooled_medians':daily_pooled.to_dict('records'),'daily_number_definition':'Median of all retained individual window W values on that day, pooling signals; not an average of signal medians.','frequency_palette':COL,'constellation_markers':MARK,'outside_AB_legends_removed':True,'AB_low_CN0_callouts':'A both endpoints,B right endpoint only; short diagonal arrows with2.5mm inward displacement','A_D_panel_titles_removed':True,'condition_inner_margins_mm':condition_margins,'standalone_panel_letter_offset_pt':4,'right_column_yaxes':'b,d,f ticklabels and single-line ylabels on right; all ylabels read in the same90degree direction','right_axis_label_gap_mm':right_axis_gaps,'right_axis_labelpad_pt':4,'axis_panel_width_mm':84,'center_gap_mm':10,'single_line_ylabels':True,'single_line_median_threshold':False,'single_line_CN0_threshold_and_unit':True,'GPS_colour_style':'light blue and light red; Galileo blue/red fixed','all_row_heights_mm':[48,48,48],'horizontal_interior_gridlines':False,'major_ticks':'bottom x and outside y,uniform for all six panels; no minor ticks','F_first_actual_whisker_labels':False,'F_statistical_legend':'P05-P95 capped range; single circle for median; common to E/F','EF_unified_median_marker':'o','A_only_arrow_tip_lift_db':.8,'B_arrow_unchanged':True,'F_numeric_medians':summary[['signal_name','median']].to_dict('records'),'signal_legend':'inside e,single row of four complete signal names','daily_heading_removed':True,'E_pooled_daily_numbers_removed':True,'C_D_bias_threshold_definition':r'abs(median(delta C/N0))>1dB within a one-minute window; not W>1dB','C_D_final_condition_label_y_db':[1.35,.25],'CD_condition_lines_share_exact_baselines':True,'CD_quantile_text_centred_on_black_lines':True,'EF_y_range_db':[0,1.2],'EF_statistical_legends':True,'EF_titles':ef_titles,'AB_endpoint_text_gaps':endpoint_annotation_gaps,'EF_lunar_surface_scope_explicit':True,'E_statistical_legend_gap_to_data_mm':estat_data_gap_mm,'E_signal_legend_gap_to_data_mm':signal_data_gap_mm,'E_single_row_legend_inner_margins_mm':legend_margins_mm,'C_D_final_count_condition':r'$N_{\rm valid}<30$'}
(QA/'figure_QC_layout_R4.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:qa[k] for k in ['output','output_sha256','a_b_overview','c_d_details','highalt_retained_windows','highalt_median_W_db','text_outside_canvas']},indent=2))






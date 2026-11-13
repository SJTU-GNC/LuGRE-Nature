"""Restack the established schematic and plots; numerical sources stay frozen."""
from pathlib import Path
import os, json, hashlib, itertools, sys, io
sys.dont_write_bytecode=True
COMPONENT=Path(__file__).resolve().parents[1]
ASSETS=COMPONENT/'data/antenna/R4_plot_assets'
R=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'antenna'
R.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt,binary_dilation
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

S=ASSETS
PREV=ASSETS
ORIG=ASSETS/'original_schematic_source.png'
ICON=ASSETS/'gps_iii_satellite_cutout.png'
SIGNALS=['GPS L1','GPS L5','Galileo E1','Galileo E5a']
LABEL={'GPS L1':'GPS L1 C/A','GPS L5':'GPS L5Q','Galileo E1':'Galileo E1C','Galileo E5a':'Galileo E5a-Q'}
COLOR={'GPS L1':'#2D6C9F','GPS L5':'#69A6D3','Galileo E1':'#E2A532','Galileo E5a':'#D96B43'}
INK='#202833';MUTED='#69727D';GRID='#E2E7EB';GREEN='#008A78';GOLD='#BE8D13'
W,H=4800,2190
CROP_LEFT=220
AB_SHIFT=160
PHYSICAL_WIDTH_MM=180
DPI=W/(PHYSICAL_WIDTH_MM/25.4)
FINAL_DPI=600
FINAL_W=round((W-CROP_LEFT)/DPI*FINAL_DPI)
FINAL_H=round(H/DPI*FINAL_DPI)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
PATTERN_ROOT=ASSETS
paths=[PREV/'density_display_source_R3.csv',S/'selected_lunar_offboresight_summary_R3.csv',PATTERN_ROOT/'reference_full_range_summary_R3.csv',S/'reference_pattern_envelopes_R3.csv',ASSETS/'A_clean.png',ORIG,ICON]
hashes={str(p):sha(p) for p in paths}
density=pd.read_csv(paths[0]);stats=pd.read_csv(paths[1]).set_index('signal_name')
env=pd.read_csv(paths[2]);archived=pd.read_csv(paths[3])
valleys=env.groupby('signal').first_valley_deg.first().to_dict()
legacy_valleys=valleys.copy()
check=env.merge(archived[['signal','theta_deg','median_delta_eirp_db','p05_delta_eirp_db','p95_delta_eirp_db']],on=['signal','theta_deg'],suffixes=('','_archived'),validate='one_to_one')
assert np.allclose(check.median_delta_eirp_db,check.median_delta_eirp_db_archived,rtol=0,atol=1e-10)
assert (check.min_delta_eirp_db<=check.p05_delta_eirp_db+1e-10).all()
assert (check.max_delta_eirp_db>=check.p95_delta_eirp_db-1e-10).all()
C_TOP=max(5,float(5*np.ceil(env.max_delta_eirp_db.max()/5)))
assert stats.n.sum()==137686
minima={k:float(stats.loc[v,'min_deg'].min()) for k,v in {'GPS':SIGNALS[:2],'Galileo':SIGNALS[2:]}.items()}
maxima={k:float(stats.loc[v,'max_deg'].max()) for k,v in {'GPS':SIGNALS[:2],'Galileo':SIGNALS[2:]}.items()}
plt.rcParams.update({'font.family':'Arial','font.size':6.2,'axes.labelsize':6.2,'axes.titlesize':7,
 'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.6,'axes.edgecolor':INK,
 'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'legend.fontsize':6.2,
 'mathtext.default':'regular','figure.facecolor':'white','savefig.facecolor':'white'})
fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI)
structural={};titles=[];direct_labels=[];arrow_labels=[]
def heading(key,text,x,y):
 letter,title=text.split('. ',1)
 label=fig.text(x/W,(H-y)/H,letter,fontproperties=FontProperties(family='Arial',size=8,weight='bold'),va='center')
 t=fig.text((x+85)/W,(H-y)/H,title,fontproperties=FontProperties(family='Arial',size=7,weight='bold'),va='center')
 structural[key+'_letter']=label;structural[key]=t;titles.append(t);return t
heading('title_a','a. Propagation and antenna geometry',100+AB_SHIFT,110)
heading('title_b','b. Off-boresight angles (50–1,000 km tangent height)',100+AB_SHIFT,1220)
heading('title_c','c. GNSS antenna transmit patterns',2190,110)

# Uniformly resize the full original schematic. Restore its cropped word before resizing.
art=np.asarray(Image.open(ASSETS/'A_clean.png').convert('RGBA')).copy()
assert art.shape[:2]==(2270,1770)
art[:750,1640:]=255
art[30:330,620:1215]=255
art[200:300,160:660]=255
roi=(2010,2315,2355,2395)
patch=np.asarray(Image.open(ORIG).convert('RGBA').crop(roi))
px,py=roi[0]-560,roi[1]-350
# The original word extends beyond the old crop; extend only its white margin.
expanded=np.full((2270,1800,4),255,dtype=np.uint8);expanded[:,:1770]=art
expanded[py:py+patch.shape[0],px:px+patch.shape[1]]=patch
assert np.array_equal(expanded[py:py+patch.shape[0],px:px+patch.shape[1]],patch)
# Replace only the old text glyphs, not the coloured sectors or green ray.
label_masks={}
for key,roi_text in {'Side lobe':(1400,930,1760,1035),'LuGRE':(1493,1320,1760,1425),'Main lobe':(1450,1965,1795,2045)}.items():
 x0,y0,x1,y1=roi_text
 rgb=expanded[y0:y1,x0:x1,:3].astype(float)
 red,grn,blu=[rgb[...,i] for i in range(3)]
 if key=='Side lobe':mask=(red<220)&(grn<215)&(red-grn>8)&(blu-grn>18)
 elif key=='LuGRE':mask=(red<220)&(grn>red*1.13)&(grn>blu*1.01)
 else:mask=(red<210)&(blu>160)&(blu-red>25)&(grn-red>10)
 mask=binary_dilation(mask,iterations=2)
 assert mask.sum()>100,('Empty text mask',key)
 before=expanded[y0:y1,x0:x1].copy()
 _,indices=distance_transform_edt(mask,return_indices=True)
 corrected=before.copy();corrected[mask]=before[indices[0][mask],indices[1][mask]]
 assert np.array_equal(before[~mask],corrected[~mask])
 expanded[y0:y1,x0:x1]=corrected
 label_masks[key]={'roi':list(roi_text),'replaced_pixels':int(mask.sum()),'outside_mask_unchanged':True}
Image.fromarray(expanded).save(R/'A_relabel_base.png')
scale=0.42
art_img=Image.fromarray(expanded).resize((round(expanded.shape[1]*scale),round(expanded.shape[0]*scale)),Image.Resampling.LANCZOS)
AX,AY=400+AB_SHIFT,180
fig.figimage(np.asarray(art_img),xo=AX,yo=H-AY-art_img.height,origin='upper',zorder=-10)
structural['alpha']=fig.text((AX+1015*scale)/W,(H-(AY+1245*scale))/H,r'$\alpha$',fontsize=6.2,ha='center',va='center',color='#2D6C9F')
icon=Image.open(ICON).convert('RGBA');iw=300;ih=round(iw*icon.height/icon.width)
icon=icon.resize((iw,ih),Image.Resampling.LANCZOS)
beam=(AX+889*scale,AY+330*scale)
ix=round(beam[0]-150*iw/420);iy=round(beam[1]-206*iw/420)
fig.figimage(np.asarray(icon),xo=ix,yo=H-iy-ih,origin='upper',zorder=-9)
structural['satellite_label']=fig.text((ix-65)/W,(H-(iy+90))/H,'GNSS satellite',fontsize=6.2,fontweight='normal',ha='right',va='center')
for key,(x,y,color) in {
 'Side lobe':(1590,980,'#705488'),
 'LuGRE':(1493,1370,GREEN),
 'Main lobe':(1900,2000,'#5A7FAD')}.items():
 structural['A_'+key]=fig.text((AX+x*scale)/W,(H-(AY+y*scale))/H,key,fontsize=6.2,fontweight='normal',ha='left' if key=='LuGRE' else 'center',va='center',color=color)
structural['A_height_scope']=fig.text((AX+1493*scale)/W,(H-(AY+1370*scale+38))/H,'(50–1,000 km tangent height)',fontsize=5.7,ha='left',va='top',color=GREEN)

def axes(box):
 x0,y0,x1,y1=box
 ax=fig.add_axes([x0/W,(H-y1)/H,(x1-x0)/W,(y1-y0)/H])
 for sp in ax.spines.values():sp.set_visible(True);sp.set_color(INK);sp.set_linewidth(.7)
 ax.tick_params(direction='out',length=2.5,width=.6,pad=2)
 ax.set_axisbelow(True)
 return ax

BBOX=(483,1370,1933,1950)
b=axes(BBOX)
for signal in SIGNALS:
 q=density[density.signal_name.eq(signal)]
 b.fill_between(q.angle_deg,q.density,color=COLOR[signal],alpha=.18,lw=0)
 b.plot(q.angle_deg,q.density,color=COLOR[signal],lw=1,label=LABEL[signal])
for constellation,y in [('GPS',.80),('Galileo',.86)]:
 lo,hi=minima[constellation],maxima[constellation]
 co=COLOR['GPS L1'] if constellation=='GPS' else GOLD
 b.annotate('',xy=(hi,y),xytext=(lo,y),arrowprops=dict(arrowstyle='<->',color=co,lw=1,shrinkA=0,shrinkB=0,mutation_scale=6))
 b.vlines([lo,hi],y-.02,y+.02,color=co,lw=.8)
 x=hi-.20 if constellation=='GPS' else (lo+hi)/2
 label=b.annotate(f'{constellation} {lo:.1f}°–{hi:.1f}°',xy=(x,y),xytext=(0,1.2),textcoords='offset points',fontsize=5.5,ha='right' if constellation=='GPS' else 'center',va='bottom',color=co)
 structural['range_'+constellation]=label;arrow_labels.append((label,b,(x,y)))
b.set_xlim(12,17);b.set_ylim(0,1);b.set_yticks([0,.25,.5,.75,1]);b.set_xticks([12,13,14,15,16,17])
b.tick_params(axis='y',right=False,labelright=False)
b.grid(True,color=GRID,lw=.45)
b.set_xlabel(r'Transmitter off-boresight angle, $\alpha$ (°)',labelpad=4)
b.set_ylabel('Probability density',labelpad=1)
structural['b_xlabel']=b.xaxis.label;structural['b_ylabel']=b.yaxis.label
# Four unchanged signal identities, laid out in one row over the B frame.
structural['b_legend']=fig.legend(*b.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(1208/W,(H-BBOX[1]+10)/H),borderaxespad=0,borderpad=0,ncol=4,frameon=False,fontsize=5.5,columnspacing=.45,handlelength=1.0,handletextpad=.3)

BOXES=[(2250,230,3350,1030),(3470,230,4570,1030),(2250,1150,3350,1950),(3470,1150,4570,1950)]
axes_c=[]
signal_tags=[]
height_scope_checks=[]
for i,(signal,box) in enumerate(zip(SIGNALS,BOXES)):
 ax=axes(box);axes_c.append(ax);color=COLOR[signal]
 q=env[env.signal.eq(signal)].sort_values('theta_deg')
 range_band=ax.fill_between(q.theta_deg,q.min_delta_eirp_db,q.max_delta_eirp_db,color=color,alpha=.20,lw=0,edgecolor='none',zorder=1)
 assert range_band.get_alpha()==.20 and not np.any(range_band.get_linewidths())
 ax.plot(q.theta_deg,q.median_delta_eirp_db,color=color,lw=1,zorder=3)
 ax.axvspan(float(stats.loc[signal,'min_deg']),float(stats.loc[signal,'max_deg']),color=GREEN,alpha=.12,lw=0,zorder=0)
 bound=valleys[signal]
 ax.axvline(bound,color=MUTED,ls=(0,(3,2)),lw=.8,zorder=4)
 arrow_y=-37
 for name,xlo,xhi in [('Main lobe',.3,bound-.3),('Side lobes',bound+.3,74.7)]:
  ax.annotate('',xy=(xhi,arrow_y),xytext=(xlo,arrow_y),arrowprops=dict(arrowstyle='<->',lw=.7,color=MUTED,mutation_scale=6),zorder=2)
  x=(xlo+xhi)/2
  label=ax.annotate(name,xy=(x,arrow_y),xytext=(0,1.2),textcoords='offset points',ha='center',va='bottom',fontsize=5.5,color=MUTED)
  structural[f'c{i}_'+name]=label;arrow_labels.append((label,ax,(x,arrow_y)))
 structural[f'c{i}_LuGRE']=ax.text(6,-20,'LuGRE',fontsize=6.2,ha='left',va='center',color=GREEN)
 structural[f'c{i}_valley']=ax.text(bound+1.3,-5,f'{bound:.0f}°',va='center',color=MUTED,fontsize=5.8)
 # Put the median label directly on a locally flat portion of the actual curve.
 candidates=[]
 for xx in range(43,61):
  within=q[q.theta_deg.between(xx-4,xx+4)]
  candidates.append((float(within.median_delta_eirp_db.std()),xx))
 median_x=min(candidates)[1]
 median_y=float(np.interp(median_x,q.theta_deg,q.median_delta_eirp_db))
 t=ax.text(median_x,median_y,'median',ha='center',va='center',fontsize=6,color='#5C6470',zorder=6)
 assert t.get_bbox_patch() is None
 structural[f'c{i}_median']=t
 direct_labels.append(dict(signal=signal,label='median',anchor=[median_x,median_y],label_position=[median_x,median_y],on_curve=True,leader_line=False))
 ax.set_xlim(0,75);ax.set_ylim(-40,C_TOP);ax.set_xticks(np.arange(0,76,10));ax.set_yticks(np.arange(-40,C_TOP+.01,10))
 ax.grid(True,color=GRID,lw=.4)
 if i%2:
  ax.yaxis.tick_right();ax.yaxis.set_label_position('right');ax.tick_params(axis='y',labelleft=False,labelright=True)
 ax.set_ylabel('Relative EIRP (dB)',labelpad=2)
 tag=ax.text(.97,.95,LABEL[signal],transform=ax.transAxes,fontsize=6.2,ha='right',va='top',color=INK)
 structural[f'c{i}_signal_tag']=tag;signal_tags.append((tag,ax,q))
 structural[f'c{i}_ylabel']=ax.yaxis.label
 if i<2:ax.tick_params(axis='x',labelbottom=False,direction='in')
 else:
  ax.set_xlabel(r'Off-boresight angle, $\alpha$ (°)',labelpad=4)
  structural[f'c{i}_xlabel']=ax.xaxis.label

fig.canvas.draw();renderer=fig.canvas.get_renderer()
def bounds(t):return [float(v) for v in t.get_window_extent(renderer).bounds]
def intersect(a,b):
 x,y,w,h=a;xx,yy,ww,hh=b
 return max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
# Annotation extents include their leader lines; use text-only extents for text collision checks.
def text_bounds(t):
 if isinstance(t,mpl.text.Annotation):return [float(v) for v in mpl.text.Text.get_window_extent(t,renderer).bounds]
 return bounds(t)
boxes_text={k:text_bounds(v) for k,v in structural.items()}
outside={k:v for k,v in boxes_text.items() if v[0]<CROP_LEFT or v[1]<0 or v[0]+v[2]>W or v[1]+v[3]>H}
signal_tag_checks=[]
for tag,ax,q in signal_tags:
 bb=tag.get_window_extent(renderer)
 assert ax.bbox.contains(bb.x0,bb.y0) and ax.bbox.contains(bb.x1,bb.y1)
 (xl,yl),(xh,yh)=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
 samples=np.linspace(xl,xh,100)
 upper=np.interp(samples,q.theta_deg,q.max_delta_eirp_db)
 assert upper.max()<yl,('Signal tag over pattern range',tag.get_text())
 signal_tag_checks.append({'signal':tag.get_text(),'inside_axes':True,'above_source_range':True,'axes_fraction':[.97,.95]})
collisions=[dict(first=a,second=b,overlap_px2=intersect(boxes_text[a],boxes_text[b])) for a,b in itertools.combinations(structural,2) if intersect(boxes_text[a],boxes_text[b])>1]
gaps=[]
for label,ax,xy in arrow_labels:
 bottom=text_bounds(label)[1];arrow_px=ax.transData.transform(xy)[1]
 gaps.append(float((bottom-arrow_px)*72/DPI))
assert all(abs(g-1.2)<.02 for g in gaps),gaps
legend_rows=len({round(t.get_window_extent(renderer).y0,1) for t in structural['b_legend'].get_texts()})
assert legend_rows==1
for key in ['range_GPS','range_Galileo']:
 assert structural[key].get_window_extent(renderer).y1<b.bbox.y1-3,('Range label too close to top frame',key)
legend_frame_gap_px=structural['b_legend'].get_window_extent(renderer).y0-b.bbox.y1
assert abs(legend_frame_gap_px-10)<.1,legend_frame_gap_px
scope_curve_intersections=[]
for scope,ax,q,signal in height_scope_checks:
 bb=scope.get_window_extent(renderer)
 (xl,yl),(xh,yh)=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
 qx=np.linspace(max(xl,0),min(xh,75),200)
 qy=np.interp(qx,q.theta_deg,q.median_delta_eirp_db)
 if ((qy>=yl)&(qy<=yh)).any():scope_curve_intersections.append({'signal':signal,'scope_bounds':[xl,yl,xh,yh]})
assert density.density.max()<1
assert all('\n' not in ax.yaxis.label.get_text() for ax in axes_c)
for p,h in hashes.items():assert sha(p)==h

OUT=R/'Extended_Data_GNSS_antenna_geometry.png'
bottom_axes=[b,*axes_c[2:]]
xlabel_bottoms=[float(ax.xaxis.label.get_window_extent(renderer).y0) for ax in bottom_axes]
xtick_bottoms=[float(ax.get_xticklabels()[0].get_window_extent(renderer).y0) for ax in bottom_axes]
assert max(xlabel_bottoms)-min(xlabel_bottoms)<.5,('Xlabel baseline mismatch',xlabel_bottoms)
assert max(xtick_bottoms)-min(xtick_bottoms)<.5,('Tick baseline mismatch',xtick_bottoms)
font_checks=[]
panel_letter_ids={id(v) for k,v in structural.items() if k.endswith('_letter')}
for t in fig.findobj(mpl.text.Text):
 if not t.get_visible() or not t.get_text().strip():continue
 size=float(t.get_fontsize());is_letter=id(t) in panel_letter_ids
 assert (size==8 if is_letter else 5<=size<=7),('Font size outside range',t.get_text(),size)
 assert t.get_fontfamily()==['Arial'],('Unexpected font',t.get_text(),t.get_fontfamily())
 font_checks.append({'text':t.get_text(),'pt':size,'font':t.get_fontfamily(),'panel_letter':is_letter})
data_line_widths=[float(line.get_linewidth()) for ax in [b,*axes_c] for line in ax.lines]
assert min(data_line_widths)>=.25 and max(data_line_widths)<=1
buffer=io.BytesIO();fig.savefig(buffer,dpi=DPI,facecolor='white');plt.close(fig)
buffer.seek(0)
Image.open(buffer).convert('RGB').crop((CROP_LEFT,0,W,H)).resize((FINAL_W,FINAL_H),Image.Resampling.LANCZOS).save(OUT,dpi=(FINAL_DPI,FINAL_DPI))
qa={'status':'CANDIDATE_PENDING_VISUAL_QA','output':str(OUT),'pixels':[FINAL_W,FINAL_H],'dpi':FINAL_DPI,
 'layout':'Compact A over B at left; shorter 2x2 C at right','A_uniform_scale':scale,'A_distortion':False,
 'A_old_text_replaced_by_larger_labels':True,'A_text_masks':label_masks,'A_new_label_pt':6.2,'A_label_weight':'normal',
 'B_bounds':list(BBOX),'C_bounds':BOXES,'B_and_C_bottom_alignment_px':1950,
 'B_actual_legend_text_rows':legend_rows,'B_legend_fontsize':5.5,'C_repeated_pattern_legends_removed':True,
 'C_direct_quantile_and_median_labels':direct_labels,'C_signal_ylabel_one_line':True,
 'C_LuGRE_xy':[6,-20],'A_LuGRE_height_scope':'(50–1,000 km tangent height)','C_height_scope_removed':True,'C_first_valley_number_y':-5,'arrow_text_gaps_pt':gaps,'B_xlim':[12,17],'B_ylim':[0,1],
 'C_ylim':[-40,C_TOP],'observed_constellation_minima_deg':minima,'observed_constellation_maxima_deg':maxima,
 'n_observations':int(stats.n.sum()),'source_hashes':hashes,'all_quantitative_source_tables_unchanged':True,
 'range_arrow_labels':{k:f'{minima[k]:.1f}°–{maxima[k]:.1f}°' for k in minima},
 'text_boxes':boxes_text,'text_outside_canvas':outside,'structural_text_collisions':collisions,
 'C_display_values_below_lower_limit':int((env[['min_delta_eirp_db','median_delta_eirp_db','max_delta_eirp_db']]<-40).sum().sum()),
 'scientific_samples_not_changed_by_display_limits':True,'physical_size_mm':[(W-CROP_LEFT)/DPI*25.4,H/DPI*25.4]}
qa.update({'proof_only':True,'D_sources_figures_PDFs_metadata_unchanged':True,'manuscript_compilation_performed':False,
 'height_change_from_previous_percent':(H/2500-1)*100,'A_height_reduction_percent':(1-scale/.54)*100,
 'C_subplot_aspect_height_over_width':800/1100,'C_height_reduction_percent':(1-800/955)*100,'B_title_canvas_y':1220,'B_group_translation_up_px':310,'B_single_line_heading':True,'B_Galileo_arrow_y':.86,'B_GPS_label_x':maxima['GPS']-.2,
 'B_right_ticks_and_numbers_removed':True,'B_legend_frame_gap_px':float(legend_frame_gap_px),'C_height_scope_median_intersections':scope_curve_intersections,
 'A_satellite_label_position':'left side of satellite','C_labels_black_no_leaders':True,
 'C_range_definition':'actual min-max across all available raw normalized satellite/azimuth cuts; no prior per-SVN azimuth collapse',
 'C_legacy_first_valleys':legacy_valleys,'C_new_first_valleys':valleys,
 'C_curve_kinds':{'GPS':'all available raw SVN and azimuth cuts; original equal-spacecraft reference median retained','Galileo':'all GRAP population-model azimuth cuts; not individual spacecraft'},
 'C_original_median_restored':True,'C_range_contains_archived_p05_p95_all_angles':True,
 'xlabel_bottoms_native_px':xlabel_bottoms,'xtick_bottoms_native_px':xtick_bottoms,'BC_bottom_labels_aligned':True,
 'font_family':'Arial','panel_letter_pt':8,'body_text_range_pt':[5.5,7],
 'C_subtitle_removed':True,'C_minmax_text_removed':True,'C_boundary_strokes_removed':True,'C_individual_curve_overlay_removed_for_uniform_fill':True,
 'C_range_uniform_opacity':.20,'C_median_label_background':'none; fully transparent',
 'C_ylabels':'Relative EIRP (dB)','C_signal_tags':signal_tag_checks,'C_median_label_colour':'#5C6470',
 'B_width_reduction_percent':(1-1450/1625)*100,'AB_translation_right_native_px':AB_SHIFT,'left_crop_native_px':CROP_LEFT,
 'C_native_geometry_and_font_sizes_unchanged':True,'B_legend_item_spacing_only_changed':True,
 'checked_text_artists':font_checks,'data_line_width_range_pt':[min(data_line_widths),max(data_line_widths)],
 'Nature_style_sources':['https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/','https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/'],
 'preview_png_not_final_journal_upload_format':True})
(R/'antenna_QA_R4.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in qa.items() if k not in ['text_boxes','source_hashes','C_direct_quantile_and_median_labels','checked_text_artists']},indent=2))



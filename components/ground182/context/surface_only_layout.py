"""Fixed-size publication typography for the observing-context figure."""
from pathlib import Path

W_MM, H_MM = 180.0, 135.0
FONT_SMALL, FONT_AXIS, FONT_TITLE, FONT_PANEL = 5.2, 6.0, 7.0, 8.0

def _rect(x, y, w, h):
    return [x / W_MM, y / H_MM, w / W_MM, h / H_MM]

def _format_orbit(ax):
    from matplotlib.offsetbox import AnnotationBbox
    icon_positions = {(1.0,708.0):(1.00,680.0), (1.22,648.0):(1.28,640.0),
                      (1.47,606.0):(1.63,620.0), (1.68,900.0):(1.68,870.0)}
    for artist in ax.artists:
        if isinstance(artist, AnnotationBbox):
            xy = tuple(float(v) for v in artist.xy)
            if xy in icon_positions:
                artist.xy = icon_positions[xy]
                artist.xybox = icon_positions[xy]
                artist.offsetbox.set_zoom(artist.offsetbox.get_zoom() * 0.80)
    name_positions = {'PlanetiQ':(1.00,730), 'Spire':(1.28,685),
                      'TSX':(1.63,665), 'MetOp':(1.68,925)}
    for text in ax.texts:
        value = text.get_text()
        if value in name_positions:
            text.set_position(name_positions[value])
        elif value.startswith('$h$'):
            text.set_bbox(None)
            text.set_text(value.replace(', ', '\n', 1))
            if '620' in value or '645' in value:
                text.set_position((0.90,570))
            elif 'varied' in value:
                text.set_position((1.28,425))
            elif '514' in value:
                text.set_position((1.63,555))
            elif '817' in value:
                text.set_position((1.68,755))
        elif value == 'day F2':
            text.set_position((0.65,360))
        elif value == 'night F':
            text.set_position((0.70,280))
    for line in ax.lines:
        if line.get_linewidth() > 1.0:
            line.set_linewidth(0.8)

def install_layout(m):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import pandas as pd
    from matplotlib import font_manager
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        portable_font = 'Arial'
    except ValueError:
        portable_font = 'DejaVu Sans'
    state = {}
    def build():
        m.OUT_DIR.mkdir(parents=True, exist_ok=True)
        old = m.load_old_module()
        old.POLAR_LAT_LIMIT = m.POLAR_LAT_LIMIT
        old.GROUND_MARKER_SIZE, old.COHERENT_MARKER_SIZE, old.ISR_MARKER_SIZE = 8.5,24,58
        old.COLORS['isr'] = '#1d4ed8'
        old.COVERAGE_ALPHA['isr_fill'] = 0.12
        plt.rcParams.update({'font.family':portable_font,'font.sans-serif':[portable_font],
            'font.size':FONT_SMALL,'axes.linewidth':0.5,'savefig.facecolor':'white',
            'figure.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
            'mathtext.fontset':'custom','mathtext.rm':portable_font,
            'mathtext.it':portable_font+':italic','mathtext.bf':portable_font+':bold'})
        ground = m.load_ground_roti_stations(old)
        ismr = m.load_updated_ismr_stations(old)
        isr = old.load_isr_stations().dropna(subset=['lat','lon'])
        coherent = m.load_current_superdarn_radars(old)
        lunar = m.load_luna_tracks()
        occultations = m.load_occultation_tracks()
        fig = plt.figure(figsize=(W_MM/25.4,H_MM/25.4),dpi=150,facecolor='white')
        a = fig.add_axes(_rect(5.8,74,50,50),projection=m.ccrs.NorthPolarStereo(central_longitude=m.POLAR_CENTRAL_LONGITUDE))
        b = fig.add_axes(_rect(65.8,74,50,50),projection=m.ccrs.SouthPolarStereo(central_longitude=m.POLAR_CENTRAL_LONGITUDE))
        c = fig.add_axes(_rect(130.5,74,47.5,50))
        d = fig.add_axes(_rect(2.2,20.5,56.8,29),projection=m.ccrs.Robinson(central_longitude=0))
        e = fig.add_axes(_rect(62.4,20.5,56.8,29),projection=m.ccrs.Robinson(central_longitude=0))
        f = fig.add_axes(_rect(122.6,20.5,56.8,29),projection=m.ccrs.Robinson(central_longitude=0))
        axes = dict(zip('abcdef',(a,b,c,d,e,f)))
        counts = {}
        counts['a'] = m.draw_polar_panel(old,a,'north',ground,ismr,isr,coherent)
        counts['b'] = m.draw_polar_panel(old,b,'south',ground,ismr,isr,coherent)
        counts['c'] = m.draw_ro_orbit_height_panel(c)
        _format_orbit(c)
        counts['d'] = m.draw_global_panel_nature(old,d,ground,ismr,isr,coherent)
        counts['e'] = m.draw_luna_track_panel_nature(old,e,lunar)
        counts['f'] = m.draw_occultation_track_panel_nature(old,f,occultations)
        for panel, ax in axes.items():
            x = ax.get_position().x0 * W_MM
            row_y = ax.get_position().y1 + (4.0 if panel in 'abc' else 2.5)/H_MM
            dx = ax.get_position().x0 - x/W_MM
            m.add_panel_label_fig(fig,ax,panel,row_y=row_y,dx=dx)
        m.add_external_title(fig,a,'North polar',dy=4.0/H_MM)
        m.add_external_title(fig,b,'South polar',dy=4.0/H_MM)
        m.add_external_title(fig,c,'Ionospheric layers and orbits',dy=4.0/H_MM)
        m.add_external_title(fig,d,'Global station distribution',dy=2.5/H_MM)
        m.add_external_title(fig,e,'LuGRE lunar-surface tracks',dy=2.5/H_MM)
        m.add_external_title(fig,f,'Daily LEO-RO occultation tracks',dy=2.5/H_MM)
        for panel,ax in [('a',a),('b',b)]:
            m.add_station_legend(ax,old,counts[panel],loc='upper center',anchor=(0.5,-0.084),
                                 ncol=2,include_context=True,fontsize=FONT_SMALL)
        d.legend(handles=m.station_legend_handles(old,counts['d'],include_context=False),
                 loc='upper center',bbox_to_anchor=(0.5,-0.10),frameon=False,
                 fontsize=FONT_SMALL,ncol=2,handlelength=0.8,handletextpad=0.22,
                 columnspacing=0.4,borderaxespad=0.0)
        e_bottom = e.get_position().y0
        note = fig.text(e.get_position().x0+e.get_position().width/2,
            e_bottom-3.2/H_MM,
            f"{counts['e']['Luna tangent tracks']:,} tracks; {counts['e']['Luna satellites']:,} satellites\nTangent height 50–1,000 km",
            ha='center',va='top',fontsize=FONT_SMALL,linespacing=1.45)
        note.set_gid('lunar-note')
        mission_handles = m.mission_legend_handles(occultations)
        mission_handles = [mission_handles[i] for i in (0,2,1,3)]
        f_legend = f.legend(handles=mission_handles,loc='upper center',bbox_to_anchor=(0.5,-0.10),
            frameon=False,fontsize=FONT_SMALL,ncol=2,handlelength=0.8,handletextpad=0.22,
            columnspacing=0.8,labelspacing=0.55,borderaxespad=0.0)
        for ax in (a,b):
            for text in ax.texts:
                if '180' in text.get_text():
                    text.set_position((0.018,0.5))
                    text.set_ha('left')
        for ax in (d,e,f):
            for text in ax.texts:
                if 'lambda_m' in text.get_text():
                    text.set_ha('right')
        fig.canvas.draw()
        for text in fig.findobj(Text):
            text.set_fontfamily('Arial')
            text.set_fontsize(FONT_SMALL)
            text.set_color('#222222')
        for ax in axes.values():
            ax.xaxis.label.set_fontsize(FONT_AXIS)
            ax.yaxis.label.set_fontsize(FONT_AXIS)
        for text in fig.texts:
            if str(text.get_gid()).startswith('panel-letter-'):
                text.set_fontsize(FONT_PANEL)
                text.set_fontweight('bold')
                text.set_fontstyle('normal')
                text.set_va('baseline')
            elif text.get_text() in ['North polar','South polar','Ionospheric layers and orbits','Global station distribution',
                                    'LuGRE lunar-surface tracks','Daily LEO-RO occultation tracks']:
                text.set_fontsize(FONT_TITLE)
            elif text.get_text() in ['North polar','South polar']:
                text.set_fontsize(FONT_AXIS)
        c.xaxis.labelpad=2
        c.yaxis.labelpad=2
        for spine in c.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('#333333')
        source_names = {'a':'North polar stations','b':'South polar stations',
            'c':'Ionospheric layers and RO orbit heights','d':'Global station distribution',
            'e':'Luna tangent-point tracks','f':'Daily LEO-RO occultation tracks by mission'}
        rows = [dict(panel=k,source=source_names[k],**{n:v for n,v in counts[k].items() if n!='summary'}) for k in 'abcdef']
        pd.DataFrame(rows).to_csv(m.OUT_TRACK_SUMMARY_CSV,index=False,encoding='utf-8-sig')
        state.update(figure=fig,axes=axes,lunar_note=note,mission_legend=f_legend)
        fig.savefig(m.OUT_NATURE_PNG,dpi=600,bbox_inches=None,pad_inches=0)
        plt.close(fig)
        return m.OUT_NATURE_PNG,m.OUT_NATURE_PDF
    m.build_nature_redraw = build
    return state

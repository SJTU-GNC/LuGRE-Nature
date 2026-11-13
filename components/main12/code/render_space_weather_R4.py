"""Review ED2 alone, preserving the frozen six-row observations and geometry."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
COMPONENT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MAIN12_OUTPUT',str(COMPONENT/'outputs')))/'space_weather'
ROOT.mkdir(parents=True,exist_ok=True)
RECIPE=COMPONENT/'code/space_weather_native_R4.py'
spec=importlib.util.spec_from_file_location('frozen_ED2_recipe',RECIPE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.ROOT=ROOT;base.OUT=ROOT
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def quantitative_signature(fig):
    signature = hashlib.sha256()
    for ax in fig.axes:
        signature.update(np.asarray([*ax.get_xlim(), *ax.get_ylim(), *ax.get_position().bounds],dtype='<f8').tobytes())
        for line in ax.lines:
            signature.update(np.asarray(line.get_xydata(),dtype='<f8').tobytes())
        for collection in ax.collections:
            if hasattr(collection, 'get_segments'):
                for segment in collection.get_segments():
                    signature.update(np.asarray(segment,dtype='<f8').tobytes())
        for patch in ax.patches:
            signature.update(np.asarray(patch.get_path().vertices,dtype='<f8').tobytes())
    return signature.hexdigest()

save(ROOT / 'FIGURE_CONTRACT_R4.json', {
    'revision':'R4', 'backend':'Python/matplotlib',
    'claim':'Place the lunar amplitude comparison beside the same measured space-weather context.',
    'archetype':'six-row quantitative time-series figure',
    'scope':'ED2 only; preserve original six rows, curve values, shading and OP means.',
    'style':{'width_mm':180,'height_mm':181,'dpi':600,'ordinary_font_pt':9,
             'secondary_font_pt':8.5,'panel_letter_font_pt':10,'family':'Arial',
             'frame_pt':.6,'frame_colour':'#262626'},
    'requested_changes':['Remove the three explanatory notes inside a,b,d',
                         'Complete all four axes borders',
                         'Increase e legend spacing along with its font size',
                         'f legend contains All latitudes and Polar only'],
    'frozen_recipe':str(RECIPE), 'formal_write_policy':'E-drive review only; do not modify D figures or PDFs.'
})

original_export = base.export
removed_notes = []

def export_review(fig, name, sources, science, original):
    assert name == 'Extended_Data_Space_weather_index.png'
    before = quantitative_signature(fig)
    fig.canvas.draw()
    notes = {'solar radio flux / EUV background proxy', 'southward IMF favors coupling',
             'polar-cap convection response'}
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_text() in notes:
                removed_notes.append(text.get_text())
                text.remove()
    assert set(removed_notes) == notes
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(text.get_fontsize()+2.)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.6)
            spine.set_color('#262626')
        ax.tick_params(labelsize=9, width=.6)
        label = next(text for text in ax.texts if text.get_ha() == 'right' and text.get_va() == 'center')
        # At 9pt, retain a clear gutter between horizontal quantities and ticks.
        label.set_x((18.-31.)/123.5)
    auroral = fig.axes[4]
    auroral.get_legend().remove()
    handles, labels = auroral.get_legend_handles_labels()
    assert labels == ['AE','AU','AL']
    auroral.legend(handles=handles,labels=labels,loc='lower left',bbox_to_anchor=(.005,1.035),
                   ncol=3,frameon=False,fontsize=9,handlelength=2.0,handletextpad=.8,
                   columnspacing=2.2,borderpad=0,borderaxespad=0)
    lunar = fig.axes[5]
    lunar.get_legend().remove()
    handles = [Line2D([],[],ls='',marker='o',mfc='white',mec='#8A8E94',ms=4,label='All latitudes'),
               Line2D([],[],ls='',marker='o',color='#B9453A',ms=4,label='Polar')]
    lunar.legend(handles=handles,loc='lower left',bbox_to_anchor=(.005,1.035),
                 ncol=2,frameon=False,fontsize=9,handlelength=1.0,handletextpad=.6,
                 columnspacing=2.2,borderpad=0,borderaxespad=0)
    fig.canvas.draw()
    assert quantitative_signature(fig) == before
    assert [text.get_text() for text in lunar.get_legend().get_texts()] == ['All latitudes','Polar']
    assert not any(text.get_text() in notes or text.get_text() == '60-s windows' for text in fig.findobj(mpl.text.Text))
    renderer = fig.canvas.get_renderer()
    e_legend = auroral.get_legend()
    e_gaps = []
    for handle, text in zip(e_legend.legend_handles,e_legend.get_texts()):
        e_gaps.append((text.get_window_extent(renderer).x0-handle.get_window_extent(renderer).x1)/fig.dpi*72)
    assert min(e_gaps) > 5
    science.update({'quantitative_artist_signature_before_and_after':before,
                    'curves_axes_OP_stems_and_Kp_spans_unchanged':True})
    report = original_export(fig,name,sources,science,original)
    report.update(status='ED2_REQUESTED_STYLE_REVIEW_READY',
                  font_roles_pt={'ordinary':9,'secondary':8.5,'panel_letters':10},
                  removed_explanatory_notes=removed_notes,
                  e_legend_handle_text_gap_pt=e_gaps,
                  e_legend_single_row=True,f_legend=['All latitudes','Polar'],
                  f_window_note_removed=True,
                  other_figures_modified=False,formal_manuscript_or_PDF_updated=False,
                  native_six_row_plotting_geometry_preserved=True)
    report['horizontal_y_label_right_edge_mm'] = 18.
    save(ROOT/'FIGURE_QA_R4.json',report)
    save(ROOT/'REVIEW_COMPLETION_R4.json',{
        'revision':'R4','status':'ED2_READY_FOR_INDIVIDUAL_REVIEW',
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'figure':report['figure'],'sha256':report['sha256'],
        'font_pt':report['font_roles_pt'],'dpi':600,'width_mm':180,
        'source_QA':str(ROOT/'FIGURE_QA_R4.json'),
        'removed_notes':removed_notes,'e_legend':labels,'f_legend':report['f_legend'],
        'data_and_six_row_layout_preserved':True,
        'formal_R4_figure_unchanged':sha(original)==report['baseline_sha256'],
        'PDF_updated':False})
    return report

base.export = export_review
if __name__ == '__main__':
    result = base.render_space()
    print(json.dumps({'status':result['status'],'figure':result['figure'],
                      'fonts':result['font_values_pt'],'outside':result['text_outside_canvas'],
                      'e_legend_gap_pt':result['e_legend_handle_text_gap_pt']},ensure_ascii=False))

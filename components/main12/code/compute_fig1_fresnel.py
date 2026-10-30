"""Verify the illustrative finite-source Fresnel curves embedded in Fig.1.

This is a theoretical geometry calculation, not LuGRE observations. Source
parameters are documented in the editable PowerPoint notes: transmitter-screen
distance26000km, L1/E1 and L5/E5a wavelengths, and screen-receiver distances inkm.
"""
from pathlib import Path
import argparse, csv, json, math, zipfile, xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'outputs/fig1')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
with zipfile.ZipFile(ROOT/'source_artwork/Fig1_R3.pptx') as archive:
    tree=ET.fromstring(archive.read('ppt/charts/chart1.xml'))
rows=[];errors=[];dt_m=26000*1000
for series in tree.findall('.//c:ser',ns):
    name=series.find('./c:tx//c:v',ns).text
    xx=[float(n.text) for n in series.findall('./c:xVal//c:numCache/c:pt/c:v',ns)]
    yy=[float(n.text) for n in series.findall('./c:yVal//c:numCache/c:pt/c:v',ns)]
    wavelength=299792458/(1575.42e6 if name.startswith('L1') else 1176.45e6)
    assert len(xx)==len(yy)
    for distance,expected in zip(xx,yy):
        effective=dt_m if 'asymptote' in name else dt_m*(distance*1000)/(dt_m+distance*1000)
        actual=math.sqrt(wavelength*effective)/1000
        errors.append(abs(actual-expected))
        rows.append({'series':name,'screen_receiver_distance_km':distance,'wavelength_m':wavelength,
            'effective_distance_km':effective/1000,'radius_km':actual,'pptx_cached_radius_km':expected})
assert max(errors)<2e-10,max(errors)
with (args.output/'Fig1_fresnel_radius_recomputed.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
qa={'status':'PASS','classification':'illustrative_finite_source_geometry_not_measurements',
    'transmitter_screen_distance_km':26000,'first_Fresnel_zone_radius':'sqrt(lambda*Dt*Dr/(Dt+Dr))',
    'series':len(tree.findall('.//c:ser',ns)),'points':len(rows),'maximum_error_from_PPTX_cache_km':max(errors),
    'notes':'Only cached scientific chart values are recomputed. The schematic artwork is separately editable in PowerPoint.'}
(args.output/'Fig1_fresnel_verification.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa))

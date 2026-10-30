from pathlib import Path
import importlib, importlib.metadata, json, platform, struct, sys
modules=['numpy','pandas','matplotlib','scipy','cartopy','pyarrow','PIL','openpyxl','pypdf','erfa']
status={}
for name in modules:
    print('Checking '+name+' ...',flush=True)
    try:
        m=importlib.import_module(name)
        status[name]={'status':'OK','version':getattr(m,'__version__','unknown')}
    except ImportError as exc:status[name]={'status':'MISSING','message':str(exc)}
result={'python':sys.version,'platform':platform.platform(),'bits':8*struct.calcsize('P'),'modules':status}
print(json.dumps(result,indent=2))
raise SystemExit(0 if all(v['status']=='OK' for v in status.values()) else 1)

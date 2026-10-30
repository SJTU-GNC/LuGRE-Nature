"""Packaging-only recipe, run on the source machine after bulk copying settles."""
from pathlib import Path
import json,hashlib,shutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/LUGRE/figures_R3/LuGRE_E23_h50_R3_20261028')
OUT=ROOT/'data/full_lugre';OUT.mkdir(parents=True,exist_ok=True)
native=['phase','op_id','sat','signal_id','signal_name','time_utc','rx_gps_seconds','cn0_dbhz','epoch_key',
        'lugre_lat_wgs84','lugre_lon_wgs84','lugre_h_tan_wgs84_km']
expected=['fit_eligible','fit_status','e23_local_arc_id','quartic_background_dbhz','e23_residual_db']
columns=native+expected
path=OUT/'native_cn0_verified_geometry_and_expected_results.parquet'
names=native+['expected_'+c for c in expected]
rows=0;writer=None
try:
    for batch in pq.ParquetFile(SOURCE/'data/all_LuGRE_E23_method_residuals.parquet').iter_batches(batch_size=100000,columns=columns):
        table=pa.Table.from_batches([batch]).rename_columns(names)
        if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd')
        writer.write_table(table);rows+=table.num_rows
finally:
    if writer is not None:writer.close()
assert rows==993093
for name in ['all_LuGRE_E23_fit_ledger.csv','all_LuGRE_E23_fit_ledger.json']:
    shutil.copyfile(SOURCE/'data'/name,OUT/name)
for name in ['refit_all_LuGRE_E23.py','verify_h50_candidate.py']:
    shutil.copyfile(SOURCE/'code'/name,ROOT/'provenance/original_code'/name)
shutil.copyfile(SOURCE/'qa/all_LuGRE_E23_method_QA.json',ROOT/'provenance/full_LuGRE_original_QA.json')
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
source_hash=digest(SOURCE/'data/all_LuGRE_E23_method_residuals.parquet')
qa={'source':str(SOURCE/'data/all_LuGRE_E23_method_residuals.parquet'),'source_sha256':source_hash,
    'expected_source_sha256':'98da6c8566682a5a4cc546c745fdc4fd44df2a9915d81fae41ae07c837805c1c',
    'rows':rows,'columns':names,'native_input_columns':native,'frozen_validation_columns':['expected_'+c for c in expected],
    'package':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path),
    'geometry_reused_as_independently_verified_input_not_recalculated':True,'scientific_values_unchanged':True}
assert source_hash==qa['expected_source_sha256']
(ROOT/'provenance/full_refit_input_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
print(json.dumps(qa),flush=True)

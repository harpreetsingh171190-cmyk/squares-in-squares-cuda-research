import json
from pathlib import Path

def test_notebook_json_and_integrity_cells():
    nb=json.loads((Path(__file__).resolve().parents[1]/'squares_cuda_professional.ipynb').read_text())
    assert nb['nbformat']==4
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    for term in ['contact_manifest.json','validate_candidate.py','roundtrip_validation.json','run_manifest.json','RECOMPUTE_SEARCH']:
        assert term in code

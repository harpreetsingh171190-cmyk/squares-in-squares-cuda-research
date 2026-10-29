import numpy as np
import pandas as pd
import pytest
from packing_ai.experiments import FEATURES
from packing_ai.v5 import offline_bandit_recommendations, save_candidate_json, gpu_inventory

def test_bandit_requires_real_rows():
    df=pd.DataFrame(columns=[*FEATURES,'objective'])
    with pytest.raises(ValueError): offline_bandit_recommendations(df,{k:(0,1) for k in FEATURES})

def test_bandit_candidate_is_unverified():
    rng=np.random.default_rng(42)
    df=pd.DataFrame(rng.random((32,5)),columns=FEATURES)
    df['objective']=df.sum(axis=1)
    result=offline_bandit_recommendations(df,{k:(0,1) for k in FEATURES},candidates=16)
    assert result['verified'] is False
    assert set(result['parameters'])==set(FEATURES)

def test_export_validation_flag(tmp_path):
    output=save_candidate_json({'verified':False},tmp_path/'candidate.json')
    assert 'UNVERIFIED' in output.read_text()

def test_gpu_inventory_shape():
    assert 'cuda_available' in gpu_inventory()

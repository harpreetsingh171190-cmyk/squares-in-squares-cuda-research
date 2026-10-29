import numpy as np
import pandas as pd
import pytest
from packing_ai.experiments import FEATURES
from packing_ai.v6 import dataset_audit,bootstrap_candidate_search,drift_report,queue_candidate,list_candidates,record_validation

def sample(n=24):
    rng=np.random.default_rng(42)
    df=pd.DataFrame(rng.random((n,len(FEATURES))),columns=FEATURES)
    df['objective']=df.sum(axis=1)
    return df

def test_audit():
    assert dataset_audit(sample())['ready_for_model']
    broken=sample();broken.loc[0,'objective']=np.nan
    assert not dataset_audit(broken)['ready_for_model']

def test_search_deterministic():
    df=sample(); bounds={k:[0,1] for k in FEATURES}
    a=bootstrap_candidate_search(df,bounds,16,3)
    b=bootstrap_candidate_search(df,bounds,16,3)
    assert a==b and all(x['validation_status']=='UNVERIFIED' for x in a)

def test_drift():
    df=sample(); result=drift_report(df,df)
    assert all(abs(x['standardized_mean_shift'])<1e-9 for x in result)

def test_registry(tmp_path):
    db=tmp_path/'registry.sqlite'; params={k:0.5 for k in FEATURES}
    cid=queue_candidate(db,params,1.23)
    assert queue_candidate(db,params)==cid
    assert list_candidates(db)[0]['status']=='UNVERIFIED'
    with pytest.raises(ValueError):record_validation(db,cid,'VALIDATED',1.0,'')
    record_validation(db,cid,'REJECTED',note='External geometry validator rejected overlap')
    assert list_candidates(db)[0]['status']=='REJECTED'
    with pytest.raises(ValueError):record_validation(db,cid,'VALIDATED',1.0,'external validation')

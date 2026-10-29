import pandas as pd
from packing_ai.advanced import data_quality,pareto_front,experiment_manifest

def test_pareto():
    df=pd.DataFrame({'a':[1,2,3,2],'b':[3,2,1,3]})
    assert set(pareto_front(df,['a','b']).index)=={0,1,2}

def test_quality():
    df=pd.DataFrame({c:[1.0]*8 for c in ('n_squares','rotation_fraction','density','restart_budget','step_scale','objective')})
    assert data_quality(df)['valid']
    assert len(experiment_manifest(df)['dataset_sha256'])==64

import numpy as np
import pandas as pd
import pytest
from packing_ai.experiments import FEATURES, TARGET
from packing_ai.v7 import offline_bandit_proposals, cpu_benchmark, monitoring_summary

def dataset():
    rng=np.random.default_rng(12)
    df=pd.DataFrame(rng.uniform(0.1,0.9,(30,len(FEATURES))),columns=FEATURES)
    df[TARGET]=np.arange(30,dtype=float)
    return df

def test_bandit_repeatable_and_unverified():
    df=dataset(); bounds={k:(0,1) for k in FEATURES}
    a=offline_bandit_proposals(df,bounds,seed=3)
    assert a==offline_bandit_proposals(df,bounds,seed=3)
    assert len(a)==10 and all(v['validation_status']=='UNVERIFIED' for v in a)

def test_bandit_invalid():
    with pytest.raises(ValueError): offline_bandit_proposals(dataset(),{k:(0,1) for k in FEATURES},epsilon=2)

def test_cpu_benchmark():
    r=cpu_benchmark(16,2)
    assert r['device']=='CPU' and r['median_ms']>=0

def test_monitoring():
    assert monitoring_summary(dataset())['measured_observations']==30

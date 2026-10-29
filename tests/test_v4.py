import numpy as np
import pandas as pd
import pytest
from packing_ai.experiments import FEATURES
from packing_ai.v4 import _bounds, evolutionary_suggestions, reinforcement_learning_status

def test_bounds():
    b={k:(0,1) for k in FEATURES}
    assert _bounds(b).shape==(5,2)
    b[FEATURES[0]]=(1,1)
    with pytest.raises(ValueError): _bounds(b)

def test_evolutionary_proposals_are_unverified():
    rng=np.random.default_rng(42)
    df=pd.DataFrame(rng.uniform(size=(40,5)),columns=FEATURES)
    df['objective']=df.sum(axis=1)
    bounds={k:(0.,1.) for k in FEATURES}
    for method in ['genetic','pso']:
        result=evolutionary_suggestions(df,bounds,method,8,3)
        assert result['verified'] is False
        assert len(result['history'])==3

def test_rl_disclaimer():
    assert reinforcement_learning_status()['status']=='experimental_design_only'

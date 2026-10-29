from pathlib import Path
import pandas as pd
import pytest
from packing_ai.experiments import load_experiments, train_surrogate, FEATURES

def test_schema_rejects_empty_template():
    with pytest.raises(ValueError): load_experiments(Path('inputs/experiment_schema.csv'))

def test_training_uses_measured_rows():
    df=pd.DataFrame([{**{k:float(i+j) for j,k in enumerate(FEATURES)},'objective':float(i*i)} for i in range(12)])
    _, metrics=train_surrogate(df)
    assert metrics['rows']==12 and metrics['mae']>=0

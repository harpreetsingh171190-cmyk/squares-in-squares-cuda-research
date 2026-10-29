import numpy as np
import pandas as pd
import pytest
from packing_ai.experiments import FEATURES, observed_bounds
from packing_ai.v4 import _bounds, evolutionary_suggestions
from packing_ai.v6 import queue_candidate, record_validation


def frame(n=30):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    df["objective"] = df.sum(axis=1)
    return df


def test_observed_bounds_handles_constant_feature():
    df = frame(); df[FEATURES[0]] = 5.0
    assert _bounds(observed_bounds(df)).shape == (5, 2)  # would raise before the fix


def test_pso_reports_best_position_found():
    result = evolutionary_suggestions(frame(), {k: (0., 1.) for k in FEATURES}, "pso", 12, 10)
    assert result["predicted_objective"] <= min(result["history"]) + 1e-12


def test_nonfinite_measured_objective_rejected(tmp_path):
    db = tmp_path / "r.sqlite"; cid = queue_candidate(db, {k: 0.5 for k in FEATURES})
    with pytest.raises(ValueError):
        record_validation(db, cid, "REJECTED", float("nan"), "external validator")

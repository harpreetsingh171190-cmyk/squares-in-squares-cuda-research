"""Surrogate learning, Bayesian search, and benchmarking; never bypasses official validation."""
from __future__ import annotations
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ('n_squares', 'rotation_fraction', 'density', 'restart_budget', 'step_scale')
TARGET = 'objective'

def load_experiments(path):
    df = pd.read_csv(path)
    missing = set(FEATURES + (TARGET,)) - set(df.columns)
    if missing:
        raise ValueError(f'Missing experiment columns: {sorted(missing)}')
    if len(df) < 8:
        raise ValueError('At least 8 measured experiments are required; no synthetic scores are generated.')
    for name in FEATURES + (TARGET,):
        df[name] = pd.to_numeric(df[name], errors='raise')
    if not np.isfinite(df[list(FEATURES) + [TARGET]].to_numpy()).all():
        raise ValueError('Nonfinite experiment features or objectives')
    return df

def observed_bounds(df, min_width=1e-6):
    """Observed (min, max) per feature; constant features get a tiny positive width so bounds stay valid."""
    out = {}
    for k in FEATURES:
        lo, hi = float(df[k].min()), float(df[k].max())
        if hi <= lo: hi = lo + min_width
        out[k] = [lo, hi]
    return out

def train_surrogate(df, model='random_forest'):
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    x, y = df[list(FEATURES)], df[TARGET]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=42)
    if model == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model == 'gradient_boosting':
        estimator = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError('model must be random_forest or gradient_boosting')
    estimator.fit(xtr, ytr)
    pred = estimator.predict(xte)
    return estimator, {'rows': len(df), 'holdout_rows': len(yte), 'mae': float(mean_absolute_error(yte, pred)), 'r2': float(r2_score(yte, pred)) if len(yte)>1 else None}

def train_deep_surrogate(df, epochs=100):
    """Optional MLP with scaling and held-out evaluation; requires torch."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    torch.manual_seed(42)
    xtr, xte, ytr, yte = train_test_split(df[list(FEATURES)].to_numpy(), df[TARGET].to_numpy(), test_size=.25, random_state=42)
    xs, ys = StandardScaler().fit(xtr), StandardScaler().fit(ytr.reshape(-1,1))
    xt = torch.tensor(xs.transform(xtr), dtype=torch.float32)
    yt = torch.tensor(ys.transform(ytr.reshape(-1,1)), dtype=torch.float32)
    network = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 64), torch.nn.ReLU(), torch.nn.Linear(64,32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    for _ in range(epochs):
        optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(network(xt), yt); loss.backward(); optimizer.step()
    with torch.no_grad():
        pred = ys.inverse_transform(network(torch.tensor(xs.transform(xte),dtype=torch.float32)).numpy()).ravel()
    return network, xs, ys, {'holdout_mae':float(mean_absolute_error(yte,pred)), 'epochs':epochs}

def bayesian_suggestions(df, bounds, n_trials=25, direction='minimize'):
    """Propose hyperparameters using Optuna TPE; objective is surrogate-only, NOT verified packing performance."""
    import optuna
    if direction not in ('minimize','maximize'): raise ValueError('Invalid direction')
    if set(bounds) != set(FEATURES): raise ValueError('Bounds must include exactly the five feature names')
    if any(float(bounds[k][1]) < float(bounds[k][0]) for k in FEATURES): raise ValueError('Each bound must satisfy low <= high')
    model, metrics = train_surrogate(df)
    def objective(trial):
        values = {k: trial.suggest_float(k, float(bounds[k][0]), float(bounds[k][1])) for k in FEATURES}
        return float(model.predict(pd.DataFrame([values]))[0])
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return {'parameters':study.best_params,'predicted_objective':study.best_value,'holdout_metrics':metrics,'verified':False,'warning':'Predicted only. Run the original solver and official geometry/CSV validator before admitting any candidate.'}

def gpu_benchmark(matrix_size=1024, iterations=10, warmup=3):
    """GPU matrix multiplication throughput; not a packing solver speedup measurement."""
    import torch
    if not torch.cuda.is_available():
        return {'available':False,'reason':'CUDA is unavailable; no GPU measurements made','platform':platform.platform()}
    device = torch.device('cuda')
    a=torch.randn(matrix_size,matrix_size,device=device); b=torch.randn_like(a)
    for _ in range(warmup): torch.mm(a,b)
    torch.cuda.synchronize(); durations=[]
    for _ in range(iterations):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); torch.mm(a,b); end.record(); torch.cuda.synchronize()
        durations.append(start.elapsed_time(end))
    med=float(np.median(durations))
    return {'available':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'matrix_size':matrix_size,'iterations':iterations,'warmup':warmup,'median_ms':med,'p95_ms':float(np.percentile(durations,95)),'estimated_tflops':2*matrix_size**3/(med/1000)/1e12,'note':'Synthetic FP32 GEMM benchmark; not the original CUDA packing kernel.'}

def save_json(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

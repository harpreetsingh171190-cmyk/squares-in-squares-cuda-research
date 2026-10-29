"""V5 research extensions: offline contextual bandit, SHAP, GPU inventory, tracking.
All candidate outputs are predictions, NOT geometrically validated packings.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import FEATURES, TARGET, train_surrogate
from .v4 import _bounds


def offline_bandit_recommendations(df, bounds, candidates=128, seed=42, exploration=0.5):
    """Conservative offline upper/lower-confidence exploration, not online RL.

    Ensemble disagreement is a heuristic uncertainty estimate, not calibrated confidence.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    if not 0 <= exploration <= 5 or not 8 <= candidates <= 10000:
        raise ValueError('Invalid exploration or candidate count')
    if len(df) < 12: raise ValueError('At least 12 measured experiments required')
    limits=_bounds(bounds)
    x=df[list(FEATURES)].to_numpy(dtype=float)
    y=df[TARGET].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite observations')
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.25,random_state=seed)
    model=RandomForestRegressor(n_estimators=128,min_samples_leaf=2,random_state=seed,n_jobs=-1)
    model.fit(xtrain,ytrain)
    rng=np.random.default_rng(seed)
    proposed=rng.uniform(limits[:,0],limits[:,1],size=(candidates,len(FEATURES)))
    predictions=np.stack([tree.predict(proposed) for tree in model.estimators_])
    mean=predictions.mean(axis=0); uncertainty=predictions.std(axis=0)
    acquisition=mean-exploration*uncertainty
    chosen=int(np.argmin(acquisition))
    return {'parameters':dict(zip(FEATURES,map(float,proposed[chosen]))),
            'predicted_objective':float(mean[chosen]),'ensemble_std_heuristic':float(uncertainty[chosen]),
            'acquisition':float(acquisition[chosen]),
            'holdout_mae':float(np.mean(np.abs(model.predict(xtest)-ytest))),
            'method':'offline ensemble exploration (NOT trained RL policy)',
            'verified':False,'requires_solver_validation':True}


def shap_explanations(df, max_rows=100, seed=42):
    """Optional SHAP tree explanations; only measured data, bounded sample size."""
    import shap
    if not 1 <= max_rows <= 500: raise ValueError('max_rows must be 1..500')
    model, metrics=train_surrogate(df)
    sample=df[list(FEATURES)].sample(n=min(len(df),max_rows),random_state=seed)
    vals=shap.TreeExplainer(model)(sample)
    arr=np.asarray(vals.values)
    return pd.DataFrame({'feature':FEATURES,'mean_abs_shap':np.mean(np.abs(arr),axis=0)}).sort_values('mean_abs_shap',ascending=False),metrics


def gpu_inventory():
    """Enumerate available CUDA GPUs; does not claim distributed execution."""
    try: import torch
    except ImportError: return {'cuda_available':False,'reason':'PyTorch not installed','devices':[]}
    if not torch.cuda.is_available(): return {'cuda_available':False,'devices':[]}
    devices=[]
    for i in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(i)
        devices.append({'index':i,'name':props.name,'total_memory_gib':round(props.total_memory/2**30,2),
                        'capability':f'{props.major}.{props.minor}'})
    return {'cuda_available':True,'devices':devices,'distributed_training_enabled':False}


def optuna_mlflow_study(df,bounds,n_trials=20,seed=42,tracking_uri=None):
    """Tune a surrogate-only candidate using Optuna; optional local MLflow tracking."""
    import optuna
    if not 1 <= n_trials <= 500: raise ValueError('n_trials must be 1..500')
    _bounds(bounds)
    model,metrics=train_surrogate(df)
    sampler=optuna.samplers.TPESampler(seed=seed)
    study=optuna.create_study(direction='minimize',sampler=sampler)
    def objective(trial):
        p={k:trial.suggest_float(k,float(bounds[k][0]),float(bounds[k][1])) for k in FEATURES}
        return float(model.predict(pd.DataFrame([p]))[0])
    study.optimize(objective,n_trials=n_trials)
    result={'parameters':study.best_params,'predicted_objective':float(study.best_value),
            'trials':len(study.trials),'holdout_metrics':metrics,'verified':False}
    if tracking_uri:
        from .v4 import log_mlflow_run
        result['mlflow_run_id']=log_mlflow_run('v5_optuna_surrogate',result,tracking_uri)
    return result


def save_candidate_json(result,output):
    """Export candidate with explicit validation status; never assert success."""
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    record={'schema_version':5,'validation_status':'UNVERIFIED','candidate':result}
    out.write_text(json.dumps(record,indent=2,default=str),encoding='utf-8')
    return out

"""V4 research utilities. All suggested candidates require independent solver validation."""
from __future__ import annotations
import json, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import FEATURES, TARGET, train_surrogate

def _bounds(bounds):
    if set(bounds) != set(FEATURES): raise ValueError('Exactly five feature bounds required')
    out=np.asarray([bounds[k] for k in FEATURES],dtype=float)
    if out.shape != (len(FEATURES),2) or not np.isfinite(out).all() or (out[:,1]<=out[:,0]).any():
        raise ValueError('Bounds must be finite and strictly increasing')
    return out

def evolutionary_suggestions(df,bounds,algorithm='genetic',population=24,generations=20,seed=42):
    """GA or PSO searches a trained surrogate; results are unverified predictions."""
    if population<4 or generations<1 or population>1000 or generations>10000: raise ValueError('Invalid search budget')
    limits=_bounds(bounds); lo,hi=limits[:,0],limits[:,1]
    model,metrics=train_surrogate(df); rng=np.random.default_rng(seed)
    x=rng.uniform(lo,hi,size=(population,len(FEATURES)))
    def score(v): return model.predict(pd.DataFrame(v,columns=FEATURES))
    history=[]
    if algorithm=='pso':
        velocity=np.zeros_like(x); best=x.copy(); best_score=score(x); global_best=best[np.argmin(best_score)].copy()
        for _ in range(generations):
            values=score(x); improved=values<best_score; best[improved]=x[improved]; best_score[improved]=values[improved]
            global_best=best[np.argmin(best_score)].copy(); history.append(float(np.min(best_score)))
            velocity=.65*velocity+1.4*rng.random(x.shape)*(best-x)+1.4*rng.random(x.shape)*(global_best-x)
            x=np.clip(x+velocity,lo,hi)
        values=score(x); improved=values<best_score; best[improved]=x[improved]; best_score[improved]=values[improved]
        x=best.copy()  # report the best positions found, not the last (possibly worse) swarm positions
    elif algorithm=='genetic':
        for _ in range(generations):
            values=score(x); order=np.argsort(values); history.append(float(values[order[0]]))
            elite=x[order[:max(2,population//4)]]
            children=[]
            while len(children)<population-len(elite):
                a,b=elite[rng.integers(len(elite),size=2)]
                child=np.where(rng.random(len(FEATURES))<.5,a,b)
                child=np.clip(child+rng.normal(0,.08,size=len(FEATURES))*(hi-lo),lo,hi)
                children.append(child)
            x=np.vstack([elite,children])
    else: raise ValueError('algorithm must be genetic or pso')
    values=score(x); idx=int(np.argmin(values))
    return {'algorithm':algorithm,'parameters':dict(zip(FEATURES,map(float,x[idx]))),'predicted_objective':float(values[idx]),'history':history,'holdout_metrics':metrics,'verified':False,'warning':'Surrogate estimate only. Run original packing solver and geometry validator.'}

def explain_surrogate(df,seed=42):
    """Permutation importance on a held-out split; avoids train-set leakage."""
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestRegressor
    xtr,xte,ytr,yte=train_test_split(df[list(FEATURES)],df[TARGET],test_size=.25,random_state=seed)
    model=RandomForestRegressor(n_estimators=150,random_state=seed,n_jobs=-1).fit(xtr,ytr)
    imp=permutation_importance(model,xte,yte,n_repeats=5,random_state=seed,scoring='neg_mean_absolute_error')
    return pd.DataFrame({'feature':FEATURES,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False)

def gpu_memory_profile(matrix_size=512,iterations=5):
    import torch
    if matrix_size<16 or matrix_size>8192 or iterations<1 or iterations>1000: raise ValueError('Invalid benchmark size')
    if not torch.cuda.is_available(): return {'available':False,'reason':'CUDA unavailable'}
    torch.cuda.reset_peak_memory_stats(); device='cuda'
    a=torch.randn((matrix_size,matrix_size),device=device); b=torch.randn_like(a)
    torch.cuda.synchronize(); start=time.perf_counter()
    for _ in range(iterations): torch.mm(a,b)
    torch.cuda.synchronize()
    return {'available':True,'gpu':torch.cuda.get_device_name(0),'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576,'note':'Synthetic GEMM memory benchmark, not packing solver memory usage'}

def save_experiment_run(path,method,result,metadata=None):
    """Local JSON tracking; optional MLflow integration can be enabled separately."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    record={'method':method,'result':result,'metadata':metadata or {},'created_unix':time.time(),'platform':platform.platform()}
    path.write_text(json.dumps(record,indent=2,default=str),encoding='utf8')
    return str(path)

def log_mlflow_run(method,result,tracking_uri=None):
    import mlflow
    if tracking_uri: mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=method) as run:
        mlflow.log_params({'method':method,'verified':False})
        for key,val in result.items():
            if isinstance(val,(int,float)) and not isinstance(val,bool) and np.isfinite(val): mlflow.log_metric(key,float(val))
        mlflow.set_tag('validation','surrogate_only')
        return run.info.run_id

def reinforcement_learning_status():
    return {'status':'experimental_design_only','reason':'No validated packing environment, action space or reward definition is supplied; RL training is intentionally not fabricated.','next_step':'Wrap the original solver as a Gymnasium environment and verify every proposed packing with official validator.'}

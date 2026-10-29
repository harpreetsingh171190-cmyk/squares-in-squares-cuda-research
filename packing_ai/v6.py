"""V6 experiment operations: safe candidate queues, uncertainty, drift, and provenance.
Predictions are not validated packings. No solver is invoked by this module.
"""
from __future__ import annotations
import hashlib, json, sqlite3, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import FEATURES, TARGET
from .v4 import _bounds

@contextmanager
def _connect(db_path):
    """Commit on success, roll back on error, and always close the connection."""
    con=sqlite3.connect(db_path)
    try:
        with con: yield con
    finally: con.close()

def dataset_audit(df):
    required=list(FEATURES)+( [TARGET] )
    missing=sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'Missing columns: {missing}')
    numeric=df[required].apply(pd.to_numeric,errors='coerce')
    issues={'rows':int(len(df)),'missing_or_non_numeric':{k:int(v) for k,v in numeric.isna().sum().items()},
            'infinite_cells':int(np.isinf(numeric.to_numpy()).sum()),'duplicate_configs':int(numeric.duplicated(subset=list(FEATURES)).sum()),
            'constant_features':[k for k in FEATURES if numeric[k].nunique(dropna=True)<2]}
    issues['ready_for_model']=len(df)>=12 and not any(issues['missing_or_non_numeric'].values()) and issues['infinite_cells']==0
    return issues

def bootstrap_candidate_search(df,bounds,n_candidates=128,n_bootstraps=12,seed=42,exploration=0.5):
    """Bootstrap ensemble lower confidence heuristic for MINIMIZATION; not calibrated intervals."""
    from sklearn.ensemble import RandomForestRegressor
    if not 16<=n_candidates<=5000 or not 3<=n_bootstraps<=64 or not 0<=exploration<=5:
        raise ValueError('Candidate count, bootstrap count or exploration outside supported range')
    audit=dataset_audit(df)
    if not audit['ready_for_model']: raise ValueError('Requires >=12 finite measured observations')
    limits=_bounds(bounds)
    rng=np.random.default_rng(seed)
    X=df[list(FEATURES)].to_numpy(float); y=df[TARGET].to_numpy(float)
    proposals=rng.uniform(limits[:,0],limits[:,1],(n_candidates,len(FEATURES)))
    predictions=[]
    for i in range(n_bootstraps):
        indices=rng.integers(0,len(X),len(X))
        m=RandomForestRegressor(n_estimators=40,min_samples_leaf=2,random_state=seed+i,n_jobs=1)
        m.fit(X[indices],y[indices]); predictions.append(m.predict(proposals))
    matrix=np.asarray(predictions)
    mean=matrix.mean(axis=0); std=matrix.std(axis=0)
    rank=mean-exploration*std
    winners=np.argsort(rank)[:min(10,n_candidates)]
    return [{'parameters':dict(zip(FEATURES,map(float,proposals[j]))),
             'predicted_objective':float(mean[j]),'bootstrap_std_heuristic':float(std[j]),
             'acquisition':float(rank[j]),'validation_status':'UNVERIFIED'} for j in winners]

def drift_report(reference,current):
    """Descriptive standardized mean shifts, not a calibrated statistical drift test."""
    for df in (reference,current):
        if len(df)<3 or not set(FEATURES).issubset(df): raise ValueError('Need >=3 rows and all feature columns')
    result=[]
    for k in FEATURES:
        a=pd.to_numeric(reference[k],errors='raise').to_numpy(float)
        b=pd.to_numeric(current[k],errors='raise').to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite features')
        sd=max(float(np.std(a)),1e-9)
        shift=float((np.mean(b)-np.mean(a))/sd)
        result.append({'feature':k,'standardized_mean_shift':shift,'abs_shift':abs(shift),'flag_2sigma':abs(shift)>=2})
    return sorted(result,key=lambda r:r['abs_shift'],reverse=True)

def init_registry(db_path):
    db=Path(db_path); db.parent.mkdir(parents=True,exist_ok=True)
    with _connect(db) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS candidates (
          id TEXT PRIMARY KEY, created_at REAL NOT NULL, parameters_json TEXT NOT NULL,
          predicted_objective REAL, status TEXT NOT NULL DEFAULT 'UNVERIFIED',
          measured_objective REAL, validation_note TEXT,
          CHECK (status IN ('UNVERIFIED','REJECTED','VALIDATED')) )""")
    return db

def queue_candidate(db_path,parameters,predicted_objective=None):
    if set(parameters)!=set(FEATURES): raise ValueError('Candidate must contain exactly the supported features')
    cleaned={k:float(parameters[k]) for k in FEATURES}
    if not all(np.isfinite(v) for v in cleaned.values()): raise ValueError('Nonfinite candidate')
    payload=json.dumps(cleaned,sort_keys=True,separators=(',',':'))
    key=hashlib.sha256(payload.encode()).hexdigest()[:24]
    pred=None if predicted_objective is None else float(predicted_objective)
    if pred is not None and not np.isfinite(pred): raise ValueError('Nonfinite prediction')
    init_registry(db_path)
    with _connect(db_path) as con:
        con.execute('INSERT OR IGNORE INTO candidates(id,created_at,parameters_json,predicted_objective) VALUES(?,?,?,?)',
                    (key,time.time(),payload,pred))
    return key

def record_validation(db_path,candidate_id,status,measured_objective=None,note=''):
    """Record EXTERNAL validator outcome only. This function does not validate geometry."""
    if status not in ('VALIDATED','REJECTED'): raise ValueError('Status must be VALIDATED or REJECTED')
    if status=='VALIDATED' and measured_objective is None:
        raise ValueError('Validated record requires finite measured objective')
    if measured_objective is not None and not np.isfinite(float(measured_objective)):
        raise ValueError('Measured objective must be finite')
    if not str(note).strip(): raise ValueError('Provide external validator provenance in note')
    init_registry(db_path)
    with _connect(db_path) as con:
        existing=con.execute('SELECT status FROM candidates WHERE id=?',(candidate_id,)).fetchone()
        if not existing: raise KeyError('Candidate ID not found')
        if existing[0]!='UNVERIFIED': raise ValueError('Candidate already finalized')
        con.execute('UPDATE candidates SET status=?,measured_objective=?,validation_note=? WHERE id=?',
                    (status,None if measured_objective is None else float(measured_objective),str(note),candidate_id))

def list_candidates(db_path):
    init_registry(db_path)
    with _connect(db_path) as con:
        rows=con.execute('SELECT id,created_at,parameters_json,predicted_objective,status,measured_objective,validation_note FROM candidates ORDER BY created_at DESC').fetchall()
    return [{'id':r[0],'created_at':r[1],'parameters':json.loads(r[2]),'predicted_objective':r[3],
             'status':r[4],'measured_objective':r[5],'validation_note':r[6]} for r in rows]

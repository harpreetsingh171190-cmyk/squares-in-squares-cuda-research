"""Research utilities; never replace the original packing geometry validator."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .experiments import FEATURES, TARGET

def data_quality(df):
    required=list(FEATURES)+[TARGET]
    missing=sorted(set(required)-set(df.columns))
    if missing: return {'valid':False,'missing_columns':missing}
    numeric=df[required].apply(pd.to_numeric,errors='coerce')
    invalid={c:int(numeric[c].isna().sum() + np.isinf(numeric[c].to_numpy()).sum()) for c in required}
    return {'valid':not any(invalid.values()) and len(df)>=8,'rows':len(df),'duplicate_rows':int(df.duplicated().sum()),'invalid_values':invalid,'missing_columns':[], 'minimum_rows_met':len(df)>=8}

def cross_validate(df, model='random_forest', folds=5):
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_validate as cv
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    if len(df)<max(8,folds): raise ValueError('Need at least max(8, folds) measured rows')
    if model=='random_forest': estimator=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
    elif model=='gradient_boosting': estimator=HistGradientBoostingRegressor(random_state=42)
    else: raise ValueError('Unknown model')
    result=cv(make_pipeline(SimpleImputer(),estimator),df[list(FEATURES)],df[TARGET],cv=KFold(n_splits=folds,shuffle=True,random_state=42),scoring={'mae':'neg_mean_absolute_error','r2':'r2'},error_score='raise')
    return {'folds':folds,'mae_mean':float(-result['test_mae'].mean()),'mae_std':float(result['test_mae'].std()),'r2_mean':float(result['test_r2'].mean()),'warning':'Cross-validation scores are estimates; no geometry is validated.'}

def pareto_front(df, objectives, minimize=None):
    """Return nondominated measured rows; all objectives numeric, finite."""
    objectives=list(objectives)
    if not objectives or any(c not in df for c in objectives): raise ValueError('Select existing objective columns')
    if minimize is None: minimize=[True]*len(objectives)
    if len(minimize)!=len(objectives): raise ValueError('Objective direction mismatch')
    a=df[objectives].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(a).all(): raise ValueError('Objectives must be finite')
    a=a*np.array([1 if v else -1 for v in minimize])
    keep=np.ones(len(df),dtype=bool)
    for i in range(len(df)):
        keep[i]=not np.any(np.all(a<=a[i],axis=1)&np.any(a<a[i],axis=1))
    return df.loc[keep].copy()

def experiment_manifest(df,config=None):
    csv=df.to_csv(index=False,float_format='%.12g').encode('utf-8')
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':hashlib.sha256(csv).hexdigest(),'rows':len(df),'columns':list(df.columns),'python':platform.python_version(),'platform':platform.platform(),'configuration':config or {},'validation_status':'UNVERIFIED: original packing solver and geometry validator required'}

def export_bundle(df,manifest,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/'experiments.csv',index=False)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    return folder

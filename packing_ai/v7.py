"""V7: offline contextual bandit simulation, safe CPU benchmarking, and monitoring.
All generated proposals remain UNVERIFIED until independently checked by the geometry solver.
"""
from __future__ import annotations
import json, time, platform
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .experiments import FEATURES, TARGET
from .v4 import _bounds
from .v6 import dataset_audit

def offline_bandit_proposals(df, bounds, n_proposals=10, n_rounds=100, seed=42, epsilon=0.15):
    """Epsilon-greedy bandit over feature-space regions using observed data only.
    This is NOT a physics environment or a trained deep RL agent.
    """
    if not 1 <= n_proposals <= 100 or not 10 <= n_rounds <= 10000 or not 0 <= epsilon <= 1:
        raise ValueError('Invalid bandit settings')
    if not dataset_audit(df)['ready_for_model']:
        raise ValueError('Need >=12 clean measured experiments')
    bounds_arr=_bounds(bounds)
    x=df[list(FEATURES)].to_numpy(dtype=float)
    y=df[TARGET].to_numpy(dtype=float)
    if np.any(bounds_arr[:,1] <= bounds_arr[:,0]): raise ValueError('Bounds must have positive widths')
    rng=np.random.default_rng(seed)
    scaled=np.clip((x-bounds_arr[:,0])/(bounds_arr[:,1]-bounds_arr[:,0]),0,1)
    region=np.minimum((scaled[:,0]*4).astype(int),3)
    counts=np.bincount(region,minlength=4).astype(float)
    scores=np.array([np.mean(y[region==j]) if counts[j] else np.inf for j in range(4)])
    if not np.isfinite(scores).any(): raise ValueError('No observed regions')
    selected=[]
    for _ in range(n_rounds):
        available=np.flatnonzero(counts>0)
        arm=int(rng.choice(available) if rng.random()<epsilon else available[np.argmin(scores[available])])
        selected.append(arm)
    proposals=[]
    for arm in selected[-n_proposals:]:
        lo=bounds_arr[:,0].copy(); hi=bounds_arr[:,1].copy()
        width=(hi[0]-lo[0])/4
        lo[0]+=arm*width; hi[0]=lo[0]+width
        sample=rng.uniform(lo,hi)
        proposals.append({'parameters':dict(zip(FEATURES,map(float,sample))),
                          'observed_region_mean':float(scores[arm]),'region':arm,
                          'validation_status':'UNVERIFIED'})
    return proposals

def cpu_benchmark(matrix_size=128, repetitions=5, seed=42):
    """CPU-only reproducible matrix multiply benchmark; never claims CUDA throughput."""
    if not 16<=matrix_size<=1024 or not 2<=repetitions<=100: raise ValueError('Benchmark settings out of range')
    rng=np.random.default_rng(seed)
    a=rng.random((matrix_size,matrix_size),dtype=np.float32)
    b=rng.random((matrix_size,matrix_size),dtype=np.float32)
    _=a@b
    latencies=[]
    for _ in range(repetitions):
        start=time.perf_counter(); result=a@b; latencies.append((time.perf_counter()-start)*1000)
    return {'device':'CPU','matrix_size':matrix_size,'repetitions':repetitions,
            'median_ms':float(np.median(latencies)),'p95_ms':float(np.percentile(latencies,95)),
            'checksum':float(result.sum()),'platform':platform.platform()}

def monitoring_summary(df):
    audit=dataset_audit(df)
    output={'quality':audit,'measured_observations':len(df),'generated_at_utc_epoch':time.time()}
    if audit['ready_for_model']:
        values=df[TARGET].to_numpy(float)
        output['objective']={'min':float(values.min()),'median':float(np.median(values)),
                             'max':float(values.max())}
    return output

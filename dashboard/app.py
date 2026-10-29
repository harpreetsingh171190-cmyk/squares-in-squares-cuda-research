"""Run: streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REGISTRY=str(ROOT/'artifacts'/'candidates_v6.sqlite')
import pandas as pd
import streamlit as st
from packing_ai.experiments import FEATURES, load_experiments, observed_bounds, train_surrogate, train_deep_surrogate, bayesian_suggestions, gpu_benchmark
from packing_ai.advanced import data_quality, cross_validate, pareto_front, experiment_manifest

st.set_page_config(page_title='CUDA Packing Research Lab',page_icon='📐',layout='wide')
st.title('📐 CUDA Packing — Research Dashboard')
st.caption('Experiment analysis and AI-guided proposals. Predicted results are not certified packing solutions.')
tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['Experiment analytics','ML & Bayesian optimization','GPU benchmark','Research QA & Pareto','Deep learning','V4 Optimization & Explainability'])
with tab1:
    upload=st.file_uploader('Upload measured experiment CSV',type='csv')
    if upload:
        try:
            df=load_experiments(upload)
            st.session_state['experiments']=df
            c1,c2,c3=st.columns(3)
            c1.metric('Experiments',len(df)); c2.metric('Best objective (min)',f'{df.objective.min():.6g}'); c3.metric('Mean objective',f'{df.objective.mean():.6g}')
            st.dataframe(df,use_container_width=True)
            st.line_chart(df['objective'])
            st.scatter_chart(df,x='density',y='objective',color='n_squares')
        except Exception as exc: st.error(str(exc))
    else: st.info('Provide real measurements. See inputs/experiment_schema.csv for the column schema (template only).')
with tab2:
    if 'experiments' not in st.session_state: st.info('Upload an experiment CSV on the first tab.')
    else:
        df=st.session_state['experiments']
        method=st.selectbox('Surrogate model',['random_forest','gradient_boosting'])
        if st.button('Train and evaluate surrogate'):
            try:
                _, metrics=train_surrogate(df,method); st.json(metrics)
            except Exception as exc: st.error(str(exc))
        st.subheader('Bayesian hyperparameter proposal (Optuna TPE)')
        trials=st.slider('Search trials',5,100,25)
        bounds=observed_bounds(df)
        st.caption('Search stays inside observed parameter ranges. Adjust these ranges in your source data.')
        if st.button('Propose next configuration'):
            try: st.json(bayesian_suggestions(df,bounds,trials))
            except ImportError: st.error('Install optional ML dependencies: pip install -r requirements-ml.txt')
            except Exception as exc: st.error(str(exc))
with tab3:
    n=st.select_slider('Matrix dimension',options=[256,512,1024,2048],value=1024)
    if st.button('Run CUDA benchmark'):
        try: st.json(gpu_benchmark(matrix_size=n))
        except ImportError: st.error('PyTorch is not installed. Follow official CUDA wheel instructions for your hardware.')
        except Exception as exc: st.error(str(exc))

with tab4:
    st.subheader('Data quality, multi-objective analysis & reproducibility')
    if 'experiments' not in st.session_state: st.info('Upload measured experiments on the first tab.')
    else:
        df=st.session_state['experiments']
        st.json(data_quality(df))
        numeric=[c for c in df.select_dtypes(include='number').columns if c not in FEATURES]
        objectives=st.multiselect('Pareto objectives (lower is better)',numeric,default=['objective'] if 'objective' in numeric else [])
        if objectives:
            try:
                frontier=pareto_front(df,objectives)
                st.metric('Nondominated measured rows',len(frontier))
                st.dataframe(frontier,use_container_width=True)
                st.download_button('Export Pareto CSV',frontier.to_csv(index=False),'pareto_front.csv','text/csv')
            except Exception as exc: st.error(str(exc))
        st.download_button('Download experiment manifest',json.dumps(experiment_manifest(df),indent=2),'manifest.json','application/json')
        if st.button('Run 5-fold cross-validation'):
            try: st.json(cross_validate(df))
            except Exception as exc: st.error(str(exc))
with tab5:
    st.subheader('Optional PyTorch neural surrogate')
    st.caption('Requires compatible PyTorch; training is CPU by default. Neural predictions are not validated packings.')
    if 'experiments' not in st.session_state: st.info('Upload measured experiments first.')
    else:
        epochs=st.slider('Training epochs',10,300,100,10)
        if st.button('Train neural network'):
            try:
                with st.spinner('Training neural surrogate...'):
                    _,_,_,metrics=train_deep_surrogate(st.session_state['experiments'],epochs=epochs)
                st.json(metrics)
            except ImportError: st.error('Install a compatible PyTorch build from pytorch.org.')
            except Exception as exc: st.error(str(exc))

with tab6:
    from packing_ai.v4 import evolutionary_suggestions, explain_surrogate, gpu_memory_profile, reinforcement_learning_status
    st.subheader('V4 evolutionary search and explainable AI')
    if 'experiments' in st.session_state:
        df=st.session_state['experiments']
        method=st.selectbox('Search method',['genetic','pso'])
        population=st.slider('Population',4,128,24)
        generations=st.slider('Generations',1,200,20)
        if st.button('Run evolutionary surrogate search'):
            try:
                bounds=observed_bounds(df)
                result=evolutionary_suggestions(df,bounds,method,population,generations)
                st.json(result); st.line_chart(result['history'])
            except Exception as exc: st.error(str(exc))
        if st.button('Explain ML features (held-out permutation importance)'):
            try: st.dataframe(explain_surrogate(df),use_container_width=True)
            except Exception as exc: st.error(str(exc))
    else: st.info('Upload real experiment data on the first tab.')
    st.subheader('GPU memory profiling')
    if st.button('Profile GPU memory (synthetic GEMM)'):
        try: st.json(gpu_memory_profile())
        except ImportError: st.error('Install a compatible PyTorch build first.')
        except Exception as exc: st.error(str(exc))
    st.subheader('Reinforcement learning readiness')
    st.json(reinforcement_learning_status())

st.divider()
st.header('V5 — AI Research & MLOps')
v5a,v5b,v5c=st.tabs(['Offline RL-style exploration','SHAP & Bayesian tracking','Multi-GPU inventory'])
with v5a:
    st.caption('Offline contextual-bandit-inspired exploration; not an actual trained reinforcement learning agent. All candidates require geometry validation.')
    if 'experiments' in st.session_state:
        from packing_ai.v5 import offline_bandit_recommendations
        exploration=st.slider('Exploration coefficient',0.0,3.0,0.5,0.1)
        candidates=st.slider('Candidates',16,512,128,16)
        if st.button('Propose V5 candidate'):
            try:
                df=st.session_state['experiments']
                bounds=observed_bounds(df)
                st.json(offline_bandit_recommendations(df,bounds,candidates=candidates,exploration=exploration))
            except Exception as exc: st.error(str(exc))
    else: st.info('Upload measured experiments first.')
with v5b:
    if 'experiments' in st.session_state:
        df=st.session_state['experiments']
        if st.button('Compute SHAP explanations (optional dependency)'):
            try:
                from packing_ai.v5 import shap_explanations
                explanation,metrics=shap_explanations(df)
                st.bar_chart(explanation.set_index('feature'))
                st.json(metrics)
            except ImportError: st.error('Install optional package: pip install shap')
            except Exception as exc: st.error(str(exc))
        trials=st.slider('Optuna trials (V5)',1,100,20)
        uri=st.text_input('Optional MLflow tracking URI (leave blank to disable)',value='')
        if st.button('Run V5 Optuna study'):
            try:
                from packing_ai.v5 import optuna_mlflow_study
                bounds=observed_bounds(df)
                st.json(optuna_mlflow_study(df,bounds,trials,tracking_uri=uri or None))
            except Exception as exc: st.error(str(exc))
    else: st.info('Upload measured experiments first.')
with v5c:
    from packing_ai.v5 import gpu_inventory
    if st.button('Inspect CUDA devices'):
        st.json(gpu_inventory())
    st.info('Multi-GPU discovery is implemented. Distributed training/execution requires a separately configured backend and is not enabled.')

st.divider()
st.header('V6 — Active Learning & Experiment Registry')
st.caption('Candidates are only model suggestions. Run the original solver and independent geometry validator before reporting measured improvements.')
from packing_ai.v6 import dataset_audit, bootstrap_candidate_search, drift_report, queue_candidate, list_candidates
v6a,v6b,v6c=st.tabs(['Bootstrap candidate queue','Dataset quality & drift','Candidate registry'])
with v6a:
    if 'experiments' in st.session_state:
        data=st.session_state['experiments']
        if st.button('Generate V6 unverified candidate shortlist'):
            try:
                limits=observed_bounds(data)
                candidates=bootstrap_candidate_search(data,limits)
                st.session_state['v6_candidates']=candidates
                st.dataframe(pd.DataFrame(candidates))
            except Exception as exc: st.error(str(exc))
        if st.session_state.get('v6_candidates'):
            index=st.selectbox('Candidate to queue',range(len(st.session_state['v6_candidates'])))
            if st.button('Queue selected candidate in local SQLite registry'):
                try:
                    item=st.session_state['v6_candidates'][index]
                    key=queue_candidate(REGISTRY,item['parameters'],item['predicted_objective'])
                    st.success(f'Queued as {key} — UNVERIFIED')
                except Exception as exc: st.error(str(exc))
    else: st.info('Upload real measured experiment CSV on the first tab.')
with v6b:
    if 'experiments' in st.session_state:
        st.json(dataset_audit(st.session_state['experiments']))
        upload=st.file_uploader('Upload a second measured CSV to compare feature drift',type='csv',key='v6_drift_csv')
        if upload is not None:
            try: st.dataframe(pd.DataFrame(drift_report(st.session_state['experiments'],pd.read_csv(upload))))
            except Exception as exc: st.error(str(exc))
    else: st.info('Upload measured experiment data first.')
with v6c:
    try:
        records=list_candidates(REGISTRY)
        if records: st.dataframe(pd.DataFrame(records))
        else: st.info('No candidates queued yet.')
        st.caption('Validation status must be recorded only after independent external geometry verification. No in-dashboard self-certification.')
    except Exception as exc: st.error(str(exc))


st.divider()
# V7: offline research exploration and CPU performance panel
from packing_ai.v7 import offline_bandit_proposals, cpu_benchmark, monitoring_summary
st.header('V7 — Offline Bandit & Research Monitoring')
v7a, v7b, v7c = st.tabs(['Offline bandit proposals','CPU benchmark','Monitoring'])
with v7a:
    st.caption('Offline epsilon-greedy region sampling. NOT a trained RL policy or validated packing.')
    if 'experiments' in st.session_state:
        rounds=st.slider('Offline bandit rounds',10,1000,100)
        if st.button('Generate V7 candidate proposals'):
            try:
                data=st.session_state['experiments']
                limits=observed_bounds(data)
                result=offline_bandit_proposals(data,limits,n_rounds=rounds)
                st.dataframe(pd.DataFrame(result))
                st.download_button('Export unverified proposals',json.dumps(result,indent=2),'v7_proposals.json','application/json')
            except Exception as exc: st.error(str(exc))
    else: st.info('Upload measured experiments first.')
with v7b:
    size=st.select_slider('CPU matrix dimension',options=[32,64,128,256,512],value=128)
    if st.button('Run CPU benchmark'):
        try: st.json(cpu_benchmark(size))
        except Exception as exc: st.error(str(exc))
    st.caption('This CPU benchmark does not measure CUDA or multi-GPU performance.')
with v7c:
    if 'experiments' in st.session_state:
        report=monitoring_summary(st.session_state['experiments'])
        st.json(report)
        st.download_button('Export monitoring JSON',json.dumps(report,indent=2),'v7_monitoring.json','application/json')
    else: st.info('Upload measured experiments first.')

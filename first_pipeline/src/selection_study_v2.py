"""Utilities for the separately frozen, exploratory classifier-selection study.

Training/validation code is in real notebook cells. This module materializes
ID-level partitions, verifies the freeze, saves outputs and draws figures.
It never changes the original experiments.py or its numerical results.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import balanced_accuracy_score

from src.experiments_v2 import ROOT, load_data, digest

OUT = ROOT / 'results/v2/selection_study'
FAMILIES = ['ManualSoftmaxClassifier', 'LinearSVC', 'ComplementNB', 'DecisionTreeClassifier']
DISPLAY = dict(zip(FAMILIES, ['Logistic', 'Linear SVM', 'Complement NB', 'Decision tree']))


def notebook_sources():
    import nbformat
    book = nbformat.read(ROOT/'notebooks/first_pipeline.ipynb', as_version=4)
    return {cell.metadata['selection_study_tag']: cell.source for cell in book.cells
            if cell.cell_type == 'code' and 'selection_study_tag' in cell.metadata}


def serial_params(params):
    return {key: (repr(value) if key == 'classifier' else value) for key, value in params.items()}


def freeze_selection(data, parameter_grid):
    """Freeze a NEW extension before its fits; earlier analyses already exist."""
    source_hashes = {tag: hashlib.sha256(source.encode()).hexdigest()
                     for tag, source in notebook_sources().items() if tag in {'candidates', 'training'}}
    candidates = [serial_params(p) for p in ParameterGrid(parameter_grid)]
    identity = {'dataset_sha256': digest(ROOT/'data/processed/model_dataset_v2.csv'),
                'original_experiment_sha256': digest(ROOT/'src/experiments_v2.py'),
                'extension_source_sha256': digest(__file__), 'manual_source_sha256':digest(ROOT/'src/manual_models.py'), 'notebook_cell_sha256': source_hashes,
                'candidate_parameters': candidates}
    target = ROOT/'config/model_selection.json'
    if target.exists():
        old = json.loads(target.read_text())
        for key, value in identity.items():
            assert old[key] == value, f'Changed after extension freeze: {key}'
        return old
    assert not (OUT/'scores.csv').exists(), 'Cannot freeze after extension results exist.'
    plans = []
    ids = data.record_id.to_numpy(); y = data.rating.to_numpy()
    for seed in [156, 1156, 2156]:
        dev, test = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(ids, y))
        inner = list(StratifiedKFold(5, shuffle=True, random_state=seed+10000).split(ids[dev], y[dev]))
        train, validation = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=seed+20000).split(ids[dev], y[dev]))
        for a,b in [(dev,test), (dev[train],dev[validation])]+[(dev[a],dev[b]) for a,b in inner]:
            assert not set(a)&set(b)
            assert set(y[a]) == set(y[b]) == {1,2,3,4}
        counts = lambda index: {str(k):int(sum(y[index] == k)) for k in [1,2,3,4]}
        plans.append({'seed':seed, 'development_ids':ids[dev].tolist(), 'test_ids':ids[test].tolist(),
                      'single_train_ids':ids[dev[train]].tolist(), 'single_validation_ids':ids[dev[validation]].tolist(),
                      'cv_splits':[{'train_ids':ids[dev[a]].tolist(), 'validation_ids':ids[dev[b]].tolist()} for a,b in inner],
                      'counts':{'development':counts(dev), 'test':counts(test), 'single_train':counts(dev[train]), 'single_validation':counts(dev[validation])}})
    protocol = {**identity, 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'exploratory204-title revision after prior203-title results; NOT pristine confirmation',
        'before_extension_fits':True, 'conditions':['genres','combined'], 'families':FAMILIES,
        'candidate_count_per_condition':len(candidates), 'primary_score':'balanced_accuracy',
        'tie_rule':'first maximum: logistic, SVM, NB, tree; ascending C/depth; descending alpha',
        'selection':'mean five-fold development BA OR single development validation BA; test never selects',
        'refit':'four family finalists per strategy on 163 development titles; cached single fits on130 retained',
        'fit_count':480, 'test_reuse':'same41 per seed across families/conditions/strategies; seeds overlap',
        'plans':plans}
    target.write_text(json.dumps(protocol, indent=2)+'\n')
    OUT.mkdir(parents=True, exist_ok=True)
    return protocol


def indices(data, plan):
    lookup = {value:i for i,value in enumerate(data.record_id)}
    dev = np.array([lookup[v] for v in plan['development_ids']]); test = np.array([lookup[v] for v in plan['test_ids']])
    local = {value:i for i,value in enumerate(plan['development_ids'])}
    inner = [(np.array([local[v] for v in s['train_ids']]), np.array([local[v] for v in s['validation_ids']])) for s in plan['cv_splits']]
    train = np.array([local[v] for v in plan['single_train_ids']]); val = np.array([local[v] for v in plan['single_validation_ids']])
    return dev, test, inner, train, val


def save_results(data, protocol, searches, choices, prediction_rows, fit_count):
    assert fit_count == protocol['fit_count'] == 480
    searches=pd.DataFrame(searches); choices=pd.DataFrame(choices); predictions=pd.DataFrame(prediction_rows)
    assert len(searches)==144 and len(choices)==48 and len(predictions)==2952
    assert choices.groupby(['seed','condition','strategy']).selected.sum().eq(1).all()
    assert predictions.groupby(['seed','condition','strategy','family']).size().eq(41).all()
    scores=[]
    for keys, group in predictions.groupby(['seed','condition','strategy','family'],sort=False):
        seed,condition,strategy,family=keys
        assert group.record_id.is_unique and set(group.record_id)==set(next(p for p in protocol['plans'] if p['seed']==seed)['test_ids'])
        scores.append(dict(zip(['seed','condition','strategy','family'],keys), BA=balanced_accuracy_score(group.true,group.prediction),
                           selected=bool(group.selected.iloc[0]),n_train=int(group.n_train.iloc[0]),n_test=len(group),
                           setting=group.setting.iloc[0]))
    scores=pd.DataFrame(scores)
    # Four families x three strategies x two conditions x three seeds.
    assert len(scores)==72
    for name, frame in [('development_search',searches),('development_choices',choices),('predictions',predictions),('scores',scores)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    selected=scores.loc[scores.selected]
    paired=selected.pivot(index=['seed','strategy'],columns='condition',values='BA')
    paired['combined_minus_genres']=paired.combined-paired.genres
    paired.to_csv(OUT/'paired_selected_scores.csv')
    audit={'fit_count':fit_count,'development_searches':len(searches),'family_choices':len(choices),'test_predictions':len(predictions),
           'score_rows':len(scores),'primary_results_untouched':True,
           'protocol_sha256':digest(ROOT/'config/model_selection.json'),
           'notebook_cell_sha256':protocol['notebook_cell_sha256'],
           'test_used_for_selection':False,'all_partitions_and_class_support_checked':True}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return searches,choices,predictions,scores,paired



"""Frozen, paired small-data experiments. No account access or remote data calls.

The notebook executes this module once after freeze_protocol() has saved all
choices and actual ID-level partitions. Only genres and synopsis are features.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.manual_models import ManualSoftmaxClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import sklearn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v2'
LABELS = np.array([1, 2, 3, 4])
NAMES = {'majority':'Majority', 'genres':'Genres', 'text':'Descriptions (TF-IDF)',
         'combined':'Genres + TF-IDF', 'topics':'Descriptions (LDA)', 'genre_topics':'Genres + LDA'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    data = pd.read_csv(ROOT/'data/processed/model_dataset_v2.csv', keep_default_na=False).sort_values('record_id').reset_index(drop=True)
    assert len(data) == 204 and data.record_id.is_unique
    assert set(data.rating) == set(LABELS) and data.rating.value_counts().to_dict() == {1:45,2:72,3:56,4:31}
    assert data.genres.str.len().min() and data.description_text.str.len().min()
    return data


def genre_tokens(text):
    return [value.strip() for value in text.split(',') if value.strip()]


def make_pipeline(condition):
    """Unit-normalize each block, concatenate, then normalize the total vector."""
    genres = Pipeline([('tokens',CountVectorizer(analyzer=genre_tokens, binary=True, dtype=np.float64)), ('norm',Normalizer())])
    text = TfidfVectorizer(lowercase=True, min_df=2, max_features=3000,
                           ngram_range=(1,1), smooth_idf=True, sublinear_tf=False, norm='l2')
    topics = Pipeline([('counts',CountVectorizer(lowercase=True, min_df=2, max_features=3000, stop_words='english')),
                       ('lda',LatentDirichletAllocation(n_components=5, learning_method='batch',
                                max_iter=20, random_state=156, n_jobs=1)), ('norm',Normalizer())])
    blocks=[]
    if condition in {'genres','combined','genre_topics'}: blocks.append(('genres',genres,'genres'))
    if condition in {'text','combined'}: blocks.append(('text',text,'description_text'))
    if condition in {'topics','genre_topics'}: blocks.append(('topics',topics,'description_text'))
    if not blocks: raise ValueError(condition)
    return Pipeline([('features',ColumnTransformer(blocks, sparse_threshold=1.0, remainder='drop')),
                     ('norm',Normalizer()),
                     ('classifier',ManualSoftmaxClassifier(C=1.0, class_weight='balanced', max_iter=2000))])


def _check_split(y, train, test, groups=None):
    assert not set(train) & set(test)
    assert set(y[train]) == set(LABELS) and set(y[test]) == set(LABELS)
    if groups is not None: assert not set(groups[train]) & set(groups[test])


def _plans(data, grouped=False):
    y=data.rating.to_numpy()
    ids=data.record_id.to_numpy()
    group_doc=json.loads((ROOT/'data/processed/franchise_groups_v2.json').read_text())
    groups=np.array([group_doc['group_by_record_id'][value] for value in ids])
    seeds=[156] if grouped else [156,1156,2156]
    choices=[(5,3),(4,3),(3,3),(3,2)] if grouped else [(5,3)]
    for outer_k,inner_k in choices:
        plans=[]
        try:
            for repeat,seed in enumerate(seeds):
                outer = StratifiedGroupKFold(outer_k,shuffle=True,random_state=seed) if grouped else StratifiedKFold(outer_k,shuffle=True,random_state=seed)
                for fold,(train,test) in enumerate(outer.split(ids,y,groups if grouped else None)):
                    _check_split(y,train,test,groups if grouped else None)
                    inner_seed=seed+100*(fold+1)
                    inner=StratifiedGroupKFold(inner_k,shuffle=True,random_state=inner_seed) if grouped else StratifiedKFold(inner_k,shuffle=True,random_state=inner_seed)
                    splits=list(inner.split(ids[train],y[train],groups[train] if grouped else None))
                    for a,b in splits: _check_split(y[train],a,b,groups[train] if grouped else None)
                    plans.append({'repeat':repeat,'seed':seed,'fold':fold,'train_ids':ids[train].tolist(),
                        'test_ids':ids[test].tolist(),'train_counts':{str(k):int(sum(y[train]==k)) for k in LABELS},
                        'test_counts':{str(k):int(sum(y[test]==k)) for k in LABELS},
                        'inner_splits':[{'train_ids':ids[train][a].tolist(),'test_ids':ids[train][b].tolist()} for a,b in splits]})
            return {'outer_folds':outer_k,'inner_folds':inner_k,'splits':plans}
        except AssertionError:
            continue
    raise ValueError('No predeclared valid grouped partition; inspect groups before fitting.')


def notebook_training_hashes():
    book=json.loads((ROOT/'notebooks/first_pipeline.ipynb').read_text())
    return {cell['metadata']['completion_tag']:hashlib.sha256(''.join(cell['source']).encode()).hexdigest()
            for cell in book['cells'] if cell.get('metadata',{}).get('completion_tag') in {'manual_learning','predictive_evaluation'}}


def freeze_protocol():
    """Call before any model fitting; a completed protocol cannot be overwritten."""
    OUT.mkdir(parents=True,exist_ok=True)
    target=ROOT/'config/evaluation.json'
    if target.exists():
        existing=json.loads(target.read_text())
        assert existing['notebook_training_sha256']==notebook_training_hashes()
        assert existing['manual_source_sha256']==digest(ROOT/'src/manual_models.py')
        assert existing['dataset_sha256']==digest(ROOT/'data/processed/model_dataset_v2.csv')
        assert existing['groups_sha256']==digest(ROOT/'data/processed/franchise_groups_v2.json')
        assert existing['experiment_source_sha256']==digest(Path(__file__)), 'Experiment code changed after protocol freeze.'
        return existing
    assert not (OUT/'scores.csv').exists(), 'Do not declare choices after inspecting results.'
    data=load_data()
    protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'before_any_model_fitting':False,'before_protocol_evaluation_fits':True,
        'pre_freeze_checks':'three fixed-C full-data solver convergence checks, no held-out scoring or selection; synthetic numerical checks separately recorded','version':2,'status':'exploratory revision after204thrating; previous203results archived unchanged',
        'manual_source_sha256':digest(ROOT/'src/manual_models.py'),'notebook_training_sha256':notebook_training_hashes(),
        'dataset_sha256':digest(ROOT/'data/processed/model_dataset_v2.csv'),
        'groups_sha256':digest(ROOT/'data/processed/franchise_groups_v2.json'),
        'experiment_source_sha256':digest(Path(__file__)),
        'n':len(data),'class_counts':{str(k):int(sum(data.rating==k)) for k in LABELS},
        'feature_whitelist':['genres','description_text'],'primary_score':'balanced_accuracy',
        'primary_contrast':'combined minus genres','main_conditions':['majority','genres','text','combined'],
        'C_grid':[0.1,1.0,10.0],'tie_rule':'first maximum; ascending C favors stronger regularization on exact ties',
        'classifier':'manual multiclass cross-entropy/gradient; balanced weights; L2; manual adaptive gradient/backtracking; tol1e-6; max_iter2000',
        'text':'lowercase unigrams; min_df2; max_features3000; smoothed IDF; raw TF; no stop-word removal; L2',
        'scaling':'genre block L2; text block L2; concatenated vector L2; no tuned block weights',
        'primary':_plans(data),'grouped':_plans(data,True),
        'secondary_LDA':{'seed':156,'topics':5,'iterations':20,'input':'counts; english stop words; min_df2; max_features3000',
            'conditions':['topics','genre_topics'],'C_grid':[0.1,1.0,10.0],
            'reason':'Predeclared bounded alternative to TF-IDF requested by James; no topic-count search or outer-score selection.'},
        'learning_curve':{'seed':156,'conditions':['genres','text','combined'],'fractions':[0.4,0.7,1.0],
            'C':1.0,'purpose':'fixed-procedure diagnostic; no tuning or model selection from curves'},
        'within_genre_rule':'all broad genres with >=40 eligible titles; descriptive paired correct/incorrect counts, overlapping groups',
        'uncertainty':'pooled OOF BA for each repetition; full range of three paired deltas describes split sensitivity; no naive CI or p-value',
        'worked_example_id':'tt0032553','example_rule':'The Great Dictator in first repetition, held-out combined model; fixed by personal opening before scores',
        'environment':{'sklearn':sklearn.__version__,'requirements_sha256':digest(ROOT/'requirements.lock.txt')}}
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def _indices(data,plan):
    lookup={value:i for i,value in enumerate(data.record_id)}
    train=np.array([lookup[value] for value in plan['train_ids']]); test=np.array([lookup[value] for value in plan['test_ids']])
    inner_lookup={value:i for i,value in enumerate(plan['train_ids'])}
    inner=[(np.array([inner_lookup[v] for v in split['train_ids']]),np.array([inner_lookup[v] for v in split['test_ids']])) for split in plan['inner_splits']]
    return train,test,inner


def _example(estimator,data,index,plan):
    row=data.iloc[[index]]
    block=estimator['features'].transform(row[['genres','description_text']])
    vector=estimator['norm'].transform(block)
    names=estimator['features'].get_feature_names_out()
    dense=vector.toarray()[0] if hasattr(vector,'toarray') else vector[0]
    clf=estimator['classifier']; scores=clf.decision_function(vector)[0]; probabilities=clf.predict_proba(vector)[0]
    np.testing.assert_allclose(softmax(scores),probabilities)
    predicted=int(clf.classes_[np.argmax(probabilities)])
    true=int(row.rating.iloc[0]); contrast=true if predicted!=true else int(clf.classes_[np.argsort(scores)[-2]])
    weights=clf.coef_[np.where(clf.classes_==predicted)[0][0]]-clf.coef_[np.where(clf.classes_==contrast)[0][0]]
    terms=[{'feature':str(names[i]),'x':float(dense[i]),'coefficient_difference':float(weights[i]),'contribution':float(dense[i]*weights[i])} for i in np.flatnonzero(dense)]
    terms.sort(key=lambda value:abs(value['contribution']),reverse=True)
    text=estimator['features'].named_transformers_['text']
    return {'record_id':row.record_id.iloc[0],'title':row.title.iloc[0],'true_rating':true,'predicted_rating':predicted,
        'contrast_rating':contrast,'fold':plan['fold'],'seed':plan['seed'],'held_out':True,
        'description':row.description_text.iloc[0],'genres':row.genres.iloc[0],
        'scores':dict(zip(map(str,clf.classes_),map(float,scores))),
        'probabilities':dict(zip(map(str,clf.classes_),map(float,probabilities))),
        'feature_count':len(names),'text_vocabulary_size':len(text.vocabulary_),
        'optimization_loss_history':clf.loss_history_,'optimization_final_max_gradient':clf.gradient_max_,
        'nonzero_terms':terms,'intercept_difference':float(clf.intercept_[np.where(clf.classes_==predicted)[0][0]]-clf.intercept_[np.where(clf.classes_==contrast)[0][0]])}


def evaluate(data,protocol,mode='primary'):
    plans=protocol['grouped']['splits'] if mode=='grouped' else protocol['primary']['splits']
    if mode=='topics': plans=[p for p in plans if p['seed']==156]
    conditions=protocol['secondary_LDA']['conditions'] if mode=='topics' else protocol['main_conditions']
    X=data[['genres','description_text']]; y=data.rating.to_numpy()
    predictions=[]; fits=[]; search_rows=[]
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for plan in plans:
        train,test,inner=_indices(data,plan)
        for condition in conditions:
            if condition=='majority':
                model=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),y[train])
                pred=model.predict(np.zeros((len(test),1))); prob=model.predict_proba(np.zeros((len(test),1)))
                C=None; feature_count=0; text_size=0; zero_rows=0
            else:
                search=GridSearchCV(make_pipeline(condition),{'classifier__C':protocol['C_grid']},
                                    scoring='balanced_accuracy',cv=inner,refit=True,n_jobs=1,error_score='raise')
                search.fit(X.iloc[train],y[train]); model=search.best_estimator_
                pred=model.predict(X.iloc[test]); prob=model.predict_proba(X.iloc[test]); C=search.best_params_['classifier__C']
                feature_count=int(model['classifier'].n_features_in_)
                text_size=0; zero_rows=0
                if condition in {'text','combined'}:
                    tfidf=model['features'].named_transformers_['text']; text_size=len(tfidf.vocabulary_)
                    zero_rows=int(sum(np.asarray(tfidf.transform(X.iloc[test].description_text).getnnz(axis=1))==0))
                for params,mean,std in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):
                    search_rows.append({'mode':mode,'repeat':plan['repeat'],'seed':plan['seed'],'fold':plan['fold'],'condition':condition,'C':params['classifier__C'],'inner_mean_BA':float(mean),'inner_fold_sd_descriptive':float(std)})
                if mode=='primary' and plan['seed']==156 and condition=='combined':
                    ix=np.where(data.record_id.to_numpy()==protocol['worked_example_id'])[0][0]
                    if ix in test: (OUT/'worked_example.json').write_text(json.dumps(_example(model,data,ix,plan),indent=2)+'\n')
            fits.append({'mode':mode,'repeat':plan['repeat'],'seed':plan['seed'],'fold':plan['fold'],'condition':condition,'C':C,'features':feature_count,'text_vocabulary':text_size,'zero_text_test_rows':zero_rows,'train_n':len(train),'test_n':len(test),'test_BA':float(balanced_accuracy_score(y[test],pred))})
            for offset,index in enumerate(test):
                result={'mode':mode,'repeat':plan['repeat'],'seed':plan['seed'],'fold':plan['fold'],'condition':condition,'record_id':data.record_id.iloc[index],'title':data.title.iloc[index],'true_rating':int(y[index]),'prediction':int(pred[offset])}
                result.update({f'p{k}':float(prob[offset,j]) for j,k in enumerate(LABELS)})
                predictions.append(result)
        print(f'{mode}: seed {plan["seed"]}, fold {plan["fold"]+1} complete',flush=True)
    predictions=pd.DataFrame(predictions); fits=pd.DataFrame(fits)
    predictions.to_csv(OUT/f'{mode}_predictions.csv',index=False); fits.to_csv(OUT/f'{mode}_folds.csv',index=False)
    pd.DataFrame(search_rows).to_csv(OUT/f'{mode}_inner_search.csv',index=False)
    scores=[]
    for (seed,condition),group in predictions.groupby(['seed','condition']):
        assert group.record_id.nunique()==len(data) and len(group)==len(data)
        matrix=confusion_matrix(group.true_rating,group.prediction,labels=LABELS)
        scores.append({'mode':mode,'seed':int(seed),'condition':condition,'BA':float(balanced_accuracy_score(group.true_rating,group.prediction)),
            **{f'recall_{k}':float(matrix[j,j]/matrix[j].sum()) for j,k in enumerate(LABELS)}})
    return pd.DataFrame(scores)


def learning_curves(data,protocol):
    results=[]; X=data[['genres','description_text']]; y=data.rating.to_numpy()
    for plan in [p for p in protocol['primary']['splits'] if p['seed']==156]:
        train,test,_=_indices(data,plan)
        for fraction in protocol['learning_curve']['fractions']:
            if fraction==1.0: subset=train
            else:
                a,_=next(StratifiedShuffleSplit(1,train_size=fraction,random_state=156+plan['fold']).split(train,y[train])); subset=train[a]
            assert set(y[subset])==set(LABELS)
            for condition in protocol['learning_curve']['conditions']:
                model=make_pipeline(condition).fit(X.iloc[subset],y[subset])
                results.append({'condition':condition,'fold':plan['fold'],'fraction':fraction,'n':len(subset),
                    'train_BA':float(balanced_accuracy_score(y[subset],model.predict(X.iloc[subset]))),
                    'validation_BA':float(balanced_accuracy_score(y[test],model.predict(X.iloc[test])))})
    frame=pd.DataFrame(results);frame.to_csv(OUT/'learning_curves.csv',index=False);return frame


def summarize(data,scores):
    scores.to_csv(OUT/'scores.csv',index=False)
    primary=scores.query('mode=="primary"')
    table=primary.groupby('condition').BA.agg(['mean','min','max']).reindex(['majority','genres','text','combined'])
    table.to_csv(OUT/'comparison.csv')
    deltas={}
    for mode in ['primary','grouped']:
        wide=scores.query('mode==@mode').pivot(index='seed',columns='condition',values='BA')
        difference=wide.combined-wide.genres
        deltas[mode]={'by_seed':{str(int(k)):float(v) for k,v in difference.items()},'mean':float(difference.mean()),'min':float(difference.min()),'max':float(difference.max())}
    raw=pd.read_csv(OUT/'primary_predictions.csv'); first=raw[raw.seed==156]
    pair=first[first.condition=='genres'][['record_id','title','true_rating','prediction']].rename(columns={'prediction':'genres_prediction'}).merge(first[first.condition=='combined'][['record_id','prediction']],on='record_id').rename(columns={'prediction':'combined_prediction'})
    pair['genre_correct']=pair.genres_prediction==pair.true_rating; pair['combined_correct']=pair.combined_prediction==pair.true_rating
    pair['transition']=np.select([pair.genre_correct & pair.combined_correct,~pair.genre_correct & pair.combined_correct,pair.genre_correct & ~pair.combined_correct],['both correct','corrected by text','broken by text'],default='both wrong')
    pair.to_csv(OUT/'paired_errors.csv',index=False)
    genre_rows=[]
    for genre in sorted({g for values in data.genres for g in genre_tokens(values)}):
        ids=set(data.loc[data.genres.map(lambda value:genre in genre_tokens(value)),'record_id'])
        if len(ids)<40: continue
        subset=pair[pair.record_id.isin(ids)]
        genre_rows.append({'genre':genre,'n':len(subset),**{f'n_rating{k}':int(sum(subset.true_rating==k)) for k in LABELS},'genre_correct':int(subset.genre_correct.sum()),'combined_correct':int(subset.combined_correct.sum()),'corrected':int(sum(subset.transition=='corrected by text')),'broken':int(sum(subset.transition=='broken by text'))})
    pd.DataFrame(genre_rows).to_csv(OUT/'within_genre.csv',index=False)
    diagnostics={'n':len(data),'deltas':deltas,'primary_mean_BA':table['mean'].to_dict(),
        'first_seed_error_transitions':pair.transition.value_counts().to_dict(),
        'first_seed_by_class':{str(int(k)):g.transition.value_counts().to_dict() for k,g in pair.groupby('true_rating')},
        'near_vs_extreme':{condition:{'adjacent_errors':int(sum(abs(group.true_rating-group.prediction)==1)),
             'two_level_errors':int(sum(abs(group.true_rating-group.prediction)==2)),
             'extreme_errors':int(sum(abs(group.true_rating-group.prediction)==3))} for condition,group in first.groupby('condition')},
        'caution':'Repeat ranges describe split sensitivity, not independent replicates or confidence intervals. Secondary grouped map is conservative and incomplete.'}
    (OUT/'summary.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    return table,diagnostics


def describe_topics(data):
    """All-data descriptive fit AFTER prediction evaluation; no score selected."""
    model=make_pipeline('topics'); features=model['features']; counts_lda=features.fit_transform(data[['genres','description_text']])
    pipeline=features.named_transformers_['topics']; words=pipeline['counts'].get_feature_names_out(); lda=pipeline['lda']
    rows=[]
    for k,weights in enumerate(lda.components_):
        rows.append({'topic':k+1,'top_words':', '.join(words[np.argsort(weights)[-8:][::-1]])})
    pd.DataFrame(rows).to_csv(OUT/'lda_topics_descriptive.csv',index=False)
    return pd.DataFrame(rows)


def run_all():
    data=load_data(); protocol=freeze_protocol()
    scores=pd.concat([evaluate(data,protocol,'primary'),evaluate(data,protocol,'grouped'),evaluate(data,protocol,'topics')],ignore_index=True)
    curve=learning_curves(data,protocol)
    comparison,summary=summarize(data,scores)
    topics=describe_topics(data)
    return {'data':data,'protocol':protocol,'scores':scores,'comparison':comparison,'summary':summary,'learning_curves':curve,'topics':topics}

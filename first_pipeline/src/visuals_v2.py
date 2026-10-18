"""Scientific figures generated directly from the executed experiment outputs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .experiments_v2 import ROOT, OUT, NAMES, LABELS

COLORS={'majority':'#82909a','genres':'#326795','text':'#bc712d','combined':'#39816c'}


def _save(fig,name):
    path=OUT/'figures'/f'{name}.png';path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    return path


def make_figures(result):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titlesize':11,'axes.labelsize':10})
    data=result['data']; figures={}
    fig,ax=plt.subplots(figsize=(11,4.2));ax.set(xlim=(-.15,11),ylim=(0,4.2));ax.axis('off')
    boxes=[(0.05,2.65,2.35,1.05,'Preserved sources\nIMDb 302; HBO 36\nDisney+ 30\nNetflix 622 events'),
           (2.85,2.65,2.25,1.05,'335 observed works\nIncludes 2 Chaplin films\nExclude unseen pilot'),
           (5.55,2.65,2.15,1.05,'Human rating ledger\n334 candidates\n204 rated; 130 excluded'),
           (8.15,2.65,2.5,1.05,'Paired nested CV\n5 outer × 3 inner folds\n3 prespecified repetitions'),
           (2.85,.6,2.25,1.05,'Inside training folds\nLearn genre vocabulary\nand word/IDF features'),
           (5.55,.6,2.15,1.05,'Same classifier family\nGenres; text; combined\nC∈{0.1,1,10}'),
           (8.15,.6,2.5,1.05,'Held-out predictions\nBalanced accuracy +errors\nCombined minus genres')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc='#eef3f6',ec='#9bafbc',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for a,b in [((2.5,3.17),(2.7,3.17)),((5.2,3.17),(5.4,3.17)),((7.8,3.17),(8.0,3.17)),((3.97,2.1),(3.97,1.8)),((5.2,1.12),(5.4,1.12)),((7.8,1.12),(8.0,1.12))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#425969'))
    ax.plot([9.4,9.4,3.97],[2.52,2.1,2.1],color='#425969',lw=1)
    ax.text(.1,.9,'130 zeros never enter\ntraining or vocabulary fitting.',fontsize=9,color='#425969')
    figures['pipeline']=_save(fig,'pipeline')

    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    counts=data.rating.value_counts().reindex(LABELS)
    axes[0].bar([str(k) for k in LABELS],counts,color='#326795',width=.6)
    for j,n in enumerate(counts):axes[0].text(j,n+1,str(n),ha='center')
    axes[0].set(xlabel='Increasing liking rating',ylabel='Title/series observations',title='Four rating categories, 204 titles',ylim=(0,82))
    lengths=data.description_text.str.findall(r"\b\w+(?:['’−-]\w+)*\b").str.len()
    axes[1].hist(lengths,bins=np.arange(5,57,5),color='#39816c',edgecolor='white')
    axes[1].axvline(lengths.median(),color='#243f50',ls='--',label=f'Median {lengths.median():.0f} words')
    axes[1].set(xlabel='Words in an IMDb synopsis',ylabel='Titles',title='Short descriptions limit word evidence');axes[1].legend(frameon=False)
    fig.tight_layout();figures['data_profile']=_save(fig,'data_profile')

    comparison=result['comparison']
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),gridspec_kw={'width_ratios':[1.2,1]})
    for j,(condition,row) in enumerate(comparison.iterrows()):
        axes[0].errorbar(row['mean']*100,j,xerr=np.array([[row['mean']-row['min']],[row['max']-row['mean']]])*100,
                         color=COLORS[condition],fmt='o',capsize=4,ms=7)
    axes[0].set_yticks(range(4),[NAMES[value] for value in comparison.index]);axes[0].invert_yaxis()
    axes[0].axvline(25,color='#9b9b9b',ls=':',label='Constant-category BA=25%')
    axes[0].set(xlabel='Balanced accuracy (%)',title='Mean and range across 3 repetitions')
    axes[0].grid(axis='x',alpha=.15)
    axes[0].text(.01,-.2,'Dotted line: constant-category BA = 25%',transform=axes[0].transAxes,fontsize=8,color='#58656e')
    delta=result['summary']['deltas']['primary']['by_seed']
    vals=np.array(list(delta.values()))*100
    axes[1].barh(range(len(vals)),vals,color=['#39816c' if v>=0 else '#b65949' for v in vals],height=.55)
    axes[1].axvline(0,color='#56616a',lw=1)
    axes[1].set_yticks(range(len(vals)),[f'Seed {v}' for v in delta]);axes[1].invert_yaxis()
    axes[1].set(xlabel='Combined − genres (percentage points)',title='Does text add predictive value?')
    fig.tight_layout();figures['primary_comparison']=_save(fig,'primary_comparison')

    pred=pd.read_csv(OUT/'primary_predictions.csv');pred=pred[pred.seed==156]
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,condition in zip(axes,['genres','text','combined']):
        subset=pred[pred.condition==condition];matrix=confusion_matrix(subset.true_rating,subset.prediction,labels=LABELS)
        shares=matrix/matrix.sum(axis=1,keepdims=True)
        ax.imshow(shares,cmap='Blues',vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]}\n{shares[i,j]:.0%}',ha='center',va='center',color='white' if shares[i,j]>.55 else '#1b2933',fontsize=9)
        ax.set(xticks=range(4),xticklabels=LABELS,yticks=range(4),yticklabels=[f'{k} (n={n})' for k,n in zip(LABELS,matrix.sum(axis=1))],xlabel='Predicted liking',ylabel='Actual liking',title=NAMES[condition])
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Same 204 held-out titles: counts and within-rating percentages (seed 156)',y=1.02)
    fig.tight_layout();figures['ordered_confusions']=_save(fig,'ordered_confusions')

    curve=result['learning_curves'];fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
    for ax,condition in zip(axes,['genres','text','combined']):
        group=curve[curve.condition==condition].groupby('fraction').agg(n=('n','mean'),train=('train_BA','mean'),validation=('validation_BA','mean'))
        ax.plot(group.n,group.train*100,'o--',color='#89949a',label='Training')
        ax.plot(group.n,group.validation*100,'o-',color=COLORS[condition],label='Held out')
        ax.axhline(25,color='#aaa',ls=':',lw=.8)
        ax.set(xlabel='Training titles (mean across 5 folds)',title=NAMES[condition],ylim=(0,105))
        ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Balanced accuracy (%)')
    fig.suptitle('Fixed C=1 learning curves: diagnostic only; no model selection',y=1.02)
    fig.tight_layout();figures['learning_curves']=_save(fig,'learning_curves')

    within=pd.read_csv(OUT/'within_genre.csv').sort_values('n')
    fig,ax=plt.subplots(figsize=(8,4.5));y=np.arange(len(within))
    ax.barh(y-.17,within.corrected,height=.3,label='Genre errors corrected by text',color='#39816c')
    ax.barh(y+.17,within.broken,height=.3,label='Genre successes lost after adding text',color='#b65949')
    ax.set_yticks(y,[f'{r.genre} (n={r.n})' for r in within.itertuples()])
    ax.set(xlabel='Number of titles, first prespecified repetition',title='Paired changes within broad genres (groups overlap)')
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();figures['within_genre']=_save(fig,'within_genre')

    example=json.loads((OUT/'worked_example.json').read_text())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.bar([str(k) for k in LABELS],[example['probabilities'][str(k)]*100 for k in LABELS],color=['#39816c' if k==example['true_rating'] else '#92adbf' for k in LABELS])
    ax.set(xlabel='Rating category',ylabel='Weighted-model probability (%)',ylim=(0,100),title=f'{example["title"]}: actual {example["true_rating"]}, predicted {example["predicted_rating"]}')
    fig.tight_layout();figures['worked_probabilities']=_save(fig,'worked_probabilities')
    (OUT/'figure_paths.json').write_text(json.dumps({key:str(path.relative_to(ROOT)) for key,path in figures.items()},indent=2)+'\n')
    return figures

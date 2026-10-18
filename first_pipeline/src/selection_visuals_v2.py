"""Presentation-only plots for the saved exploratory study; no fitting."""
import numpy as np
import pandas as pd
from src.selection_study_v2 import OUT, FAMILIES, DISPLAY

def make_selection_figures(data, scores):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    paths=[]
    fig, axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    counts=data.rating.value_counts().sort_index()
    axes[0].bar(counts.index,counts,color='#326a9a')
    for rating,n in counts.items():axes[0].text(rating,n+1,str(n),ha='center')
    axes[0].set(xlabel='Increasing liking category',ylabel='Titles',title='204 explicit ratings',xticks=[1,2,3,4],ylim=(0,82))
    lengths=data.description_text.str.split().str.len()
    axes[1].hist(lengths,bins=np.arange(0,61,10),color='#398570',edgecolor='white')
    axes[1].axvline(lengths.median(),color='black',ls='--',label='Median: 27 words')
    axes[1].legend(frameon=False);axes[1].set(xlabel='Whitespace-delimited words',ylabel='Titles',title='Short synopsis text')
    genres=data.assign(genre=data.genres.str.split(',')).explode('genre', ignore_index=True)
    genres['genre']=genres.genre.str.strip()
    support=pd.crosstab(genres.genre,genres.rating).loc[genres.genre.value_counts().head(8).index]
    support.iloc[::-1].plot.barh(stacked=True,ax=axes[2],color=['#557fa3','#86aec4','#78a891','#335f50'],width=.8)
    axes[2].set(xlabel='Overlapping title memberships',ylabel='',title='Genre composition')
    axes[2].legend(title='Rating',ncol=4,loc='lower right',fontsize=8)
    path=folder/'eda.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    fig,ax=plt.subplots(figsize=(9.5,4.6),layout='constrained')
    for offset,strategy,color,label in [(-.12,'cv_refit','#326a9a','Five-fold selection; refit 163'),(.12,'single_refit','#b34538','Single validation; refit 163')]:
        for j,family in enumerate(FAMILIES):
            values=scores.query('condition == "combined" and strategy == @strategy and family == @family').BA.to_numpy()*100
            ax.scatter(np.full(3,j+offset),values,color=color,alpha=.8,s=38,label=label if j==0 else None)
            ax.plot([j+offset-.08,j+offset+.08],[values.mean()]*2,color=color,lw=3)
    ax.axhline(25,color='gray',ls=':',label='Majority BA 25%')
    ax.set(xticks=range(4),xticklabels=[DISPLAY[f] for f in FAMILIES],ylabel='Common-test balanced accuracy (%)',ylim=(10,55),title='Development-tuned families: genres + TF-IDF')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.13),ncol=2,frameon=False)
    path=folder/'classifier_comparison.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    order=['cv_refit','single_refit','single_no_refit']
    for ax,condition,title in zip(axes,['genres','combined'],['Genres','Genres + TF-IDF']):
        for seed,color in zip([156,1156,2156],['#326a9a','#b34538','#398570']):
            values=scores.query('selected and condition == @condition and seed == @seed').set_index('strategy').loc[order].BA*100
            ax.plot(range(3),values,marker='o',label=f'Seed {seed}',color=color)
        ax.axhline(25,color='gray',ls=':')
        ax.set(xticks=range(3),xticklabels=['CV\nrefit 163','Single\nrefit 163','Single\nfit 130'],ylabel='Common-test balanced accuracy (%)',title=title,ylim=(10,55))
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.17),ncol=3,frameon=False)
    path=folder/'validation_strategy.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    return paths

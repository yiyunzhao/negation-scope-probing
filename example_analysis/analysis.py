
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pickle
import seaborn as sns
import scipy.stats as stats
from pathlib import Path

def load_pickle(d):
    with open(d,'rb') as f:
        data = pickle.load(f)
    return data    
# available data ('bert-base','bert-large','roberta-base','roberta-large')
def load_data(model):
    parent_path = str(Path().absolute().parent)
    data = ['story-pretrained','story-neg','story-control','review-pretrained','review-neg','review-control']
    attention_scores = {}
    for d in data:
        data_path = parent_path+'/finetuned-data/'+model+'/'+d+'-'+model+'.pickle'
        attention_scores[d] = load_pickle(data_path)
    return attention_scores


# ploting the pretrain and consistency
def plotting_pretrain(data,measure='f1'): # measures: precision, recall, f1
    measures = {"precision":0, "recall":1, "f1":2}
    m = measures[measure]
    story_pretrain, review_pretrain = data['story-pretrained'],data['review-pretrained']
    ln,hn = story_pretrain.shape[:-1]
    fig, ax =plt.subplots(1,2,figsize=(14,6))
    f1= sns.heatmap(story_pretrain[:,:,m],ax=ax[0],vmax=0.8)
    f2=sns.heatmap(review_pretrain[:,:,m],ax=ax[1],vmax=0.8)
    f1.set(xlabel='head',ylabel='layer',title='ConanDolye-neg',xticklabels=np.arange(1,ln+1), yticklabels=np.arange(1,hn+1))
    f1.set_yticklabels(f1.get_yticklabels(), rotation=0, horizontalalignment='right')
    f1.set_xticklabels(f1.get_xticklabels(), rotation=90, horizontalalignment='left')
    f2.set(xlabel='head',ylabel='layer',title='SFU-review',xticklabels=np.arange(1,ln+1), yticklabels=np.arange(1,hn+1))
    f2.set_yticklabels(f2.get_yticklabels(), rotation=0, horizontalalignment='right')
    f2.set_xticklabels(f2.get_xticklabels(), rotation=90, horizontalalignment='left')
    ax[0].set_ylim(0,ln+1)
    ax[1].set_ylim(0,ln+1)
    sns.set(color_codes=False,style="whitegrid") 
    fig.subplots_adjust(bottom=0.15,left=0.07,right=0.97)
    plt.show()
    tau, p_value = stats.kendalltau(story_pretrain[:,:,m].flatten(), review_pretrain[:,:,m].flatten())
    print('tau coefficient:',tau,'p_value',p_value)

# average performance measure
def overall_performance(data,run=False):
    measures = ["precision","recall","f1"]
    story_pretrain, story_neg, story_control = data['story-pretrained'],data['story-neg'],data['story-control']
    for mindx, m in enumerate(measures):
        print('MEASURE:',m)
        print('pretrain:', np.average(story_pretrain[:,:, mindx])*100)
        runs=[np.average(i[:,:,mindx]) for i in story_neg]
        print('negation task:', np.average(runs)*100, np.std(runs)*100)
        if run: print('individual runs:', runs)
        runs=[np.average(i[:,:,mindx]) for i in story_control]
        print('control task:', np.average(runs)*100, np.std(runs)*100)
        if run: print('individual runs:', runs)
        print('--------------------')

# individual head change
def head_change(data,measure = 'f1'):
    measures = {"precision":0, "recall":1, "f1":2}
    m = measures[measure]
    story_pretrain, story_neg, story_control = data['story-pretrained'],data['story-neg'],data['story-control']
    neg_change = np.average(story_neg[:,:,:,m],axis=0)- story_pretrain[:,:,m]
    control_change = np.average(story_control[:,:,:,m],axis=0)- story_pretrain[:,:,m]
    ln,hn = story_pretrain.shape[:-1]
    fig, ax =plt.subplots(1,2,figsize=(14,6))
    f1= sns.heatmap(neg_change,ax=ax[0],vmin=-0.2,vmax=0.2,center=0)
    f2=sns.heatmap(control_change,ax=ax[1],vmin=-0.2,vmax=0.2,center=0)
    f1.set(xlabel='head',ylabel='layer',title='negation task',xticklabels=np.arange(1,ln+1), yticklabels=np.arange(1,hn+1))
    f1.set_yticklabels(f1.get_yticklabels(), rotation=0, horizontalalignment='right')
    f1.set_xticklabels(f1.get_xticklabels(), rotation=90, horizontalalignment='left')
    f2.set(xlabel='head',ylabel='layer',title='control task',xticklabels=np.arange(1,ln+1), yticklabels=np.arange(1,hn+1))
    f2.set_yticklabels(f2.get_yticklabels(), rotation=0, horizontalalignment='right')
    f2.set_xticklabels(f2.get_xticklabels(), rotation=90, horizontalalignment='left')
    ax[0].set_ylim(0,ln+1)
    ax[1].set_ylim(0,ln+1)
    print(ln,hn)
    sns.set(color_codes=False,style="whitegrid") 
    fig.subplots_adjust(bottom=0.15,left=0.07,right=0.97)
    plt.show()
    
def rich_hypothesis(data, measure='f1',runs=False):
    measures = {"precision":0, "recall":1, "f1":2}
    m = measures[measure]
    story_pretrain, story_neg = data['story-pretrained'],data['story-neg']
    print('MEASURE:',measure)
    neg_taus = []
    neg_ps = []
    for i in range(10):
        tau,p=stats.kendalltau(story_pretrain[:,:,m], story_neg[i,:,:,m]-story_pretrain[:,:,m])
        if runs: print('run:'+str(i+1), 'tau:',tau, '  ','p-value:',p)
        neg_taus.append(tau)
        neg_ps.append(p<.05)
    print('negation task:', 'average coefficient:', np.average(neg_taus),np.std(neg_taus),'sig.frac:', np.sum(neg_ps)/len(neg_ps))
    print('--------------')
      
# consistency measure
def consistency(data,measure='f1', runs = False):
    measures = {"precision":0, "recall":1, "f1":2}
    m = measures[measure]
    story_pretrain, story_neg, story_control = data['story-pretrained'],data['story-neg'],data['story-control']
    review_pretrain, review_neg, review_control = data['review-pretrained'],data['review-neg'],data['review-control']
    tau,p=stats.kendalltau(story_pretrain[:,:,m], review_pretrain[:,:,m])
    print('MEASURE:',measure)
    print('Pretrain:', 'tau coeffiecienty:', tau,'  ', 'p-value:', p)
    print('--------------') 
    neg_taus = []
    neg_ps = []
    for i in range(10):
        tau,p=stats.kendalltau(story_neg[i,:,:,m], review_neg[i,:,:,m])
        if runs: print('negation run:'+str(i+1), 'tau:',tau, '  ','p-value:',p)
        neg_taus.append(tau)
        neg_ps.append(p<.05)
    print('negation task:', 'average coefficient:', np.average(neg_taus),np.std(neg_taus),'sig.frac:', np.sum(neg_ps)/len(neg_ps))
    print('--------------')
    contr_taus = []
    contr_ps = []
    for i in range(10):
        tau,p=stats.kendalltau(story_control[i,:,:,m], review_control[i,:,:,m])
        if runs: print('control run:'+str(i+1), 'tau:',tau, '  ','p-value:',p)
        contr_taus.append(tau)
        contr_ps.append(p<.05)
    print('control task:', 'average coefficient:', np.average(contr_taus),np.std(contr_taus),'sig.frac:', np.sum(contr_ps)/len(contr_ps))
    print('--------------')
    
def layerwise_change(data, measure='f1'):
    measures = {"precision":0, "recall":1, "f1":2}
    m = measures[measure]
    neg_changes = []
    control_changes = []
    xs= []
    story_pretrain, story_neg, story_control = data['story-pretrained'],data['story-neg'],data['story-control']
    layer = story_pretrain.shape[0]
    for i in range(10):
        neg_changes+= list(np.average(story_neg[i,:,:,m]-story_pretrain[:,:,m],axis=-1))
        control_changes += list(np.average(story_control[i,:,:,m]-story_pretrain[:,:,m],axis=-1))
        xs+=list(range(1,layer+1))
    sns.boxplot(xs,neg_changes)
    plt.axhline(0,color='red')
    plt.ylim(-0.3,0.3)
    plt.show()
    sns.boxplot(xs,control_changes)
    plt.axhline(0,color='red')
    plt.ylim(-0.3,0.3)
    plt.show()
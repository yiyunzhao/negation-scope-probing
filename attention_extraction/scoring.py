import sys
import os
import json
import pickle
import numpy as np
import collections
from collections import Counter
input_file = sys.argv[1]

def load_pickle(fname):
  with open(fname, "rb") as f:
    return pickle.load(f,encoding='latin1')

def attn_head_predictor(layer, head, mode="normal"):
  """Assign each word the most-attended-to other word as its head."""
  def predict(example):
    attn = np.array(example["attns"][layer][head])
    if mode == "transpose":
      attn = attn.T
    elif mode == "both":
      attn += attn.T
    else:
      assert mode == "normal"
    # ignore attention to self and [CLS]/[SEP] tokens
    attn[range(attn.shape[0]), range(attn.shape[0])] = 0
    attn = attn[1:-1, 1:-1]
    return np.argmax(attn, axis=-1) #+ 1  # +1 because ROOT is at index 0
  return predict


def evaluate_predictor(data,model):
    num=0 # zyy
    accum=[]
    errors = []
    for idx,example in enumerate(data):
        if len(example['sent'])!= len(model(example)):
            num+=1
            errors.append(idx)
        evaluate_one(example,model(example),accum)
    accum = np.array(accum)
    if errors:
    	print(errors)
    return accum

def evaluate_one(example,pred,accum):
    #bad_example=0
    sents = list(example['sent'].keys())
    if len(sents) == len(pred):
        for frame in example['negframe']:
            tp = 0
            total_p = 0
            neg_cue = frame.get('neg_cue')
            neg_scopes = frame.get('neg_scopes')
      
            for pos,p in enumerate(pred):
                this_word = sents[pos]
                if p >=0:
                    pred_head = sents[p]
                    if pred_head in frame['neg_cue']:
                        if this_word in frame['neg_scopes']:
                            tp+=1
                        total_p+=1
                else:
                    pass
            if len(frame['neg_scopes']) != 0:
            	accum.append([tp,total_p,len(frame['neg_scopes'])])
            else:
                accum.append([tp,total_p,0])
    else:
        print(sents)
        print('bad example!')
        pass
    return accum 

def get_measures(head_pred):
    tp,total_p,sys_p= np.sum(head_pred,axis=0)
    prec = tp/total_p
    recall = tp/sys_p
    f1=2*prec*recall/(prec+recall)
    return [prec,recall,f1 ]


def get_scores(data,mode="normal",lim=[12,12]):
  """Get the accuracies of every attention head."""
  scores = collections.defaultdict(dict)
  for layer in range(lim[0]): 
    for head in range(lim[1]):
      scores[layer][head] = evaluate_predictor(data,
          attn_head_predictor(layer, head, mode))
  return scores


def score(datafile):
    data_attn = load_pickle(datafile)
    data_attn = [i for i in data_attn if i['attns'] is not None]
    w,l = data_attn[0]['attns'].shape[:2]
    x1=get_scores(data_attn,lim=[w,l])
    res=[]
    for layer in x1:
        node_preds = np.array([get_measures(head) for head in x1[layer].values()])
        res.append(node_preds)
        prec,recall,f1 =np.max(node_preds,axis=0)
        precnode,recallnode,f1node = np.argmax(node_preds,axis=0)
        print('layer {} MAX VALS: prec is {} @ node {}, recall is {} @ node {}, f1 is {} @ node {}, '.format(layer,round(prec,4),precnode,round(recall,4),recallnode,round(f1,4),f1node))
        aveprec,averecall,avef1= np.round(np.average(node_preds,axis=0),4)
        print('layer {} AVE VALS: prec is {}, recall is {}, f1 is {}, '.format(layer, aveprec,averecall,avef1))
        print()
    res = np.array(res)
    res = np.nan_to_num(res)
    return res

def writeout(data,outname):
    with open(outname,'wb') as f:
        pickle.dump(data,f)

data = score(input_file)
writeout(data,'score_'+input_file)

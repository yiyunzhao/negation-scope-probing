# The script is adapted from Clark et al (2019)
import numpy as np
import sys
import argparse
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import BertTokenizer, TFBertModel, BertConfig
from transformers import XLNetTokenizer, TFXLNetModel
import copy
import pickle
import json

parser = argparse.ArgumentParser(description='Extracting the attention maps from transformers.')
parser.add_argument('--modelpath',help='add the model path or default models in transformers e.g.{bert-base-uncased, bert-large-uncased, roberta-base,...}')
parser.add_argument('--outpath', help='specify the output path')
parser.add_argument('--datapath',help='specify the data path')
parser.add_argument('--MODELTYPE', help='clarify the model type: bert or roberta')
parser.add_argument('--UNCASE',type= bool, help ='case or uncase? for bert-uncased please select specify it as True, for roberta specify it as False')
parser.add_argument('--MODE', help = 'token piece attention aggragation method: first/mean/max')

args = parser.parse_args()
print('the model path: {}, the output path:{}, the data path: {}, the MODELTYPE: specified {}, uncase: {} and attention mode: {}'.format(args.modelpath, args.outpath,args.datapath,args.MODELTYPE,args.UNCASE,args.MODE))

class AttnMapExtractor(object):
    def __init__(self, MODELPATH,MODEL=None):
        self.special_token_set = {'roberta':(['<s>','</s>'],'be'), 'bert':(['[CLS]','[SEP]'],'be'),'xlnet':(['<sep>','<cls>'],'e')}
        self.tokenizer = None
        self.model = None
        self.modeltype = None
        self.add_prefix_space =None
        if MODEL:
            MODEL = MODEL
        else:
            MODEL = MODELPATH.split('/')[-1]
        print(MODEL,MODELPATH)
        if MODEL.startswith('roberta'):
            self.modeltype = 'roberta'
            self.tokenizer = RobertaTokenizer.from_pretrained(MODELPATH,add_special_tokens=False)
            self.model = TFRobertaModel.from_pretrained(MODELPATH, output_attentions=True)
            self.add_prefix_space=True
        if MODEL.startswith('bert'):
            self.modeltype = 'bert'
            self.tokenizer = BertTokenizer.from_pretrained(MODELPATH,add_special_tokens=False)
            self.model = TFBertModel.from_pretrained(MODELPATH,output_attentions=True)
            self.add_prefix_space=False
        if MODEL.startswith('xlnet'):
            self.modeltype = 'xlnet'
            self.tokenizer = XLNetTokenizer.from_pretrained(MODELPATH,add_special_tokens=False)
            self.model = TFXLNetModel.from_pretrained(MODELPATH,output_attentions=True)
            self.add_prefix_space=False
    def get_attn_maps(self,example_id):
        example_id = tf.constant(example_id)[None,:]
        example_attns = self.model(example_id,training=False)[-1] # adding training=False
        return example_attns    
    def get_special_tokens(self):
        return self.special_token_set[self.modeltype]

def tokenize_and_align(word_list,tokenizer,special_tokens=(['<s>','</s>'],'be'), uncase=True,add_prefix_space=True):
    if uncase:
        word_list = [w.lower() for w in word_list]
    tks, pos = special_tokens
    if pos == 'be':
        word_list = [tks[0]] + word_list + [tks[1]]
    if pos == 'e':
        word_list = word_list + tks
    if pos == 'b':
        word_list = tks + word_list
    word_levels = []
    token_ids = []
    i=0
    for w in word_list:
        if add_prefix_space:
            token_id = tokenizer.encode(w,add_prefix_space=add_prefix_space)
        else:
            token_id = tokenizer.encode(w)
        token_ids += token_id
        l= len(token_id)
        word_levels.append(list(range(i,i+l)))
        i+= l
    assert len(word_list) == len(word_levels)
    return word_levels, token_ids

def get_word_word_attention(token_token_attention, words_to_tokens,mode="first"):
    "conver to word level for each attention head"
    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
      not_word_starts += word[1:]
    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
      word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)
    
    # several options for combining attention maps for words that have been split
    # we use "mean" in the paper
    for word in words_to_tokens:
      if mode == "first":
        pass
      elif mode == "mean":
        word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
      elif mode == "max":
        word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
        word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
      else:
        raise ValueError("Unknown aggregation mode", mode)
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)
    return word_word_attention

def make_attn_word_level(data, tokenizer, cased,mode):
    i=0
    for features in utils.logged_loop(data):
        words_to_tokens = tokenize_and_align(tokenizer, features["words"], cased)
    assert sum(len(word) for word in words_to_tokens) == len(features["tokens"])
    try:
        features["attns"] = np.stack([[
            get_word_word_attention(attn_head, words_to_tokens,mode)
            for attn_head in layer_attns] for layer_attns in features["attns"]])
    except:
        i+=1
        features['attns']=None
    print(i)

def writeout_pickle(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)

def load_pickle(data,path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def process(examples,MODELPATH,PATH,MODELTYPE,uncase,mode):
    extractor = AttnMapExtractor(MODELPATH,MODEL=MODELTYPE)
    tokenizer = extractor.tokenizer
    spc_tks = extractor.get_special_tokens()
    print(spc_tks,extractor.add_prefix_space, extractor.modeltype,uncase)
    feature_dicts_with_attn = []
    for enum, example in enumerate(examples):
        if (enum+1) % 100 == 0:
            print(enum+1)
        features = copy.deepcopy(example)
        words_to_tokens,exp_id = tokenize_and_align(example['words'],tokenizer,special_tokens=spc_tks,uncase=uncase,add_prefix_space=extractor.add_prefix_space)
        features['token_id'] = exp_id
        token_token_attention = extractor.get_attn_maps(exp_id)
        try:
            features['attns'] = np.stack([[get_word_word_attention(attn_head, words_to_tokens,mode=mode) 
            for attn_head in tf.squeeze(layer_attns,axis=0)] for layer_attns in token_token_attention])
        except:
            features['attns'] = None
        feature_dicts_with_attn.append(features)
    if PATH:
        writeout_pickle(feature_dicts_with_attn,PATH)
    return feature_dicts_with_attn
    
examples = load_json(args.datapath)
res=process(examples,args.modelpath,args.outpath,MODELTYPE=args.MODELTYPE,uncase=args.UNCASE,mode=args.MODE)
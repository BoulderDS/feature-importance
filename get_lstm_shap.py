import re
import os
import sys
import shap
import spacy
import utils
import torch
import pickle
import lstm as lc
import collections
import numpy as np
import random
from functools import partial
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import deep_id_pytorch
import warnings
warnings.filterwarnings('ignore')

### save_dir
REPO_DIR = os.path.dirname(os.path.abspath('data'))
DATA_ROOT = os.path.join(REPO_DIR, 'data')

SAVE_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
SAVE_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
SAVE_SST_DIR = os.path.join(DATA_ROOT, 'sst')

def split_tokens(l):
    return [i.split() for i in l]

def get_token_masks(data, model, length):
    ids, masks = [], []
    for d in data:
        id_values, mask_values = model.get_words_to_ids_masks(d.split(), length)
        ids.append(id_values)
        masks.append(mask_values)
    return torch.stack(ids), masks

def get_lstm_shap(model, train_data, test_data, background_length=100,
                  padding_length=512, gpu_memory_efficient=True,
                  feature_path=None, model_path=None):
    train_data = train_data.copy()
    np.random.seed(1001)
    np.random.shuffle(train_data)
    background = train_data[:background_length]
    bg_data, bg_masks = get_token_masks(background, model, padding_length)
    print("======preparing background data ===========")
    if gpu_memory_efficient:
        explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(model, bg_data, bg_masks,
                                                                 gpu_memory_efficient=gpu_memory_efficient)
    else:
        explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(model, bg_data.cuda(), bg_masks,
                                                                 gpu_memory_efficient=gpu_memory_efficient)
    model.train() # in case that shap complains that autograd cannot be called
    lstm_values = []
    features = []
    start = 0
    if os.path.exists(feature_path):
        start = len(utils.load_pickle(feature_path))
    for t in tqdm(test_data[start:]):
        td, tm = get_token_masks([t], model, padding_length)
        if not gpu_memory_efficient:
            td = td.cuda()
        lstm_shap_values = explainer.shap_values(td, tm)
        class_token_values = [[] for _ in lstm_shap_values]
        tokens = t.split()[:padding_length]
        for (i, token) in enumerate(tokens):
            if token in model.word_dict:
                w = model.word_dict[token]
            else:
                w = 0
            for c in range(len(lstm_shap_values)):
                class_token_values[c].append(lstm_shap_values[c][0, i, w])
        lstm_values.append(class_token_values)
        features.append(tokens)
        if len(features) % 50 == 0:
            if feature_path:
                utils.save_pickle(features, feature_path)
                utils.save_pickle(lstm_values, model_path)
    return features, lstm_values

    
def init_model(train_dev_tokens, d, path, use_gpu=True):
    tokens = split_tokens(train_dev_tokens)
    model = lc.LSTMAttentionClassifier(tokens, 
                                       emb_dim=d['emb_dim'],
                                       hidden_dim=d['hidden_dim'],
                                       num_layers=d['num_layers'],
                                       min_count=d['min_count'],
                                       bidirectional=True,
                                       use_gpu=use_gpu)
    if use_gpu:
        model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

def save_shap_val(hp_filename, filename, name, SAVE_DIR, train_data, test_data, test_labels, use_gpu=True,
                 background_length=100, padding_length=512):
    hp_d = 'models/{}.pkl'.format(hp_filename)
    hp_path = utils.get_abs_path(SAVE_DIR, hp_d)
    d = utils.load_pickle(hp_path)
    model_d = 'models/{}.pkl'.format(filename)
    model_path = utils.get_abs_path(SAVE_DIR, model_d)
    model = init_model(train_data, d, model_path, use_gpu=use_gpu)
    features_l, importance_l = [], []
    features = 'features/{}_shap_all_features.pkl'.format(name)
    feature_path = utils.get_abs_path(SAVE_DIR, features)
    scores = 'feature_importance/{}_shap_all_scores.pkl'.format(name)
    model_path = utils.get_abs_path(SAVE_DIR, scores)
    features_l, importance_l = get_lstm_shap(model, train_data, test_data,
                                             background_length=background_length, padding_length=padding_length,
                                             feature_path=feature_path, model_path=model_path)
    utils.save_pickle(features_l, feature_path)
    utils.save_pickle(importance_l, model_path)
    

if __name__ == "__main__":
    ### deception dataset
    if sys.argv[1] == "deception":
        print('=== deception ===')
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('deception')
        save_shap_val("lstm_att_hp", 'lstm_att', 'lstm_att', SAVE_DECEPTION_DIR,
                      train_dev_tokens, test_tokens, test_labels,
                      use_gpu=True, background_length=100, padding_length=877)
    
    elif sys.argv[1] == "yelp":
        ### yelp binary dataset
        print('=== yelp ===')
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('yelp')
        save_shap_val("lstm_att_hp", 'lstm_att', 'lstm_att', SAVE_YELP_DIR,
                      train_dev_tokens, test_tokens, test_labels,
                      use_gpu=True, background_length=100, padding_length=1104)
        # 1104 is max length in the dataset
    
    ### sst binary dataset
    elif sys.argv[1] == "sst":
        print('=== sst ===')
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('sst')
        save_shap_val("lstm_att_hp", 'lstm_att', 'lstm_att', SAVE_SST_DIR,
                      train_dev_tokens, test_tokens, test_labels,
                      use_gpu=True, background_length=100, padding_length=56)


import re
import os
import shap # use downloaded package instead of local package
import spacy
import utils
import torch
import pickle
import lstm as lc
import collections
import numpy as np
from functools import partial
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

### save_dir
REPO_DIR = os.path.dirname(os.path.abspath('data'))
DATA_ROOT = os.path.join(REPO_DIR, 'data')

SAVE_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
SAVE_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
SAVE_SST_DIR = os.path.join(DATA_ROOT, 'sst')

def get_svm_coef_d(pipeline):
    classifier = pipeline.named_steps['clf']
    feature = pipeline.named_steps['feature']
    coefficients = classifier.coef_[0]
    vocabulary = feature.vocabulary_
    d = {}
    for word, index in vocabulary.items():
        score = float(coefficients[index])
        d[str(word)] = score
    return d

def save_svm_coef(file, name, SAVE_DIR):
    model = 'models/{}.pkl'.format(file)
    path = utils.get_abs_path(SAVE_DIR, model)
    print('model path: {}'.format(path))
    pipeline = None
    if file == 'svm':
        pipeline = utils.load_pickle(path, encoding=False)
    else:
        pipeline = utils.load_pickle(path)
    svm_coef_d = get_svm_coef_d(pipeline)
    features = 'features/{}_coef_all_features.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(svm_coef_d, path)

def get_xgb_impt_d(pipeline):
    importance = pipeline.named_steps['clf'].feature_importances_
    vocab = pipeline.named_steps['feature'].vocabulary_
    d = {}
    for word, index in vocab.items():
        score = float(importance[index])
        d[str(word)] = score
    return d

def save_xgb_impt(file, name, SAVE_DIR):
    model = 'models/{}.pkl'.format(file)
    path = utils.get_abs_path(SAVE_DIR, model)
    print('model path: {}'.format(path))
    pipeline = utils.load_pickle(path)
    xgb_impt_d = get_xgb_impt_d(pipeline)
    features = 'features/{}_impt_all_features.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(xgb_impt_d, path)
    
def wrapper_clf_predict(test_tokens, model=None, model_name=None):
    labels = []
    if model_name == 'lstm_att':
        test_split_tokens = split_tokens(test_tokens)
        mapping = [model.get_words_to_ids(l) for l in test_split_tokens]
        labels = np.array(model.predict(test_split_tokens, mapping, return_probablity=True))
    elif model_name == "xgb":
        labels = model.predict_proba(test_tokens)
    else:
        labels = model.predict(test_tokens)
        labels = np.array([[0.999, 0.001] if l == -1 else [0.001, 0.999] for l in labels])
    return labels

def get_lime(model, test_tokens, model_name):
    explainer = LimeTextExplainer(class_names=["genuine", "deceptive"],
                                  split_expression=u'\s+')
    W = []
    for idx, text in enumerate(test_tokens):
        tmp_d = {}
        for i in text.split():
            tmp_d[i] = 1
        exp = explainer.explain_instance(text, 
                                         partial(wrapper_clf_predict, model=model, model_name=model_name), 
                                         num_features=len(text.split()), 
                                         num_samples=1000)
        if len(tmp_d) != len(exp.as_list()):
            print(idx, len(tmp_d), len(dict(exp.as_list())))
        W.append(dict(exp.as_list()))
        if (idx+1) % 10 == 0:
            print('{} instances have been processed..'.format(idx+1))
    features_l, scores_l = [], []
    for d in W:
        features, scores = [], []
        for key, score in d.items():
            features.append(key)
            tmp = ' '.join(features)
            scores.append(score) # abs value should be taken subsequently
        features_l.append(tmp)
        scores_l.append(scores)
    return features_l, scores_l

def save_lime_coef(filename, model_name, SAVE_DIR, train_dev_tokens, test_tokens, d_file=None):
    model = 'models/{}.pkl'.format(filename)
    path = utils.get_abs_path(SAVE_DIR, model)
    if 'svm' in model_name:
        model = utils.load_pickle(path, encoding=False)
    else:
        if model_name == 'lstm_att':
            hp_d = 'models/{}.pkl'.format(d_file)
            hp_path = utils.get_abs_path(SAVE_DIR, hp_d)
            d = utils.load_pickle(hp_path)
            model = init_model(train_dev_tokens, d, path)
        else:
            model = utils.load_pickle(path)
    features_l, importance_l = get_lime(model, test_tokens, model_name)
    features = 'features/{}_lime_all_features.pkl'.format(model_name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(features_l, path)
    scores = 'feature_importance/{}_lime_all_scores.pkl'.format(model_name)
    path = utils.get_abs_path(SAVE_DIR, scores)
    utils.save_pickle(importance_l, path)

def split_tokens(l):
    return [i.split() for i in l]

def init_model(train_dev_tokens, d, path):
    tokens = split_tokens(train_dev_tokens)
    model = lc.LSTMAttentionClassifier(tokens, 
                                       emb_dim=d['emb_dim'],
                                       hidden_dim=d['hidden_dim'],
                                       num_layers=d['num_layers'],
                                       min_count=d['min_count'],
                                       bidirectional=True)
    model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model

def get_shap(clf_name, pipeline, train_dev_tokens, test_tokens):
    feature = pipeline.named_steps['feature']
    clf = pipeline.named_steps['clf']
    vocab = feature.vocabulary_
    index_feature_d = {}
    for word, index in vocab.items():
        index_feature_d[index] = word
    X_train = feature.transform(train_dev_tokens)
    X_test = feature.transform(test_tokens).toarray()
    explainer = None
    if 'svm' in clf_name:
        explainer = shap.LinearExplainer(clf, X_train, feature_dependence="independent")
    else:
        explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # get all features
    features_l, importance_l = [], []
    for idx, row in enumerate(shap_values):
        word_shap_val_d = {}
        for idx_b, shap_val in enumerate(row):
            feature = index_feature_d[idx_b]
            word_shap_val_d[feature] = abs(shap_val) # taking absolute value
        features_tmp = list(word_shap_val_d.keys())
        features = " ".join(features_tmp)
        features_l.append(features)
        scores = list(word_shap_val_d.values())
        importance_l.append(scores)
    return features_l, importance_l

def save_shap_val(file, name, SAVE_DIR, train_data, test_data):
    model = 'models/{}.pkl'.format(file)
    path = utils.get_abs_path(SAVE_DIR, model)
    print('model path: {}'.format(path))
    model = None
    if file == 'svm':
        model = utils.load_pickle(path, encoding=False)
    else:
        model = utils.load_pickle(path)
    features_l, importance_l = [], []
    if 'svm' in name:
        features_l, importance_l = get_shap('svm', model, train_dev_tokens, test_tokens)
    elif 'xgb' in name:
        features_l, importance_l = get_shap('xgb', model, train_dev_tokens, test_tokens)
    features = 'features/{}_shap_all_features.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, features)
    utils.save_pickle(features_l, path)
    scores = 'feature_importance/{}_shap_all_scores.pkl'.format(name)
    path = utils.get_abs_path(SAVE_DIR, scores)
    utils.save_pickle(importance_l, path)
    
def save_data(SAVE_DIR, train_dev_tokens, test_tokens):
#     save_svm_coef('svm', 'svm', SAVE_DIR)
#     save_svm_coef('svm_l1', 'svm_l1', SAVE_DIR)
#     save_xgb_impt('xgb', 'xgb', SAVE_DIR)
    
#     save_lime_coef('svm', 'svm', SAVE_DIR, train_dev_tokens, test_tokens)
#     save_lime_coef('svm_l1', 'svm_l1', SAVE_DIR, train_dev_tokens, test_tokens) 
#     save_lime_coef('xgb', 'xgb', SAVE_DIR, train_dev_tokens, test_tokens)
#     save_lime_coef('lstm_att', 'lstm_att', SAVE_DIR, \
#                    train_dev_tokens, test_tokens, d_file='lstm_att_hp')
    
#     save_shap_val('svm', 'svm', SAVE_DIR, train_dev_tokens, test_tokens)
#     save_shap_val('svm_l1', 'svm_l1', SAVE_DIR, train_dev_tokens, test_tokens)
    save_shap_val('xgb', 'xgb', SAVE_DIR, train_dev_tokens, test_tokens)
    

if __name__ == "__main__":
    ### deception dataset
    print('=== deception ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('deception')
    save_data(SAVE_DECEPTION_DIR, train_dev_tokens, test_tokens)

    ### yelp binary dataset
    print('=== yelp binary ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('yelp')
    save_data(SAVE_YELP_DIR, train_dev_tokens, test_tokens)
    
    ### sst binary dataset
    print('=== sst binary ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('sst')
    save_data(SAVE_SST_DIR, train_dev_tokens, test_tokens)
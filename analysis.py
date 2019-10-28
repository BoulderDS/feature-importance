import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

import lstm as lc
import scikit_classification as sc

import os
import utils
import torch

from collections import defaultdict, OrderedDict, Counter

def convert_to_absolute(d):
    for k, v in d.items():
        d[k] = abs(v)
    return d

def create_copies(d, total):
    ret = []
    for i in range(total):
        ret.append(d)
    return ret

def get_built_in(save_dir, model_name, total):
    path = None
    if model_name == 'svm':
        path = utils.get_abs_path(save_dir, 'features/svm_coef_all_features.pkl')
    elif model_name == 'svm_l1':
        path = utils.get_abs_path(save_dir, 'features/svm_l1_coef_all_features.pkl')
    elif model_name == 'xgb':
        path = utils.get_abs_path(save_dir, 'features/xgb_impt_all_features.pkl')
    tmp = utils.load_pickle(path)
    tmp = convert_to_absolute(tmp) # take absoluate values
    ret = create_copies(tmp, total)
    return ret

def get_word_score_ds(save_dir, model_name, explainer, test_labels=None):
    path = None
    path = utils.get_abs_path(save_dir, 'features/{}_{}_all_features.pkl'.format(model_name, explainer))
    tmp_features = utils.load_pickle(path)
    if model_name == 'lstm_att' and explainer == 'shap':
        tmp_features = [' '.join(l) for l in tmp_features]
        
    path = utils.get_abs_path(save_dir, 'feature_importance/{}_{}_all_scores.pkl'.format(model_name, explainer))
    tmp_scores = utils.load_pickle(path)
    ret = create_word_score_ds(tmp_features, tmp_scores, model_name, explainer, labels=test_labels)
    return ret

def get_average_shap(d):
    ret = {}
    for token, values_l in d.items():
        ret[token] = np.mean(values_l)
    return ret

def create_word_score_ds(features_l, scores_l, model_name, explainer, labels=None):
    ds = []
    for review_idx, review in enumerate(features_l):    
        if labels != None: # if combination is lstm x shap
            d = defaultdict(lambda:[])
            actual_label = labels[review_idx]
            if actual_label == -1:
                tmp = scores_l[review_idx][0]
            else:
                tmp = scores_l[review_idx][1]
            for token_idx, token in enumerate(review.split()):
                d[token].append(abs(tmp[token_idx])) # append and take average
            d = get_average_shap(d) # take average
            ds.append(d)
        else:
            d = {}
            tmp = scores_l[review_idx]
            for token_idx, token in enumerate(review.split()):
                d[token] = abs(tmp[token_idx]) # take absolute numbers
            ds.append(d)
    return ds

def create_model_d(save_dir, model_name, test_labels=None): 
    lime = get_word_score_ds(save_dir, model_name, 'lime')
    shap = None
    if model_name == 'lstm_att':
        shap = get_word_score_ds(save_dir, model_name, 'shap', test_labels)
    else:
        shap = get_word_score_ds(save_dir, model_name, 'shap')
    built_in = None
    if model_name == 'lstm_att':
        built_in = get_word_score_ds(save_dir, model_name, 'weights')
    elif model_name == 'bert':
        built_in = get_word_score_ds(save_dir, model_name, 'impt')
    else:
        built_in = get_built_in(save_dir, model_name, len(lime))
    d = {
        'built_in': built_in,
        'lime': lime,
        'shap': shap 
    }
    d_keys = list(d.keys())
    return d, d_keys

def create_explainer_d(save_dir, explainer, total, test_labels=None, second=False, heatmap=False):
    svm, svm_l1, xgb, lstm_att = None, None, None, None
    if second == False:
        if heatmap == False:
            if explainer == 'built_in':
                svm = get_built_in(save_dir, 'svm', total)
                xgb = get_built_in(save_dir, 'xgb', total)
                lstm_att = get_word_score_ds(save_dir, 'lstm_att', 'weights')
            else:
                svm = get_word_score_ds(save_dir, 'svm', explainer) 
                xgb = get_word_score_ds(save_dir, 'xgb', explainer)
                if explainer == 'shap':
                    lstm_att = get_word_score_ds(save_dir, 'lstm_att', explainer, test_labels)
                else:
                    lstm_att = get_word_score_ds(save_dir, 'lstm_att', explainer)
            d = {
                'svm': svm,
                'xgb': xgb,
                'lstm_att': lstm_att,
            }
            d_keys = list(d.keys())
        else:
            # for heatmap and figure 1 row 1
            if explainer == 'built_in':
                svm = get_built_in(save_dir, 'svm', total)
                svm_l1 = get_built_in(save_dir, 'svm_l1', total)
                xgb = get_built_in(save_dir, 'xgb', total)
                lstm_att = get_word_score_ds(save_dir, 'lstm_att', 'weights')
                bert = get_word_score_ds(save_dir, 'bert', 'impt')
            else:
                svm = get_word_score_ds(save_dir, 'svm', explainer) 
                svm_l1 = get_word_score_ds(save_dir, 'svm_l1', explainer) 
                xgb = get_word_score_ds(save_dir, 'xgb', explainer)
                if explainer == 'shap':
                    lstm_att = get_word_score_ds(save_dir, 'lstm_att', explainer, test_labels)
                else:
                    lstm_att = get_word_score_ds(save_dir, 'lstm_att', explainer)
                bert = get_word_score_ds(save_dir, 'bert', explainer)                
            d = {
                'svm': svm,
                'svm_l1': svm_l1,
                'xgb': xgb,
                'lstm_att': lstm_att,
                'bert': bert
            }
            d_keys = list(d.keys())
    else:
        if explainer == 'built_in':
            svm_l1 = get_built_in(save_dir, 'svm_l1', total)
            xgb = get_built_in(save_dir, 'xgb', total)
            bert = get_word_score_ds(save_dir, 'bert', 'impt')
        else:
            svm_l1 = get_word_score_ds(save_dir, 'svm_l1', explainer) 
            xgb = get_word_score_ds(save_dir, 'xgb', explainer)
            bert = get_word_score_ds(save_dir, 'bert', explainer)
        d = {
            'svm_l1': svm_l1,
            'xgb': xgb,
            'bert': bert,
        }
        d_keys = list(d.keys())
    return d, d_keys

def get_model_combinations(combi=True, display=False, second=False):
    if second == False:
        ret = [
            ('built_in', 'lime'),
            ('lime', 'shap'),
            ('shap', 'built_in'),
        ]
        if display:
            ret = [
                ('built-in', 'LIME'),
                ('LIME', 'SHAP'),
                ('built-in', 'SHAP'),
            ]        
        if combi != True:
            ret = ['built-in', 'LIME', 'SHAP']
    return ret

def get_explainer_combinations(combi=True, display=False, heatmap=False, second=False):
    ret = None
    if second == False:
        ret = [
            ('svm', 'xgb'),
            ('svm', 'lstm_att'),
            ('xgb', 'lstm_att'),
        ]
        if display:
            ret = [
                ('SVM', 'XGB'),
                ('SVM', 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if heatmap:
            ret = [
                (r"SVM ($\ell_1$)", r"SVM ($\ell_2$)"),
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'LSTM')
                (r"SVM ($\ell_2$)", 'XGB'),
                (r"SVM ($\ell_2$)", 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if combi != True:
            ret = ['SVM', 'XGB', 'LSTM']
    else:
        ret = [
            ('svm_l1', 'xgb'),
            ('svm_l1', 'bert'),
            ('xgb', 'bert'),
        ]
        if display:
            ret = [
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'BERT'),
                ('XGB', 'BERT'),
            ]
        if heatmap:
            ret = [
                (r"SVM ($\ell_1$)", r"SVM ($\ell_2$)"),
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'LSTM')
                (r"SVM ($\ell_2$)", 'XGB'),
                (r"SVM ($\ell_2$)", 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if combi != True:
            ret = [r"SVM ($\ell_1$)", 'XGB', 'BERT']
    return ret   

def jacc_simi(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    words = list(list1 & list2)
    intersection = len(words)
    union = (len(list1) + len(list2)) - intersection
    return words, float(intersection / union)

def top_k(row, word_score_d, k):
    split_tokens = row.split()
    d = {}
    for word in split_tokens:
        if word in word_score_d:
            score = word_score_d[word]
            d[word] = score
    
    od = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
    top_k_features = list(od.keys())[-k:]
    top_k_scores = list(od.values())[-k:]
    return top_k_features, top_k_scores

def total_jacc(test_tokens, word_score_d1, word_score_d2, k, overlap=False):
    total_jacc, overlap_tokens = [], []
    for idx, row in enumerate(test_tokens):
        features_a, scores_a = top_k(row, word_score_d1[idx], k)
        features_b, scores_b = top_k(row, word_score_d2[idx], k)
        if idx == 0:
            #print(features_a, features_b)
            pass
        overlap_words, jacc_score = jacc_simi(features_a, features_b)
        total_jacc.append(jacc_score)
        overlap_tokens.append(overlap_words)
    if overlap:
        return total_jacc, overlap_tokens
    else:
        return total_jacc 
    
def get_min_max(list_of_lists):
    min_val, max_val = [], []
    for l in list_of_lists:
        min_val.append(np.min(l))
        max_val.append(np.max(l))
    return np.min(min_val), np.max(max_val)

def get_random_score(dataset_name):
    d = {
        'deception': [0.012614687500000001, 0.017019895833333333, 0.02319471875, 0.029685470238095236, 0.03633779389880952, 0.04323065557359309, 0.05014282876845375, 0.057315806478243976, 0.06454390648735454, 0.07189246692140848],
        'yelp_binary': [0.018970083333333335, 0.02572034722222222, 0.035245687500000004, 0.04551008809523809, 0.05124671203703702, 0.05716533145743147, 0.06328206441336441, 0.06963590307192807, 0.07620000904883514, 0.08301396587221686],
        'sst_binary': [0.07007078528281166, 0.09894574409665019, 0.11582246384770273, 0.1298662861849847, 0.1313647350574863]
    }
    return d[dataset_name]

def generate_simi_change_score(test_tokens, dicts, combi, if_model, k, y_err=False):
    d1 = combi[0]
    d2 = combi[1]
    total = total_jacc(test_tokens, dicts[d1], dicts[d2], k)
    avg_jacc = np.mean(total)
    y_err = ss.sem(total)
    if avg_jacc == 0:
        avg_jacc = 0.00000000000000000000000000000000000000000000000000001
    if y_err == 0:
        y_err = 0.00000000000000000000000000000000000000000000000000001
    if y_err:
        return avg_jacc, y_err
    else:
        return avg_jacc

def get_explainer_combinations(combi=True, display=False, heatmap=False, second=False):
    ret = None
    if second == False:
        ret = [
            ('svm', 'xgb'),
            ('svm', 'lstm_att'),
            ('xgb', 'lstm_att'),
        ]
        if display:
            ret = [
                ('SVM', 'XGB'),
                ('SVM', 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if heatmap:
            ret = [
                (r"SVM ($\ell_1$)", r"SVM ($\ell_2$)"),
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'LSTM')
                (r"SVM ($\ell_2$)", 'XGB'),
                (r"SVM ($\ell_2$)", 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if combi != True:
            ret = ['SVM', 'XGB', 'LSTM']
    else:
        ret = [
            ('svm_l1', 'xgb'),
            ('svm_l1', 'bert'),
            ('xgb', 'bert'),
        ]
        if display:
            ret = [
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'BERT'),
                ('XGB', 'BERT'),
            ]
        if heatmap:
            ret = [
                (r"SVM ($\ell_1$)", r"SVM ($\ell_2$)"),
                (r"SVM ($\ell_1$)", 'XGB'),
                (r"SVM ($\ell_1$)", 'LSTM')
                (r"SVM ($\ell_2$)", 'XGB'),
                (r"SVM ($\ell_2$)", 'LSTM'),
                ('XGB', 'LSTM'),
            ]
        if combi != True:
            ret = [r"SVM ($\ell_1$)", 'XGB', 'BERT']
    return ret    

### START OF HETEREOGENEITY ###
def get_k_combi_pred(train_dev_tokens, test_tokens, test_labels, dicts1, dicts2, key, \
                     model1, model2, k_list, save_dir, d_pred):
    different, same = [], []
    different_err, same_err = [], []
    for k in k_list:
        jacc_score = total_jacc(test_tokens, dicts1[key], dicts2[key], k)
        assert len(jacc_score) == len(test_tokens)
        model1_predictions = get_prediction(test_tokens, model1, save_dir, \
                                            train_dev_tokens, test_labels, 'lstm_att_hp')
        model2_predictions = get_prediction(test_tokens, model2, save_dir, \
                                            train_dev_tokens, test_labels, 'lstm_att_hp')

        tmp_different, tmp_same = [], []
        for idx, model1_pred in enumerate(model1_predictions):
            model2_pred = model2_predictions[idx]
            jacc_simi = jacc_score[idx]
            if model1_pred != model2_pred:
                tmp_different.append(jacc_simi)
            else:
                tmp_same.append(jacc_simi)
        different.append(np.mean(tmp_different))
        different_err.append(ss.sem(tmp_different))
        same.append(np.mean(tmp_same))
        same_err.append(ss.sem(tmp_same))
    return different, different_err, same, same_err

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

def get_prediction(test_tokens, model_name, save_dir, train_dev_tokens, test_labels, hp_name):
    hp_path, pipeline, model, predictions = None, None, None, None
    model_path = utils.get_abs_path(save_dir, 'models/{}.pkl'.format(model_name))
    if 'deception' in save_dir and model_name == 'svm':
        pipeline = utils.load_pickle(model_path, encoding=False)
    elif model_name == 'bert':
        pass
    else:
        pipeline = utils.load_pickle(model_path)
    if model_name == 'bert':
        if 'deception' in save_dir:
            dataset_name = 'deception'
        elif 'yelp' in save_dir:
            dataset_name = 'yelp'
        elif 'sst' in save_dir:
            dataset_name = 'sst'
        path = '/data/BERT_att_weights/{}-bert-preds.npy'.format(dataset_name)
        predictions = np.load(path)
    elif model_name == 'lstm_att':
        hp_path = utils.get_abs_path(save_dir, 'models/{}.pkl'.format(hp_name))
        d = utils.load_pickle(hp_path)
        model = init_model(train_dev_tokens, d, model_path)
        tokens = split_tokens(test_tokens)
        mapping = [model.get_words_to_ids(l) for l in tokens]
        predictions = model.predict(tokens, mapping)
    else:
        predictions, accuracy = sc.heldout_test(pipeline, test_tokens, test_labels)
    assert len(predictions) == len(test_tokens)
    return predictions   

def show_pred_plot(x_data, all_combi_data, x_label, y_label, file_name, \
                   fig_size, folder_name, y_err=None, x_min=None, x_max=None, \
                   y_min=None, y_max=None, combi_index=None, if_combi=True, \
                   second=False, save=False):
    fig, ax = plt.subplots(figsize=fig_size)
    colors = ['#073B4C', '#118AB2', '#06D6A0']
    line_styles= ['-', '--', ':']
    markers = None
    
    first_labels = None
    if second == False:
        first_labels = [('SVM', 'XGB'), ('SVM', 'LSTM'), ('XGB', 'LSTM')]
    else:
        first_labels = [(r"SVM ($\ell_1$)", 'XGB'), (r"SVM ($\ell_1$)", 'BERT'), ('XGB', 'BERT')]
    sec_labels = get_model_combinations(combi=False)            
    markers = ['^', 'o', 's']
        
    diff_data, same_data = all_combi_data[0], all_combi_data[1]
    diff_err, same_err = y_err[0], y_err[1]
    for idx_type, type_data in enumerate(diff_data):
        # same
        same_y = same_data[idx_type]
        label = '{} - agree'.format(sec_labels[idx_type])
        plt.errorbar(x_data, same_y, color=colors[idx_type], yerr=same_err[idx_type], \
                     fmt='-{}'.format(markers[idx_type]), linestyle=line_styles[0], \
                     label=label)
        # diff
        label = '{} - disagree'.format(sec_labels[idx_type])
        plt.errorbar(x_data, type_data, color=colors[idx_type], yerr=diff_err[idx_type], \
                     fmt='-{}'.format(markers[idx_type]), linestyle=line_styles[1], \
                     label=label)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    if x_min != None and x_max != None:
        plt.xlim(x_min, x_max)
    if y_min != None and y_max != None:
        plt.ylim(y_min, y_max)
    plt.xticks(np.arange(min(x_data), max(x_data)+1, 1))
    if save:
        path = get_save_path(folder_name, file_name)
    plt.show()
    plt.close() 
    
def get_tokens_length(test_tokens):
    ret = []
    for row in test_tokens:
        length = len(row.split()) # len counted by num of tokens
        ret.append(length)
    return ret

def get_tokens_ratio(test_tokens):
    ret = []
    for row in test_tokens:
        words = row.split()
        ratio = len(set(words)) / len(words)
        ret.append(ratio)
    return ret

def get_rho(test_tokens, dicts, combi, k_list, variable_l):
    data = []
    for c in combi:
        k_rho = []
        for k in k_list:
            jacc_simi = total_jacc(test_tokens, dicts[c[0]], dicts[c[1]], k)
            rho, _ = ss.spearmanr(jacc_simi, variable_l)
            if np.isnan(rho):
                rho = 0
            k_rho.append(rho)
        data.append(k_rho)
    return data
### END OF HETEREOGENEITY ###

### START OF ENTROPY ###
def get_tokens_top_k(tokens, word_score_d, k):
    top_k_l = []
    for idx, row in enumerate(tokens):
        top_k_features, top_k_scores = top_k(row, word_score_d[idx], k)
        top_k_l.append(top_k_features)
    return top_k_l

def get_entropy(test_tokens, dicts, d_keys, k_list):
    data = []
    for key in d_keys:
        tmp = []
        for k in k_list:
            top_k_tokens = [' '.join(l) for l in get_tokens_top_k(test_tokens, dicts[key], k)]
            assert len(top_k_tokens) == len(test_tokens)
            list_words = ' '.join(top_k_tokens)
            counter = Counter(list_words.split())
            total = sum(counter.values())
            if k == 1:
                #print('counter: {}'.format(counter))
                pass
            proba = [counter[k] / total for k in counter]
            entropy = ss.entropy(proba, base=2)
            tmp.append(entropy)
        data.append(tmp)
    return data
### END OF ENTROPY ###

### START OF POS ###
def get_pos_val(top_k_l, token_pos_d, num_tokens=None):
    pos_count_d = defaultdict(lambda:0)
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET']
    for row in top_k_l:
        for token in row:
            pos_tag = token_pos_d[token]
            if pos_tag in pos_tags:
                pos_count_d[pos_tag] += 1
    ret = []
    for tag in pos_tags:
        if num_tokens != None:
            ret.append(pos_count_d[tag] / num_tokens * 100)
        else:
            ret.append(pos_count_d[tag])   
    return ret

def get_vocab_size(test_tokens):
    d = defaultdict(lambda:0)
    for row in test_tokens:
        for token in row.split():
            d[token] += 1
    values = list(d.values())
    total = 0
    for v in values:
        total += v
    return total

def get_combi_pos(d_keys, dicts, test_tokens, k, token_pos_d, vocab_size):
    data = []
    # add backrgound
    tokens = [row.split() for row in test_tokens]
    bg_vocab_size = get_vocab_size(test_tokens)
    background_pos = get_pos_val(tokens, token_pos_d, bg_vocab_size)
    data.append(background_pos)
    for key in d_keys:
        top_k_l = get_tokens_top_k(test_tokens, dicts[key], k)
        y_data = get_pos_val(top_k_l, token_pos_d, vocab_size)   
        data.append(y_data)
    return data

def get_token_pos_d(data, pos):
    d = {}
    for row_idx, row in enumerate(data):
        for token_idx, token in enumerate(row.split()):
            d[token] = pos[row_idx].split()[token_idx]
    return d

def format_pos_data(tmp_y_data, pos_types):
    pos_data_d = defaultdict(lambda: [])
    for data in tmp_y_data:
        for idx, val in enumerate(data):
            pos_tag = pos_types[idx]
            pos_data_d[pos_tag].append(val)
    return list(pos_data_d.values()) 
### END OF POS ###
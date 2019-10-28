import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

import lstm as lc
import analysis as als
import similarity as simi
import scikit_classification as sc

import os
import utils
import torch

from scipy.spatial import distance
from collections import defaultdict, OrderedDict, Counter

'''
START OF ENTROPY FUNCTIONS
'''
def run_entropy(dataset_name, save_dir, k_list, models, feature_types, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)

    y_data, min_vals, max_vals, y_min_val, y_max_val = [], [], [], 0, 0
    for model_name in models:
        dicts, d_keys = als.create_model_d(save_dir, model_name, test_labels=test_labels)
        tmp_y_data = als.get_entropy(test_tokens, dicts, d_keys, k_list)
        assert len(tmp_y_data) == len(d_keys)
        tmp_min_val, tmp_max_val = als.get_min_max(tmp_y_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(tmp_y_data) 
    y_min_val = np.min(min_vals) - 0.25
    y_min_val = max(0, y_min_val)
    y_max_val = np.max(max_vals) + 0.25
    simi.show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Entropy', '', \
                        (13, 12), '', x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                        y_min=y_min_val, y_max=y_max_val, if_model=True, second=second, \
                        if_combi=False, if_builtin_posthoc=True) 
'''
END OF ENTROPY FUNCTIONS
'''


'''
START POS WITH BUILT-IN FUNCTIONS
'''
def show_bar_plot(x_data, all_combi_data, x_label, y_label, file_name, \
                  fig_size, folder_name, y_err=None, x_min=None, x_max=None, \
                  y_min=None, y_max=None, labels=None, save=False):
    fig, ax = plt.subplots(figsize=fig_size)
    index = np.arange(len(x_data))
    bar_width = 0.3
    colors = ['#344b5b', '#356384', '#367bac', '#4892c6', '#69a6d0', '#8abbdb']
       
    for idx, y_data in enumerate(all_combi_data):
        bar = plt.bar(index*(bar_width*len(labels)+0.2)+(bar_width*idx), y_data, width=bar_width, \
                      tick_label=x_data, color=colors[idx], label=labels[idx])
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    ax.set_xticks(index*(bar_width*len(labels)+0.2)+bar_width*idx - bar_width*(len(labels)/2))
    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    if x_min != None and x_max != None:
        plt.xlim(x_min, x_max)
    if y_min != None and y_max != None:
        plt.ylim(y_min, y_max)
    if save:
        path = get_save_path(folder_name, file_name)
    plt.show()
    plt.close()

def run_pos_percent(dataset_name, data_dir, save_dir, models, feature_types, k):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    train_pos, dev_pos, train_dev_pos, test_pos = utils.get_pos(dataset_name, data_dir)
    
    pos_types = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET']
    token_pos_d = als.get_token_pos_d(test_tokens, test_pos) 
    vocab_size = len(test_tokens) * k
    min_vals, max_vals, y_min_val, y_max_val = [], [], 0, 0
    for idx, feature_type in enumerate(feature_types):
        dicts, d_keys = als.create_explainer_d(save_dir, feature_type, len(test_labels), test_labels=test_labels)
        tmp_y_data = als.get_combi_pos(d_keys, dicts, test_tokens, k, token_pos_d, vocab_size)
        tmp_min_val, tmp_max_val = als.get_min_max(tmp_y_data)
    
        y_min_val = max(tmp_min_val-1, 0)
        y_max_val = min(tmp_max_val+1, 100)
        
        y_data = als.format_pos_data(tmp_y_data, pos_types)
        assert len(y_data) == len(pos_types)
        display_model_names = als.get_explainer_combinations(combi=False)
        display_feature_names = als.get_model_combinations(combi=False)
        x_data = []
        x_data.append('Background')
        for model_name in display_model_names:
            label = '{}'.format(model_name)
            x_data.append(label)
        
        show_bar_plot(x_data, y_data, '', 'Percentage', \
                      '', (15, 14), '', y_min=y_min_val, \
                      y_max=y_max_val, labels=pos_types)
'''
END POS WITH BUILT-IN FUNCTIONS
'''


'''
START OF DISTANCE BETWEEN POS & IMPORTANT WORDS FUNCTIONS
'''
def get_jensen_shannon(test_tokens, dicts, d_keys, k_list, var, combinations=None, token_pos_d=None):
    data = []
    if var == 'word_dist':
        background_keys, background_total, background_counter = als.get_keys_total(test_tokens)
        k_combi_keys, k_combi_total, k_combi_counter = als.get_combi_keys_total(test_tokens, dicts, d_keys, k_list)
        
        for idx_a, combi_keys in enumerate(k_combi_keys):
            tmp = []
            for idx_b, combi_k in enumerate(combi_keys):
                all_keys = set(combi_k) | set(background_keys)
                first_proba = [background_counter.get(k, 0) / background_total for k in all_keys]
                cur_counter, cur_total = k_combi_counter[idx_a][idx_b], k_combi_total[idx_a][idx_b]
                second_proba = [cur_counter.get(k, 0) / cur_total for k in all_keys]
                jensen_shannon = distance.jensenshannon(first_proba, second_proba, base=2.0)
                tmp.append(jensen_shannon)
            data.append(tmp)
    elif var == 'pos':
        for combi in combinations:
            combi1, combi2 = combi[0], combi[1]
            tmp = []
            for k in k_list:
                combi1_top_k_l = als.get_tokens_top_k(test_tokens, dicts[combi1], k)
                combi2_top_k_l = als.get_tokens_top_k(test_tokens, dicts[combi2], k)
                assert len(combi1_top_k_l) == len(combi2_top_k_l)
                combi1_pos_val = als.get_pos_val(combi1_top_k_l, token_pos_d)
                combi2_pos_val = als.get_pos_val(combi2_top_k_l, token_pos_d)
                assert len(combi1_pos_val) == len(combi2_pos_val)
                jensen_shannon = distance.jensenshannon(combi1_pos_val, combi2_pos_val, base=2.0)
                tmp.append(jensen_shannon)
            data.append(tmp)
    elif var == 'background':
        tokens = [row.split() for row in test_tokens]
        vocab_size = als.get_vocab_size(test_tokens)
        background_pos_val = als.get_pos_val(tokens, token_pos_d, vocab_size)
        for key in combinations:
            tmp = []
            for k in k_list:
                combi1_top_k_l = als.get_tokens_top_k(test_tokens, dicts[key], k)
                combi1_pos_val = als.get_pos_val(combi1_top_k_l, token_pos_d)
                assert len(combi1_pos_val) == len(background_pos_val)
                jensen_shannon = distance.jensenshannon(combi1_pos_val, background_pos_val, base=2.0)
                tmp.append(jensen_shannon)
            data.append(tmp)        
    return data

def run_js_pos(dataset_name, data_dir, save_dir, models, feature_types, k_list, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    train_pos, dev_pos, train_dev_pos, test_pos = utils.get_pos(dataset_name, data_dir)
    token_pos_d = als.get_token_pos_d(test_tokens, test_pos) 
    # compare with background
    y_data, min_vals, max_vals, y_min_val, y_max_val = [], [], [], 0, 0
    for model_name in models:
        dicts, d_keys = als.create_model_d(save_dir, model_name, test_labels=test_labels)
        tmp_y_data = get_jensen_shannon(test_tokens, dicts, d_keys, k_list, 'background', \
                                        combinations=d_keys, token_pos_d=token_pos_d)
        tmp_min_val, tmp_max_val = als.get_min_max(tmp_y_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(tmp_y_data)
    y_min_val = np.min(min_vals) - 0.05
    y_max_val = np.max(max_vals) + 0.05
    simi.show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Jensen-Shannon Score', '', \
                        (13, 12), '', y_min=y_min_val, y_max=y_max_val, if_model=True, second=second, \
                        if_combi=False, if_background=True, if_builtin_posthoc=True) 
'''
START OF DISTANCE BETWEEN POS & IMPORTANT WORDS FUNCTIONS
'''
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

from collections import defaultdict, OrderedDict, Counter

'''
START OF SIMILARITY BETWEEN TWO MODELS FUNCTIONS
'''
def run_pred_combi(dataset_name, save_dir, train_dev_tokens, test_tokens, \
                   test_labels, d_pred, combinations, k_list, second):
    for idx, combi in enumerate(combinations):
        model1, model2 = combi[0], combi[1]
        print('{}: {} vs. {}'.format(dataset_name, model1, model2))
        dicts1, dict_keys1 = als.create_model_d(save_dir, model1, test_labels)
        dicts2, dict_keys2 = als.create_model_d(save_dir, model2, test_labels)
        keys_different, keys_same, keys_different_err, keys_same_err = [], [], [], []
        for key in dict_keys1:
            different, different_err, same, same_err = als.get_k_combi_pred(train_dev_tokens, test_tokens, \
                                                                        test_labels, dicts1, dicts2, key, \
                                                                        model1, model2, k_list, save_dir, \
                                                                        d_pred)
            keys_different.append(different)
            keys_same.append(same)
            keys_different_err.append(different_err)
            keys_same_err.append(same_err)
        y_data, y_err = [], []
        y_data.append(keys_different)
        y_data.append(keys_same)
        y_err.append(keys_different_err)
        y_err.append(keys_same_err)
        
        diff_min_val, diff_max_val = als.get_min_max(keys_different)
        same_min_val, same_max_val = als.get_min_max(keys_same)
        y_min = max(0, min(diff_min_val, same_min_val)-0.05)
        y_max = min(1, max(diff_max_val, same_max_val)+0.05)
        
        file_name = '{}_{}_{}'.format(dataset_name, model1, model2)
        als.show_pred_plot(k_list, y_data, 'Number of important features (k)', 'Jaccard Similarity', 
                       file_name, (12, 12), '', y_err=y_err, \
                       y_min=y_min, y_max=y_max, combi_index=idx, \
                       second=second)
        
def run_simi_pred(dataset_name, save_dir, models, k_list, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    
    d_pred = {}
    for model_name in models:
        pred = als.get_prediction(test_tokens, model_name, save_dir, train_dev_tokens, test_labels, 'lstm_att_hp')
        assert len(pred) == len(test_tokens)
        d_pred[model_name] = pred
    
    combinations = als.get_explainer_combinations(second=second)
    combinations = [('svm', 'xgb'), ('xgb', 'lstm_att'), ('svm', 'lstm_att')]
    if second:
        combinations = [('svm_l1', 'xgb'), ('xgb', 'bert'), ('svm_l1', 'bert')]
    run_pred_combi(dataset_name, save_dir, train_dev_tokens, test_tokens, test_labels, \
                   d_pred, combinations, k_list, second)
'''
END OF SIMILARITY BETWEEN TWO MODELS FUNCTIONS
'''


'''
START OF SIMILARITY COMPARISON VS. LENGTH FUNCTIONS
'''
def run_simi_length(dataset_name, save_dir, k_list, models, feature_types, folder_name, var, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    if var == 'len':
        variable_l = als.get_tokens_length(test_tokens)
    else:
        variable_l = als.get_tokens_ratio(test_tokens)
    # generate models line plot
    y_data, min_vals, max_vals, y_min_val, y_max_val = [], [], [], 0, 0
    for model_name in models:
        combinations = als.get_model_combinations()
        dicts, dict_keys = als.create_model_d(save_dir, model_name, test_labels)
        all_combi_data = als.get_rho(test_tokens, dicts, combinations, k_list, variable_l)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
    y_min_val = np.min(min_vals) - 0.05
    y_min_val = min(y_min_val, 0-0.05)
    y_max_val = np.max(max_vals) + 0.05
    y_max_val = max(y_max_val, 0+0.05)
    simi.show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Spearman correlation', '', \
                        (13, 12), '', x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                        y_min=y_min_val, y_max=y_max_val, if_model=True, second=second, \
                        if_builtin_posthoc=True) 
'''
END OF SIMILARITY COMPARISON VS. LENGTH FUNCTIONS
'''


'''
START OF COMPARISON BETWEEN MODELS SIMILARITY VS. TYPE-TOKEN RATIO FUNCTIONS
'''
def run_comp_models_type_token(dataset_name, save_dir, k_list, models, feature_types, folder_name, var, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    if var == 'len':
        variable_l = als.get_tokens_length(test_tokens)
    else:
        variable_l = als.get_tokens_ratio(test_tokens)
    # generate explainers line plot
    y_data, min_vals, max_vals, y_min_val, y_max_val = [], [], [], 0, 0
    for feature_type in feature_types:
        combinations = als.get_explainer_combinations(second=second)
        dicts, dict_keys = als.create_explainer_d(save_dir, feature_type, len(test_tokens), test_labels, second=second)
        all_combi_data = als.get_rho(test_tokens, dicts, combinations, k_list, variable_l)
        min_val, max_val = als.get_min_max(all_combi_data)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
    y_min_val = np.min(min_vals) - 0.05
    y_min_val = min(y_min_val, 0-0.05)
    y_max_val = np.max(max_vals) + 0.05
    y_max_val = max(y_max_val, 0+0.05)
    simi.show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Spearman correlation', '', \
                        (13, 12), '', x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                        y_min=y_min_val, y_max=y_max_val, if_model=False, second=second, \
                        if_builtin_posthoc=True) 
'''
END OF COMPARISON BETWEEN MODELS SIMILARITY VS. TYPE-TOKEN RATIO FUNCTIONS
'''


'''
START OF COMPARISON BETWEEN METHODS SIMILARITY VS. TYPE-TOKEN RATIO FUNCTIONS
'''
def run_comp_methods_type_token(dataset_name, save_dir, k_list, models, feature_types, folder_name, var, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    if var == 'len':
        variable_l = als.get_tokens_length(test_tokens)
    else:
        variable_l = als.get_tokens_ratio(test_tokens)
    # generate models line plot
    y_data, min_vals, max_vals, y_min_val, y_max_val = [], [], [], 0, 0
    for model_name in models:
        combinations = als.get_model_combinations()
        dicts, dict_keys = als.create_model_d(save_dir, model_name, test_labels)
        all_combi_data = als.get_rho(test_tokens, dicts, combinations, k_list, variable_l)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
    y_min_val = np.min(min_vals) - 0.05
    y_min_val = min(y_min_val, 0-0.05)
    y_max_val = np.max(max_vals) + 0.05
    y_max_val = max(y_max_val, 0+0.05)
    simi.show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Spearman correlation', '', \
                        (13, 12), '', x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                        y_min=y_min_val, y_max=y_max_val, if_model=True, second=second, \
                        if_builtin_posthoc=True) 
'''
END OF COMPARISON BETWEEN METHODS SIMILARITY VS. TYPE-TOKEN RATIO FUNCTIONS
'''
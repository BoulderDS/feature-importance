import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

import lstm as lc
import analysis as als
import scikit_classification as sc

import os
import utils
import torch

from collections import defaultdict, OrderedDict, Counter

'''
START OF JACC SIMILARITY FUNCTIONS
'''
def show_heatmap(d_keys, data, fig_size, vmin, vmax):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=fig_size)
    sns_plot = sns.heatmap(df, annot=True, cmap="PuBu", vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(d_keys))+0.5, (r"SVM ($\ell_2$)", r"SVM ($\ell_1$)", "XGB", "LSTM", "BERT"))
    plt.yticks(np.arange(len(d_keys))+0.5, (r"SVM ($\ell_2$)", r"SVM ($\ell_1$)", "XGB", "LSTM", "BERT"), \
                                           va="center")
    fig = sns_plot.get_figure()
    plt.show()
    return fig

def generate_heatmap_data(test_tokens, word_score_ds, d_keys, k):
    data = []
    for i in d_keys:
        tmp = []
        for j in d_keys:
            total = als.total_jacc(test_tokens, word_score_ds[i], word_score_ds[j], k)
            avg_jacc = np.mean(total)
            tmp.append(avg_jacc)
        data.append(tmp)
    return data

def generate_heatmap(test_tokens, d, d_keys, folder_name, file_name, k):
    data = generate_heatmap_data(test_tokens, d, d_keys, k)
    d_keys = [r"SVM ($\ell_2$)", r"SVM ($\ell_1$)", "XGB", "LSTM", "BERT"]
    fig = show_heatmap(d_keys, data, (15, 10), 0, 1.0)

def generate_k_heatmaps(save_dir, model_name, test_tokens, test_labels, \
                        dataset_name, k_list, folder_name, explainer_name=None):
    dicts, dict_keys, type_name = None, None, None
    if explainer_name == None:
        dicts, dict_keys = als.create_model_d(save_dir, model_name, test_labels)
        type_name = model_name
    else:
        dicts, dict_keys = als.create_explainer_d(save_dir, explainer_name, len(test_tokens), test_labels, heatmap=True)
        type_name = explainer_name
    for k in k_list:
        generate_heatmap(test_tokens, dicts, dict_keys, folder_name, '', k)

def run_heatmap(dataset_name, save_dir, models, feature_types, k_list, folder_name=''):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    for feature_type in feature_types:
        generate_k_heatmaps(save_dir, '', test_tokens, test_labels, \
                            dataset_name, k_list, folder_name, explainer_name=feature_type)
'''
END OF JACC SIMILARITY FUNCTIONS
'''


'''
START OF SIMILARITY CHANGE UTILITY FUNCTIONS
'''
def generate_simi_change(save_dir, model_name, test_tokens, test_labels, \
                         dataset_name, k_list, explainer_name=None, y_err=False,
                         second=False, if_comp_model=False):
    dicts, dict_keys, type_name = None, None, None
    if_model = True
    if explainer_name == None:
        dicts, dict_keys = als.create_model_d(save_dir, model_name, test_labels)
        type_name = model_name
    else:
        dicts, dict_keys = als.create_explainer_d(save_dir, explainer_name, len(test_tokens), test_labels, second=second)
        type_name = explainer_name
        if_model = False
    combinations = None
    if if_model:
        combinations = als.get_model_combinations()
    else:
        if if_comp_model:
            combinations = [('svm', 'lstm_att')]
            if second:
                combinations = [('svm_l1', 'bert')]
        else:
            combinations = als.get_explainer_combinations(second=second)
    all_combi_data, all_y_err = [], []
    for combi in combinations:
        d1 = combi[0]
        d2 = combi[1]
        combi_data, y_err_data = [], []
        for k in k_list:
            avg_jacc, yerr = als.generate_simi_change_score(test_tokens, dicts, combi, if_model, k, y_err=True)
            combi_data.append(avg_jacc)
            y_err_data.append(yerr)
        all_combi_data.append(combi_data)
        all_y_err.append(y_err_data)
    if y_err:
        return all_combi_data, all_y_err
    else:
        return all_combi_data
    
def show_simi_plot(x_data, all_combi_data, x_label, y_label, file_name, \
                   fig_size, folder_name, y_err=None, x_min=None, x_max=None, \
                   y_min=None, y_max=None, if_model=True, if_combi=True, \
                   dataset_name=None, if_background=False, second=False, \
                   model_combi=False, save=False, if_comp_models=False, \
                   if_builtin_posthoc=False, if_comp_methods=False):
    fig, ax = plt.subplots(figsize=fig_size)
    line_styles= ['-', '--', ':']
    colors, markers = None, None
    
    first_labels, sec_labels = None, None
    if if_model: # model combination
        if if_combi:
            first_labels = als.get_explainer_combinations(combi=False, display=True, second=second)
            sec_labels = als.get_model_combinations(display=True)
        else:
            first_labels = als.get_explainer_combinations(combi=False, second=second)
            sec_labels = als.get_model_combinations(combi=False)            
        markers = ['^', 'o', 's']
        colors = ['#613F75', '#7D82B8', '#EF798A']
    else: # just models
        if if_combi:
            if if_comp_models:
                first_labels = ['built-in']
            else:
                first_labels = als.get_model_combinations(combi=False, display=True)
            sec_labels = als.get_explainer_combinations(display=True, second=second)
        else:
            first_labels = als.get_model_combinations(combi=False)
            sec_labels = als.get_explainer_combinations(combi=False, second=second)             
        markers = ['v', 'X', 'D']
        colors = ['#073B4C', '#118AB2', '#06D6A0']
    
    for idx_a, y_data in enumerate(all_combi_data):
        for idx, i in enumerate(y_data):
            if if_combi:
                label1, label2 = sec_labels[idx][0], sec_labels[idx][1]
                label = '{} - {} x {}'.format(first_labels[idx_a], label1, label2)
                if model_combi:  
                    if if_builtin_posthoc:
                        label = '{} x {} - {}'.format('SVM', 'LSTM', first_labels[idx_a])
                        if second:
                            label = '{} x {} - {}'.format(r"SVM ($\ell_1$)", "BERT", first_labels[idx_a])
                    else:
                        label = '{} x {} - {}'.format(label1, label2, first_labels[idx_a])
            else:
                label = '{} - {}'.format(first_labels[idx_a], sec_labels[idx])
                if if_background:
                    label = '{} - {} x background'.format(first_labels[idx_a], sec_labels[idx])
            if y_err != None:
                if if_comp_models:
                    plt.errorbar(x_data, i, color=colors[idx], yerr=y_err[idx_a][idx], \
                                 fmt='-{}'.format(markers[idx]), linestyle=line_styles[idx], label=label)
                if if_builtin_posthoc:
                    plt.errorbar(x_data, i, color=colors[idx_a], yerr=y_err[idx_a][idx], \
                                 fmt='-{}'.format(markers[idx_a]), linestyle=line_styles[idx_a], label=label)
                if if_comp_methods:
                    plt.errorbar(x_data, i, color=colors[idx_a], yerr=y_err[idx_a][idx], \
                                 fmt='-{}'.format(markers[idx]), linestyle=line_styles[idx], label=label)
            else:
                plt.plot(x_data, i, color=colors[idx_a], marker=markers[idx], \
                         linestyle=line_styles[idx], label=label)
    
    random_score = None
    if dataset_name != None:
        random_score = als.get_random_score(dataset_name)
        plt.plot(x_data, random_score, color='#95a5a6', label='Random')
    plt.axhline(y=0, color='#95a5a6', linestyle='-')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    if x_min != None and x_max != None:
        plt.xlim(x_min, x_max)
    if y_min != None and y_max != None:
        if dataset_name != None:
            plt.ylim(min(y_min, np.min(random_score)), y_max)
        else:
            plt.ylim(y_min, y_max)
    plt.xticks(np.arange(min(x_data), max(x_data)+1, 1))
    plt.show()
    plt.close()  
'''
END OF SIMILARITY CHANGE UTILITY FUNCTIONS
'''


'''
START OF COMPARISON BETWEEN MODELS FUNCTIONS
'''
def run_comp_btw_models(dataset_name, save_dir, k_list, models, feature_types, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    # generate explainer - model x model
    y_data, y_err_data = [], []
    min_vals, max_vals, y_min_val, y_max_val = [], [], 0, 0
    for feature_type in feature_types:
        all_combi_data, y_err = generate_simi_change(save_dir, '', test_tokens, test_labels, \
                                                     dataset_name, k_list, explainer_name=feature_type, \
                                                     y_err=True, second=second)
        min_val, max_val = als.get_min_max(all_combi_data)
        assert len(all_combi_data) == len(y_err)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
        y_err_data.append(y_err)
        
        y_min_val = np.min(min_vals) - 0.05
        y_min_val = max(0, y_min_val)
        y_max_val = np.max(max_vals) + 0.05
        show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Jaccard Similarity', '', \
                       (13, 12), '', y_err=y_err_data, x_min=np.min(k_list)-0.5, \
                       x_max=np.max(k_list)+0.5, y_min=y_min_val, y_max=y_max_val, if_model=False, \
                       dataset_name=dataset_name, second=second, if_comp_models=True)

'''
END OF COMPARISON BETWEEN MODELS FUNCTIONS
'''


'''
START OF COMPARISON BETWEEN BUILT-IN & POST-HOC FUNCTIONS
'''
def run_comp_builtin_posthoc(dataset_name, save_dir, k_list, models, feature_types, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    y_data, y_err_data = [], []
    min_vals, max_vals, y_min_val, y_max_val = [], [], 0, 0
    for feature_type in feature_types:
        all_combi_data, y_err = generate_simi_change(save_dir, '', test_tokens, test_labels, dataset_name, \
                                                     k_list, explainer_name=feature_type, y_err=True, \
                                                     second=second, if_comp_model=True)
        min_val, max_val = als.get_min_max(all_combi_data)
        assert len(all_combi_data) == len(y_err)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
        y_err_data.append(y_err)
    
    y_min_val = np.min(min_vals) - 0.05
    y_min_val = max(0, y_min_val)
    y_max_val = np.max(max_vals) + 0.05
    show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Jaccard Similarity', '', \
                   (13, 12), '', y_err=y_err_data, x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                   y_min=y_min_val, y_max=y_max_val, if_model=False, dataset_name=dataset_name, \
                   model_combi=True, second=second, if_builtin_posthoc=True) 
'''
END OF COMPARISON BETWEEN BUILT-IN & POST-HOC FUNCTIONS
'''


'''
START OF COMPARISON BETWEEN METHODS FUNCTIONS
'''
def run_comp_btw_methods(dataset_name, save_dir, k_list, models, feature_types, second=False):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    # generate models line plot
    y_data, y_err_data = [], []
    min_vals, max_vals, y_min_val, y_max_val = [], [], 0, 0
    for model_name in models:
        all_combi_data, y_err = generate_simi_change(save_dir, model_name, test_tokens, test_labels, \
                                                     dataset_name, k_list, y_err=True, second=second)
        assert len(all_combi_data) == len(y_err)
        tmp_min_val, tmp_max_val = als.get_min_max(all_combi_data)
        min_vals.append(tmp_min_val)
        max_vals.append(tmp_max_val)
        y_data.append(all_combi_data)
        y_err_data.append(y_err)
    y_min_val = np.min(min_vals) - 0.05
    y_min_val = max(0, y_min_val)
    y_max_val = np.max(max_vals) + 0.05
    show_simi_plot(k_list, y_data, 'Number of important features (k)', 'Jaccard Similarity', '', \
                   (13, 12), '', y_err=y_err_data, x_min=np.min(k_list)-0.5, x_max=np.max(k_list)+0.5, \
                   y_min=y_min_val, y_max=y_max_val, if_model=True, dataset_name=dataset_name, \
                   second=second, if_comp_methods=True)
'''
END OF COMPARISON BETWEEN METHODS FUNCTIONS
'''
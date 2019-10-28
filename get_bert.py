import os
import utils
import pprint
import numpy as np
from collections import defaultdict, OrderedDict

### save_dir
REPO_DIR = os.path.dirname(os.path.abspath('data'))
DATA_ROOT = os.path.join(REPO_DIR, 'data')

SAVE_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
SAVE_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
SAVE_SST_DIR = os.path.join(DATA_ROOT, 'sst')

def get_shap_impt(span_name, test_tokens):
    path = 'data/SHAP_features/{}-bert-shap.npy'.format(span_name)
    att_weights = np.load(path)
    assert len(test_tokens) == len(att_weights)
    return att_weights

def get_aligned_shap_impt(dataset_name, span_name, max_length, explainer):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)

    att_weights = get_shap_impt(span_name, test_tokens)
    ws_spans, bert_spans = get_spans(span_name)

    aligned_att_weights, aligned_att_weights_len = [], []
    for i in range(len(test_tokens)):
        label = test_labels[i]
        class_idx = None
        if label == -1:
            class_idx = 0
        else:
            class_idx = 1
        length = len(bert_spans[i])
        tmp_avg_att_weights = None
        
        if len(bert_spans[i]) >= max_length: # if len >= max length, take whole list
            tmp_avg_att_weights = att_weights[i][class_idx][1:]
        else: # else, cut list into respective length
            tmp_avg_att_weights = att_weights[i][class_idx][1:1+length]
        att_score, att_score_len = get_att_score(ws_spans[i], bert_spans[i], tmp_avg_att_weights)
        aligned_att_weights.append(att_score)
        aligned_att_weights_len.append(att_score_len)
    assert len(aligned_att_weights) == len(test_tokens)
    return aligned_att_weights, aligned_att_weights_len

def get_max_att_weights(test_tokens, att_weights):
    ret = []
    for i in range(len(test_tokens)):
        avg_att = np.max(att_weights[i], axis=0)
        ret.append(avg_att)
    return ret

def get_avg_att_weights(test_tokens, att_weights):
    ret = []
    for i in range(len(test_tokens)):
        avg_att = np.average(att_weights[i], axis=0)
        ret.append(avg_att)
    return ret

def get_total_layers(span_name):
    att_weights = []
    if 'yelp' in span_name:
        path = 'data/BERT_att_weights/{}-bert-att-all1000.npy'.format(span_name)
        tmp1 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all2000.npy'.format(span_name)
        tmp2 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        tmp3 = np.load(path)
        test = np.concatenate((tmp1, tmp2), axis=1)
        att_weights = np.concatenate((test, tmp3), axis=1)
    elif 'sst' in span_name:
        path = 'data/BERT_att_weights/{}-bert-att-all1000.npy'.format(span_name)
        tmp1 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        tmp2 = np.load(path)
        att_weights = np.concatenate((tmp1, tmp2), axis=1)
    else:
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        att_weights = np.load(path)
    return len(att_weights)

def get_layer_att_weights(span_name, layer):
    att_weights = []
    if 'yelp' in span_name:
        path = 'data/BERT_att_weights/{}-bert-att-all1000.npy'.format(span_name)
        tmp1 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all2000.npy'.format(span_name)
        tmp2 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        tmp3 = np.load(path)
        test = np.concatenate((tmp1, tmp2), axis=1)
        att_weights = np.concatenate((test, tmp3), axis=1)
    elif 'sst' in span_name:
        path = 'data/BERT_att_weights/{}-bert-att-all1000.npy'.format(span_name)
        tmp1 = np.load(path)
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        tmp2 = np.load(path)
        att_weights = np.concatenate((tmp1, tmp2), axis=1)
    else:
        path = 'data/BERT_att_weights/{}-bert-att-all.npy'.format(span_name)
        att_weights = np.load(path)
    return att_weights[layer]

def get_aligned_att_weights(test_tokens, span_name, max_length, att_type, layer, impt=False):
    att_weights = None
    if impt == True:
        # use the last layer
        att_weights = get_layer_att_weights(span_name, 11)
    else:
        att_weights = get_layer_att_weights(span_name, layer)
    assert len(att_weights) == len(test_tokens)
    
    if att_type == 'avg':
        att_weights = get_avg_att_weights(test_tokens, att_weights)
    elif att_type == 'max':
        att_weights = get_max_att_weights(test_tokens, att_weights)
    ws_spans, bert_spans = get_spans(span_name)
    assert len(ws_spans) == len(test_tokens)
    assert len(bert_spans) == len(test_tokens)

    aligned_att_weights, aligned_att_weights_len = [], []
    for i in range(len(test_tokens)):
        length = len(bert_spans[i])
        tmp_avg_att_weights = None
        if len(bert_spans[i]) >= max_length:
            tmp_avg_att_weights = att_weights[i][1:]
        else:
            tmp_avg_att_weights = att_weights[i][1:1+length]
        att_score, att_score_len = get_att_score(ws_spans[i], bert_spans[i], tmp_avg_att_weights)
        aligned_att_weights.append(att_score)
        aligned_att_weights_len.append(att_score_len)
    assert len(aligned_att_weights) == len(test_tokens)
    return aligned_att_weights, aligned_att_weights_len

def get_att_score(whitespace_spans, bert_spans, att_weights):
    att_weights_len = len(att_weights)
    i = 0 # whitespace span idx
    j = 0 # bert span idx
    whitespace_scores = []
    tmp_score = 0
    while j < att_weights_len:
        bert_span_j_start_idx = bert_spans[j][0]
        bert_span_j_end_idx = bert_spans[j][1]
        whitespace_span_i_start_idx = whitespace_spans[i][0]
        whitespace_span_i_end_idx = whitespace_spans[i][1]
        
        denom = bert_span_j_end_idx - bert_span_j_start_idx + 1
        if bert_span_j_end_idx < whitespace_span_i_end_idx: 
            numer = bert_span_j_end_idx - max(bert_span_j_start_idx, whitespace_span_i_start_idx) + 1
            tmp_score += att_weights[j] * numer / denom
            j += 1
        else:
            numer = whitespace_span_i_end_idx - max(bert_span_j_start_idx, whitespace_span_i_start_idx) + 1
            tmp_score += att_weights[j] * numer / denom
            whitespace_scores.append(tmp_score)
            tmp_score = 0
            if whitespace_span_i_end_idx == bert_span_j_end_idx:
                j += 1
            i += 1
    return whitespace_scores, len(whitespace_scores)

def get_relevant_features(dataset_name, aligned_att_weights_len, tutorial_tokens=None):
    ret = []
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    
    if tutorial_tokens != None:
        for idx, text in enumerate(tutorial_tokens):
            length = aligned_att_weights_len[idx]
            relevant_tokens = text.split()[:length]
            assert len(relevant_tokens) == length
            relevant_tokens_str = ' '.join(relevant_tokens)
            ret.append(relevant_tokens_str)     
    else:
        for idx, text in enumerate(test_tokens):
            length = aligned_att_weights_len[idx]
            relevant_tokens = text.split()[:length]
            assert len(relevant_tokens) == length
            relevant_tokens_str = ' '.join(relevant_tokens)
            ret.append(relevant_tokens_str)
    return ret

def get_spans(span_name):
    ws_spans, bert_spans = '', ''
    bert_spans_path = 'data/{}/bert-spans.npz'.format(span_name)
    ws_spans_path = 'data/{}/ws-spans.npz'.format(span_name)
    bert_spans = np.load(bert_spans_path, allow_pickle=True)
    ws_spans = np.load(ws_spans_path, allow_pickle=True)
    return ws_spans, bert_spans

def create_bert_d(features_l, scores_l):
    ds = []
    for review_idx, review in enumerate(features_l):    
        d = defaultdict(lambda:[])
        tmp = scores_l[review_idx]
        for token_idx, token in enumerate(review.split()):
            #d[token].append(abs(tmp[token_idx])) # append and take average
            d[token].append(tmp[token_idx]) # append and take average
        d = get_average_shap(d) # take average
        ds.append(d)
    return ds

def get_bert_top_k(dataset_name, span_name, max_length, var):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)

    total_layers = get_total_layers(span_name)
    all_top_k = []
    for layer in range(total_layers):
        aligned_att_weights, avg_weights_len = None, None
        if var == 'avg':
            aligned_att_weights, avg_weights_len = get_aligned_att_weights(test_tokens, span_name, max_length, 'avg', layer)
        else:
            aligned_att_weights, avg_weights_len = get_aligned_att_weights(test_tokens, span_name, max_length, 'max', layer)
        relevant_tokens = get_relevant_features(dataset_name, avg_weights_len)

        var_d = create_bert_d(relevant_tokens, aligned_att_weights)
        top_k_tokens = [' '.join(l) for l in get_tokens_top_k(relevant_tokens, var_d, k)]
        all_top_k.append(top_k_tokens)
    assert len(all_top_k) == total_layers
    return all_top_k

def get_bert_svm_jacc(all_top_k, svm_top_k_tokens):
    layer_avg_score, all_y_err = [], []
    for idx_layer, layer_n_tokens in enumerate(all_top_k):
        total_jacc_score = []
        for idx_row, bert_tokens in enumerate(layer_n_tokens):
            svm_tokens = svm_top_k_tokens[idx_row]
            if idx_row == 0: 
                print(type(svm_tokens), type(bert_tokens.split()))
                print(svm_tokens, bert_tokens.split())
            _, svm_bert_jacc = jacc_simi(svm_tokens, bert_tokens.split())
            total_jacc_score.append(svm_bert_jacc)
        assert len(total_jacc_score) == len(all_top_k[0])

        avg_jacc_score = np.mean(total_jacc_score)
        y_err = stats.sem(total_jacc_score)
        layer_avg_score.append(avg_jacc_score)
        all_y_err.append(y_err)
    assert len(layer_avg_score) == len(all_top_k)
    return layer_avg_score, all_y_err

def get_y_data(dataset_name, span_name, save_dir, max_length, var, k):
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data(dataset_name)
    all_top_k = get_bert_top_k(dataset_name, span_name, max_length, var)
    svm_d = get_built_in(save_dir, 'svm', len(test_tokens))
    assert len(svm_d) == len(test_tokens)
    svm_top_k_tokens = get_tokens_top_k(test_tokens, svm_d, k)
    y_data, y_err = get_bert_svm_jacc(all_top_k, svm_top_k_tokens)
    return y_data, y_err

def save_files(save_dir, model_name, features_l, importance_l, explainer):
    features = 'features/{}_{}_all_features.pkl'.format(model_name, explainer)
    path = utils.get_abs_path(save_dir, features)
    utils.save_pickle(features_l, path)
    print('saved features at {}'.format(path))

    scores = 'feature_importance/{}_{}_all_scores.pkl'.format(model_name, explainer)
    path = utils.get_abs_path(save_dir, scores)
    utils.save_pickle(importance_l, path)
    print('saved scores at {}'.format(scores))
    
if __name__ == '__main__':
    '''
    BERT IMPORTANCE
    '''
    explainer = 'impt'
    layer_num = 11

    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('deception')
    aligned_att_weights, aligned_att_weights_len = get_aligned_att_weights(test_tokens, 'deception', 300, \
                                                                           'avg', layer_num, impt=True)
    relevant_tokens = get_relevant_features('deception', aligned_att_weights_len)
    save_files(SAVE_DECEPTION_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))

    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('yelp')
    aligned_att_weights, aligned_att_weights_len = get_aligned_att_weights(test_tokens, 'yelp', 512, \
                                                                           'avg', layer_num, impt=True)
    relevant_tokens = get_relevant_features('yelp', aligned_att_weights_len)
    save_files(SAVE_YELP_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))

    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('sst')
    aligned_att_weights, aligned_att_weights_len = get_aligned_att_weights(test_tokens, 'sst', 128, \
                                                                           'avg', layer_num, impt=True)
    relevant_tokens = get_relevant_features('sst', aligned_att_weights_len)
    save_files(SAVE_SST_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))
    
    
    '''
    BERT LIME
    '''
    fp = 'data/LIME_features/deception/new_feature_l.npy'
    sp = 'data/LIME_features/deception/new_scores_l.npy'
    features = np.load(fp)
    scores = np.load(sp)
    print(len(features), len(scores))
    tmp = 'features/bert_lime_all_features.pkl'
    path = utils.get_abs_path(SAVE_DECEPTION_DIR, tmp)
    utils.save_pickle(features, path)
    tmp = 'feature_importance/bert_lime_all_scores.pkl'
    path = utils.get_abs_path(SAVE_DECEPTION_DIR, tmp)
    utils.save_pickle(scores, path)
    
    fp = 'data/LIME_features/deception/new_feature_l.npy'
    sp = 'data/LIME_features/deception/new_scores_l.npy'
    features = np.load(fp)
    scores = np.load(sp)
    print(len(features), len(scores))
    tmp = 'features/bert_lime_all_features.pkl'
    path = utils.get_abs_path(SAVE_YELP_DIR, tmp)
    utils.save_pickle(features, path)
    tmp = 'feature_importance/bert_lime_all_scores.pkl'
    path = utils.get_abs_path(SAVE_YELP_DIR, tmp)
    utils.save_pickle(scores, path)
    
    fp = 'data/LIME_features/deception/new_feature_l.npy'
    sp = 'data/LIME_features/deception/new_scores_l.npy'
    features = np.load(fp)
    scores = np.load(sp)
    print(len(features), len(scores))
    tmp = 'features/bert_lime_all_features.pkl'
    path = utils.get_abs_path(SAVE_SST_DIR, tmp)
    utils.save_pickle(features, path)
    tmp = 'feature_importance/bert_lime_all_scores.pkl'
    path = utils.get_abs_path(SAVE_SST_DIR, tmp)
    utils.save_pickle(scores, path)
    
    '''
    BERT SHAP
    '''
    explainer = 'shap'

    aligned_att_weights, aligned_att_weights_len = get_aligned_shap_impt('deception', 'deception', 300, explainer)
    relevant_tokens = get_relevant_features('deception', aligned_att_weights_len)
    save_files(SAVE_DECEPTION_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))

    aligned_att_weights, aligned_att_weights_len = get_aligned_shap_impt('yelp', 'yelp', 512, explainer)
    relevant_tokens = get_relevant_features('yelp', aligned_att_weights_len)
    save_files(SAVE_YELP_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))

    aligned_att_weights, aligned_att_weights_len = get_aligned_shap_impt('sst', 'sst', 128, explainer)
    relevant_tokens = get_relevant_features('sst', aligned_att_weights_len)
    save_files(SAVE_SST_DIR, 'bert', relevant_tokens, aligned_att_weights, explainer)
    print(len(aligned_att_weights), len(aligned_att_weights_len))
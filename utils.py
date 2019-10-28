import os
import json
import gzip
import spacy
import pickle
import random
import numpy as np

def get_abs_path(save_dir, file_name):
    __file__ = os.path.join(save_dir, file_name)
    path = os.path.abspath(__file__)
    return path

def save_pickle(content, path):
    pickle.dump(content, open(path, "wb"))
    print('saved {} pickle..'.format(path))
    
def load_pickle(path, encoding=True):
    if encoding == False:
        return pickle.load(open(path, 'rb'), encoding='latin1')
    else:
        return pickle.load(open(path, 'rb'))

def write_json_list(data, filename):
    with gzip.open(filename, "wt") as fout:
        for d in data:
            fout.write("%s\n" % json.dumps(d))

def load_json_list(filename):
    data = []
    with gzip.open(filename, "rt") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def create_json_list(data, out_path):
    file = gzip.open(out_path,'w+') 
    for i in data:
        i_str = json.dumps(i) + '\n'
        i_bytes = i_str.encode('utf-8')
        file.write(i_bytes)
    file.close()
    
def get_uci(data):
    ret = []
    for d in data:
        tmp = []
        for idx, feature in enumerate(d):
            if idx != len(d)-1: # ignore result column
                tmp.append(d[feature])
        ret.append(tmp)
    return ret

def get_tokens_labels(dataset_name, save_dir):
    # train dataset
    file_name = '{}_train.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    train_data = load_json_list(path)
    train_tokens = [d["tokens"] for d in train_data]
    train_pos = [d["pos"] for d in train_data]
    train_labels = [d["label"] for d in train_data]
    # dev dataset
    file_name = '{}_dev.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    dev_data = load_json_list(path)
    dev_tokens = [d["tokens"] for d in dev_data]
    dev_pos = [d["pos"] for d in dev_data]
    dev_labels = [d["label"] for d in dev_data]
    # train_dev dataset
    file_name = '{}_train_dev.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    train_dev_data = load_json_list(path)
    train_dev_tokens = [d["tokens"] for d in train_dev_data]
    train_dev_pos = [d["pos"] for d in train_dev_data]
    train_dev_labels = [d["label"] for d in train_dev_data]
    # test dataset
    file_name = '{}_test.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    test_data = load_json_list(path)
    test_tokens = [d["tokens"] for d in test_data]
    test_pos = [d["pos"] for d in test_data]
    test_labels = [d["label"] for d in test_data]
    return train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
            train_labels, dev_labels, train_dev_labels, test_labels

def get_pos(dataset_name, save_dir):
    # train dataset
    file_name = '{}_train.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    train_data = load_json_list(path)
    train_pos = [d["pos"] for d in train_data]
    # dev dataset
    file_name = '{}_dev.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    dev_data = load_json_list(path)
    dev_pos = [d["pos"] for d in dev_data]
    # train_dev dataset
    file_name = '{}_train_dev.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    train_dev_data = load_json_list(path)
    train_dev_pos = [d["pos"] for d in train_dev_data]
    # test dataset
    file_name = '{}_test.jsonlist.gz'.format(dataset_name)
    path = get_abs_path(save_dir, file_name)
    test_data = load_json_list(path)
    test_pos = [d["pos"] for d in test_data]
    return train_pos, dev_pos, train_dev_pos, test_pos
    
def get_uci_tokens_labels(save_dir):
    # train dataset
    path = get_abs_path(save_dir, 'uci_train.jsonlist.gz')
    train_data = load_json_list(path)
    train_tokens = get_uci(train_data)
    train_labels = [d["label"] for d in train_data]
    # dev dataset
    path = get_abs_path(DATA_UCI_DIR, 'uci_dev.jsonlist.gz')
    dev_data = load_json_list(path)
    dev_tokens = get_uci(dev_data)
    dev_labels = [d["label"] for d in dev_data]
    # train dev dataset
    path = get_abs_path(save_dir, 'uci_train_dev.jsonlist.gz')
    train_data = load_json_list(path)
    train_tokens = get_uci(train_data)
    train_labels = [d["label"] for d in train_data]
    # test dataset
    path = get_abs_path(save_dir, 'uci_test.jsonlist.gz')
    test_data = load_json_list(path)
    test_tokens = get_uci(test_data)
    test_labels = [d["label"] for d in test_data]
    return train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
            train_labels, dev_labels, train_dev_labels, test_labels
    
def load_data(dataset_name):
    REPO_DIR = os.path.dirname(os.path.abspath('data'))
    DATA_ROOT = os.path.join(REPO_DIR, 'data')
    DATA_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
    DATA_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
    DATA_SST_DIR = os.path.join(DATA_ROOT, 'sst')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens = [], [], [], []
    train_labels, dev_labels, train_dev_labels, test_labels = [], [], [], []
    if dataset_name == 'deception':
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = get_tokens_labels(dataset_name, DATA_DECEPTION_DIR)
    elif 'yelp' in dataset_name:
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = get_tokens_labels(dataset_name, DATA_YELP_DIR)
    elif 'sst' in dataset_name:
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = get_tokens_labels(dataset_name, DATA_SST_DIR)
    elif dataset_name == 'uci':
        train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
        train_labels, dev_labels, train_dev_labels, test_labels = get_uci_tokens_labels(DATA_UCI_DIR)
    return train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
            train_labels, dev_labels, train_dev_labels, test_labels

def get_tokens_pos(review, nlp, lower=True):
    #nlp = spacy.load("en")
    doc = nlp(review)
    tokens, pos, tag = [], [], []
    for token in doc:
        tmp = token.text
        if lower:
            tmp = token.text.lower()
        tokens.append(tmp)
        pos.append(token.pos_)
        tag.append(token.tag_)
    tokens = " ".join(tokens)
    pos = " ".join(pos)
    tag = " ".join(tag)
    return tokens, pos, tag

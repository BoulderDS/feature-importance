import os
import torch
import utils
import lstm as lc
from sklearn.metrics import accuracy_score

### save_dir
REPO_DIR = os.path.dirname(os.path.abspath('data'))
DATA_ROOT = os.path.join(REPO_DIR, 'data')

SAVE_DECEPTION_DIR = os.path.join(DATA_ROOT, 'deception')
SAVE_YELP_DIR = os.path.join(DATA_ROOT, 'yelp')
SAVE_SST_DIR = os.path.join(DATA_ROOT, 'sst')

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

def run(train_dev_tokens, test_tokens, save_dir):
    file = 'lstm_att'
    model = 'models/{}.pkl'.format(file)
    path = utils.get_abs_path(save_dir, model)
    d_file = 'lstm_att_hp'
    hp_d = 'models/{}.pkl'.format(d_file)
    hp_path = utils.get_abs_path(save_dir, hp_d)
    d = utils.load_pickle(hp_path)
    model = init_model(train_dev_tokens, d, path)
    test_split_tokens = split_tokens(test_tokens)
    mapping = [model.get_words_to_ids(l) for l in test_split_tokens]
    predictions, word_score_ds = model.predict(test_split_tokens, mapping, True)
    accuracy = accuracy_score(test_labels, predictions)
    return word_score_ds

def save(word_score_ds, save_dir):
    features_l, scores_l = get_att_weights(word_score_ds)
    save_att_weights(word_score_ds, save_dir)
    
def get_att_weights(word_score_ds):
    features_l, scores_l = [], []
    for word_score_d in word_score_ds:
        tmp_features = list(word_score_d.keys())
        features = ' '.join(tmp_features)
        scores = list(word_score_d.values())
        features_l.append(features)
        scores_l.append(scores)
    return features_l, scores_l

def save_att_weights(word_score_ds, save_dir):
    features_l, importance_l = get_att_weights(word_score_ds)
    features_file_name = 'features/lstm_att_weights_all_features.pkl'
    path = utils.get_abs_path(save_dir, features_file_name)
    utils.save_pickle(features_l, path)
    scores_file_name = 'feature_importance/lstm_att_weights_all_scores.pkl'
    path = utils.get_abs_path(save_dir, scores_file_name)
    utils.save_pickle(importance_l, path)
    
def do(train_dev_tokens, test_tokens, save_dir):
    word_score_ds = run(train_dev_tokens, test_tokens, save_dir)
    save(word_score_ds, save_dir)
    

if __name__ == "__main__":
    ### deception
    print('=== deception ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('deception')
    do(train_dev_tokens, test_tokens, SAVE_DECEPTION_DIR)
    
    ### yelp binary
    print('=== yelp binary ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('yelp')
    do(train_dev_tokens, test_tokens, SAVE_YELP_DIR)
    
    ## sst binary
    print('=== sst binary ===')
    train_tokens, dev_tokens, train_dev_tokens, test_tokens, \
    train_labels, dev_labels, train_dev_labels, test_labels = utils.load_data('sst')
    do(train_dev_tokens, test_tokens, SAVE_SST_DIR)
    
    
    
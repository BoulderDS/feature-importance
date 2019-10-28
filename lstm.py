
"""
This file uses torch to implement LSTM and LSTM with attention
"""
import os
import utils
import collections
import random
import torch
import torch.cuda as tcuda
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import accuracy_score, mean_squared_error
from nltk.tokenize import wordpunct_tokenize

torch.manual_seed(1)

UNK_TOKEN = "unk"

def tokenize(text, lower=True):
    words = wordpunct_tokenize(text)
    if lower:
        return [w.lower() for w in words]
    else:
        return words

class LSTMClassifier(nn.Module):
    def __init__(self,
                 data,
                 emb_dim,
                 hidden_dim, 
                 num_layers,
                 min_count,
                 non_trainable=False,
                 bidirectional=False, 
                 glove_embedding_file=None,
                 use_gpu=True):
        if use_gpu:
            self.torch = tcuda
        else:
            self.torch = torch
        super(LSTMClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_count = min_count
        self.non_trainable = non_trainable
        self.bidirectional = bidirectional
        self.glove_embedding_file = glove_embedding_file
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        if bidirectional:
            self.prediction_layer = nn.Linear(hidden_dim * 2, 2)
        else:
            self.prediction_layer = nn.Linear(hidden_dim, 2)
        self.create_word_id_dict(data, self.non_trainable)
        self.use_gpu = use_gpu

    def create_word_id_dict(self, texts, non_trainable):
        self.word_dict = {}
        self.word_dict[UNK_TOKEN] = 0
        counter = collections.Counter()
        for text in texts:
            for token in text:
                counter[token] += 1
        for token in counter:
            if counter[token] < self.min_count:
                continue
            self.word_dict[token] = len(self.word_dict)
        if self.glove_embedding_file:
            self.word_embeddings = self.create_glove_embedding_layer(non_trainable=non_trainable)
        else:
            self.word_embeddings = nn.Embedding(len(self.word_dict), self.emb_dim)
            
    def get_words_to_ids(self, text):
        ids = []
        for token in text:
            if token not in self.word_dict:
                ids.append(self.word_dict[UNK_TOKEN])
            else:
                ids.append(self.word_dict[token])
        # force this on cpu to save space
        return self.torch.LongTensor(ids)
        #return ids
    
    
    def get_words_to_ids_masks(self, text, padding_length):
        ids, masks = np.zeros((padding_length, len(self.word_dict))), []
        for (i, token) in enumerate(text):
            if i >= padding_length:
                break
            if token not in self.word_dict:
                ids[i, 0] = 1
            else:
                ids[i, self.word_dict[token]] = 1
            masks.append(1)
        if padding_length > len(text):
            for _ in range(padding_length - len(text)):
                masks.append(0)
        return torch.FloatTensor(ids), masks
    
    def create_glove_embedding_layer(self, non_trainable=False):
        weight_matrix = np.zeros((len(self.word_dict), self.emb_dim))
        glove_embedding, avg = utils.load_glove_embedding(self.glove_embedding_file, self.word_dict)
        for word, word_id in self.word_dict.items():
            if word in glove_embedding:
                weight_matrix[word_id] = glove_embedding[word]
            else:
                # if word cannot be found in glove
                weight_matrix[word_id] = avg
        weight_matrix = self.torch.FloatTensor(weight_matrix)
        emb_layer = nn.Embedding(len(self.word_dict), self.emb_dim)
        emb_layer.load_state_dict({'weight': weight_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer
    
    def init_hidden(self):
        if self.bidirectional:
            count = self.num_layers * 2
        else:
            count = self.num_layers
        if self.use_gpu:
            return (torch.zeros(count, 1, self.hidden_dim).cuda(), torch.zeros(count, 1, self.hidden_dim).cuda())
        else: 
            return (torch.zeros(count, 1, self.hidden_dim), torch.zeros(count, 1, self.hidden_dim))

    def make_target(self, label):
        if label == -1:
            return self.torch.LongTensor([0])
        else:
            return self.torch.LongTensor([1])

    def fit(self, data, labels, learning_rate, epochs=30, optimizer_name=None,
            val_data=None, val_labels=None, model_prefix=None, save_dir=None, optim_checkpoint=None):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if optim_checkpoint != None:
            optimizer.load_state_dict(optim_checkpoint)
        loss_record = {"train": {}, "val": {}}
        for epoch in range(epochs):
            cnt, total_loss = 0, 0
            idx = list(range(len(data)))
            random.shuffle(idx) # reduce variance and allow model to remain general
            for i in idx:
                self.zero_grad() # clears old gradient from the last step
                words_to_ids = self.get_words_to_ids(data[i])
                prediction = self.forward(data[i], words_to_ids)
                loss = loss_function(prediction, self.make_target(labels[i]))
                loss.backward() # computes derivatives of loss using backprop
                optimizer.step() # tells optimizer to take a step based on
                total_loss += loss.item()
                cnt += 1
            print("#epoch %d #instance %d training loss %f" % (epoch, cnt, total_loss / cnt))
            loss_record["train"][epoch] = total_loss / cnt 
            mappings = [self.get_words_to_ids(i) for i in data]
            train_predictions = self.predict(data, mappings)
            train_accuracy = accuracy_score(labels, train_predictions)
            print("#epoch %d #instance %d training accuracy %f" % (epoch, cnt, train_accuracy))
            
            if val_data is not None:
                mappings = [self.get_words_to_ids(i) for i in val_data]
                val_predictions = self.predict(val_data, mappings)
                error = mean_squared_error(val_labels, val_predictions)
                loss_record["val"][epoch] = error
                print("#epoch %d validation loss %f" % (epoch, error))
                val_accuracy = accuracy_score(val_labels, val_predictions)
                print("#epoch %d #instance %d validation accuracy %f" % (epoch, len(val_data), val_accuracy))
                if model_prefix:
                    name = 'models/tuning/{}_{}.pkl'.format(model_prefix, epoch)
                    __file__ = os.path.join(save_dir, name)
                    out_path = os.path.abspath(__file__)
                    torch.save({
                                'model_state_dict': self.state_dict(),
                                'optim_state_dict': optimizer.state_dict()
                                }, out_path)
                    print('saved {} pickle..'.format(out_path))
        return loss_record
        
    def predict(self, texts):
        predictions = []
        for text in texts:
            prediction = self.forward(text)
            _, label = torch.max(prediction, 1)
            label = label.item()
            if label == 0:
                predictions.append(-1)
            else:
                predictions.append(1)
        return predictions

    def forward(self, d):
        words_to_ids = self.get_words_to_ids(d)
        hidden = self.init_hidden()
        emb = self.word_embeddings((words_to_ids))
        out, _ = self.lstm(emb.view(len(words_to_ids), 1, -1), hidden)
        if self.bidirectional:
            out = torch.cat((out[-1, :, :self.hidden_dim], out[0, :, self.hidden_dim:]), 1)
        else:
            out = out[-1, :, :]
        prediction = self.prediction_layer(out)
        return prediction
    
    def load_model(self, filename, glove):
        self.load_state_dict(torch.load(filename))

    def save_model(self, filename):
        torch.save({self.state_dict()}, filename)
        
        
class LSTMAttentionClassifier(LSTMClassifier):
    def __init__(self,
                 data,
                 emb_dim,
                 hidden_dim, 
                 num_layers,
                 min_count,
                 non_trainable=False,
                 bidirectional=False, 
                 glove_embedding_file=None,
                 use_gpu=True):
        super(LSTMAttentionClassifier, self).__init__(data, 
                                                      emb_dim, 
                                                      hidden_dim,
                                                      num_layers=num_layers,
                                                      min_count=min_count,
                                                      non_trainable=non_trainable,
                                                      bidirectional=bidirectional,
                                                      glove_embedding_file=glove_embedding_file,
                                                      use_gpu=use_gpu) 

        if bidirectional:
            self.context_vector = nn.Linear(hidden_dim * 2, 1)
        else:
            self.context_vector = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, text, words_to_ids, save_features=False, top_k_features_str=None):
        hidden = self.init_hidden()
        emb = self.word_embeddings((words_to_ids))
        out, _ = self.lstm(emb.view(len(words_to_ids), 1, -1), hidden)
        out = out.squeeze(1) # very easy for no batches
        attention = self.context_vector(out)
        weights = self.softmax(attention)
        
        # save top k features
        if save_features:
            word_score_dict = defaultdict(lambda:[])
            for index, w in enumerate(weights): 
                word_score_dict[text[index]].append(float(w))

            # take average weights if a token has more than 1 weight
            avg_word_score_d = {}
            for k, v in word_score_dict.items():
                if len(v) != 1:
                    v = np.mean(v)
                else:
                    v = v[0]
                avg_word_score_d[k] = v
        
        # manually change weights
        # purpose of this part is to predict using top k features
        # when weights are manually adjusted, only top k features have weights,
        # the rest of the features are assigned 0
        if top_k_features_str:
            # top_k_features_str is a string
            # change it to tokens
            top_k_features = top_k_features_str.split()
            features_to_ids = self.get_words_to_ids(top_k_features)
            
            # get index of features
            feature_index = []
            for index, _id in enumerate(words_to_ids):
                if _id in features_to_ids:
                    feature_index.append(index)
            
            for index, w in enumerate(weights):
                weights[index] = torch.tensor(0)
                if index in feature_index:
                    weights[index] = torch.tensor(1 / len(feature_index))
        
        weights = weights.transpose(1, 0)
        out = torch.matmul(weights, out)
        prediction = self.prediction_layer(out)
        
        if save_features:
            return prediction, avg_word_score_d
        return prediction

    def predict(self, texts, words_to_ids, save=False, features=None, return_probablity=False):
        predictions = []
        word_score_ds = []
        for index, text in tqdm(enumerate(texts)):
            if len(text) == 0:
                if return_probablity:
                    label = [0.5, 0.5]
                else:
                    label = random.randint(0, 1) * 2 - 1
            else:
                prediction = self.forward(text, words_to_ids[index])
                if save:
                    prediction, avg_word_score_d = self.forward(text, words_to_ids[index], True)
                    word_score_ds.append(avg_word_score_d)
                if features:
                    prediction = self.forward(text, words_to_ids[index], False, features[index])
                if return_probablity:
                    label = torch.softmax(prediction, 1)[0]
                    label = label.cpu().detach().numpy()
                else:
                    _, label = torch.max(prediction, 1)
                    label = int(label.item()) * 2 - 1
            predictions.append(label)
        predictions = np.array(predictions)
        if save:
            return predictions, word_score_ds
        return predictions
    
    
    def forward_shap(self, token_ids, mask, full_id_matrix=False):
        if not token_ids.is_cuda:
            token_ids = token_ids.cuda()
        hidden = self.init_hidden()
        if not full_id_matrix:
            emb = self.word_embeddings(self.torch.LongTensor(token_ids))
        else:
            emb = torch.matmul(token_ids, self.word_embeddings.weight)
        out, _ = self.lstm(emb.view(len(token_ids), 1, -1), hidden)
        out = out.squeeze(1) # very easy for no batches
        attention = self.context_vector(out)
        if mask is not None:
            # need more cuda care
            mask = self.torch.FloatTensor(mask).reshape(attention.shape)
            attention = torch.nn.functional.softmax(attention * mask, dim=-1)
            attention = attention * mask
            weights = attention / (attention.sum(dim=-1, keepdim=True) + 1e-13)
        else:
            weights = self.softmax(attention)
        weights = weights.transpose(1, 0)
        out = torch.matmul(weights, out)
        prediction = self.prediction_layer(out)
        return prediction


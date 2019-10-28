from sklearn.svm import LinearSVC, SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, ParameterGrid

import numpy as np
import scipy.stats as ss

class BOWFeatures(TfidfVectorizer):
    def __init__(self, tokenizer=None, norm="l2",
                 use_idf=False, max_df=1.0, min_df=5,
                 lowercase=True, ngram_range=(1, 1), 
                 analyzer='word', token_pattern=r"\S+"):
        super(BOWFeatures, self).__init__(tokenizer=tokenizer,
                                          norm=norm,
                                          max_df=max_df,
                                          min_df=min_df,
                                          lowercase=lowercase,
                                          use_idf=use_idf,
                                          ngram_range=ngram_range,
                                          analyzer=analyzer,
                                          token_pattern=token_pattern) # splits words by spaces

    def fit(self, data, y=None):
        return super(BOWFeatures, self).fit(data, y)

    def transform(self, data, y=None):
        transform_data = data
        if type(data) is np.ndarray:
            transform_data = []
            for i in data[0]:
                transform_data.append(i)
        X = super(BOWFeatures, self).transform(transform_data)
        return X

    def fit_transform(self, X, y=None):
        # make sure that the base class does not do "clever" things
        return self.fit(X, y).transform(X, y)

    def get_feature_names(self):
        feature_names = super(BOWFeatures, self).get_feature_names()
        return feature_names

def get_logistic_regression(C=1.0, penalty="l2", dual=False,
                            fit_intercept=False):
    pipe_end = [('clf',
                 LogisticRegression(random_state=0,
                                    penalty=penalty,
                                    dual=dual,
                                    fit_intercept=fit_intercept,
                                    C=C))]
    return pipe_end

def get_SVC(C=1.0, kernel="linear"):
    pipe_end = [('clf',
                 SVC(random_state=0,
                      kernel=kernel,
                      probability=True,
                      C=C))]
    return pipe_end

def get_linear_SVC(C=1.0, penalty="l2", dual=True, fit_intercept=False):
    pipe_end = [('clf', 
                  LinearSVC(random_state=0, 
                            penalty=penalty, 
                            dual=dual, 
                            fit_intercept=fit_intercept, 
                            C=C))]

    return pipe_end

def get_linear_SVC_l1(C=1.0, penalty="l1", dual=False, fit_intercept=False):
    pipe_end = [('clf', 
                  LinearSVC(random_state=0, 
                            penalty=penalty,
                            dual=dual, 
                            fit_intercept=fit_intercept, 
                            C=C))]

    return pipe_end

def get_random_forest(max_depth=50, n_estimators=400, max_features='auto',
                      min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    pipe_end = [('clf', 
                  RandomForestClassifier(max_depth=max_depth,
                                         n_estimators=n_estimators,
                                         max_features=max_features,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         bootstrap=bootstrap))]

    return pipe_end

def get_light_gbm(boosting_type='gbdt', objective='binary', num_leaves=6, 
                  max_bin=512, learning_rate=0.1, n_estimators=100, 
                  device='gpu', reg_lambda=1):
    pipe_end = [('clf', 
                  LGBMClassifier(boosting_type=boosting_type, 
                                 objective=objective,
                                 num_leaves=num_leaves, 
                                 max_bin=max_bin,
                                 learning_rate=learning_rate, 
                                 n_estimators=n_estimators, 
                                 device=device,
                                 reg_lambda=reg_lambda))]

    return pipe_end



def get_xgb(silent=False, scale_pos_weight=1, learning_rate=0.01, 
            colsample_bytree=0.4, subsample=0.8, 
            objective='binary:logistic', n_estimators=1000, 
            reg_alpha=0.3, max_depth=4, gamma=10,
            min_child_weight=1):
    pipe_end = [('clf', 
                 XGBClassifier(silent=silent, 
                               scale_pos_weight=scale_pos_weight,
                               learning_rate=learning_rate, 
                               colsample_bytree=colsample_bytree,
                               subsample=subsample,
                               objective=objective, 
                               n_estimators=n_estimators, 
                               reg_alpha=reg_alpha,
                               max_depth=max_depth, 
                               gamma=gamma, 
                               min_child_weight=min_child_weight))]

    return pipe_end

def cross_val_train(train_data, train_y, training_pipeline, parameter_grid,
                    log_file=None, verbose=True, func=accuracy_score, cv=5):
    """Using a validation set instead of cross validation"""
    grid = ParameterGrid(parameter_grid)
    params, scores, best_score, best_params = [], [], -1, None
    if func == mean_squared_error:
        best_score = 100
    for idx, g in enumerate(grid):
        print('running {}/{} in grid ...'.format(idx+1, len(grid)))
        training_pipeline.set_params(**g)
        cv_scores = cross_val_score(training_pipeline, train_data, train_y, cv=cv, scoring=make_scorer(func))
        accuracy = np.mean(cv_scores)
        print('cv score: {}'.format(accuracy))
        params.append(g)
        scores.append(accuracy)
        if (accuracy > best_score and func == accuracy_score) \
           or (accuracy < best_score and func == mean_squared_error):
            best_score = accuracy
            best_params = g
    training_pipeline.set_params(**best_params)
    training_pipeline.fit(train_data, train_y)
    return training_pipeline, best_score, best_params, params, scores

def heldout_test(estimator, heldout_data, heldout_y,
                 filename=None, func=accuracy_score,
                 return_std=False):
    pred = estimator.predict(heldout_data)
    accuracy = func(heldout_y, pred)
    if return_std:
        return pred, accuracy, ss.sem([t == p for (t, p) in zip(heldout_y, pred)])
    return pred, accuracy



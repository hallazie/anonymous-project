# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: xgb_approach.py
# @time: 2020/12/21 21:22
# @desc:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import *
from score import scoring

import joblib
import numpy as np
import datatable as dt
import os

train = dt.fread(TRAIN_PATH)
train = train.to_pandas()
train = train[train['weight'] != 0]
train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']
w_train = train.loc[:, 'weight']
r_train = train.loc[:, 'resp']
d_train = train.loc[:, 'date']

X_train, X_test, y_train, y_test, w_train, w_test, r_train, r_test, d_train, d_test = train_test_split(X_train, y_train, w_train, r_train, d_train, random_state=666, test_size=0.08)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f'train and test set init finished with size: {X_train.shape} {y_train.shape}, {X_test.shape} {y_test.shape}')
print(f'train date: {len(set(d_train))}, test date: {len(set(d_test))}')

model_size = 1


params = {
    'verbosity': 2,
    'max_depth': 8,
    'n_estimators': 500,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'tree_method': 'gpu_hist',
    'random_state': 666
}


params_grid = {
    'verbosity': 2,
    'max_depth': [5, 6, 7, 8, 9, 10],
    'n_estimators': [256, 512, 568, 1024],
    'learning_rate': [0.01, 0.05, 0.001],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'tree_method': 'gpu_hist',
    'random_state': 666
}


def train():
    for i in range(model_size):
        model = XGBClassifier(**params)
        print('start training...')
        model.fit(X_train, y_train)
        joblib.dump(model, f'checkpoints/baseline-xgb-{i}.pkl')


def test():
    pred_list = []
    for i in range(model_size):
        path = f'checkpoints/baseline-xgb-{i}.pkl'
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        print('start validation...')
        pred = model.predict(X_test)
        pred_list.append(pred)
    pred_final = np.array(pred_list)
    pred_final = pred_final.transpose().mean(axis=1)
    score = scoring(date_list=list(d_test), weight_list=list(w_test), resp_list=list(r_test), action_list=list(pred_final))
    print(score)


if __name__ == '__main__':
    train()
    test()


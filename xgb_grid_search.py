# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: xgb_approach.py
# @time: 2020/12/21 21:22
# @desc:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import *
from score import scoring

import joblib
import numpy as np
import pandas as pd
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


params_grid = {
    'max_depth': [12, 13, 14],
    'n_estimators': [1280, 1408],
    'learning_rate': [0.01],
    'subsample': [0.9],
    'colsample_bytree': [1.0],
}


xgb = XGBClassifier(tree_method='gpu_hist', random_state=30, objective='binary:logistic', silent=True, nthread=1)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=30)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=params_grid,
    scoring='roc_auc',
    verbose=3,
    cv=skf.split(X=X_train, y=y_train)
)

grid_search.fit(X_train, y_train)


print('\n All results:')
print(grid_search.cv_results_)
print('\n Best estimator:')
print(grid_search.best_estimator_)
print('\n Best score:')
print(grid_search.best_score_ * 2 - 1)
print('\n Best parameters:')
print(grid_search.best_params_)
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('output/xgb-grid-search-results-02.csv', index=False)



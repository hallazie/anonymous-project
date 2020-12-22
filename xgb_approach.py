# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: xgb_approach.py
# @time: 2020/12/21 21:22
# @desc:

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import *
from score import scoring

import joblib
import datatable as dt

train = dt.fread(TRAIN_PATH)
train = train.to_pandas()
train = train[train['weight'] != 0]
train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']
w_train = train.loc[:, 'weight']
r_train = train.loc[:, 'resp']
d_train = train.loc[:, 'date']

X_train, X_test, y_train, y_test, w_train, w_test, r_train, r_test, d_train, d_test = train_test_split(X_train, y_train, w_train, r_train, d_train,random_state=666, test_size=0.2)
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

print(f'train and test set init finished with size: {X_train.shape} {y_train.shape}, {X_test.shape} {y_test.shape}')
print(f'train date: {len(set(d_train))}, test date: {len(set(d_test))}')


params1 = {
    'verbosity': 1,
    'max_depth': 8,
    'n_estimators': 500,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'tree_method': 'gpu_hist',
    'random_state': 666
}


def train():
    model = XGBClassifier(**params1)
    model = MLPClassifier(hidden_layer_sizes=(256, 256, 512, 512, 128), verbose=2, activation='logistic')
    print('start training...')
    model.fit(X_train, y_train)
    joblib.dump(model, 'checkpoints/baseline-mlp.pkl')


def test():
    model = joblib.load('checkpoints/baseline-mlp.pkl')
    print('start validation...')
    pred = model.predict(X_test)
    score = scoring(date_list=list(d_test), weight_list=list(w_test), resp_list=list(r_test), action_list=list(pred))
    print(score)


if __name__ == '__main__':
    train()
    test()


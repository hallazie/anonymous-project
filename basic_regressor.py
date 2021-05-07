# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from utils import segment
from config import *
from tfidf import VocabVectorizer

import pandas as pd
import numpy as np
import os


vectorizer = VocabVectorizer(retrain=True)
regressor = MLPRegressor(hidden_layer_sizes=(128, 128, 256, 256, 512))
df = pd.read_csv(os.path.join(BASE_DIR, 'data/train.csv')).fillna('')
x = np.array([vectorizer.vectorize(x[3]) for x in df.values])
y = np.array([x[4] for x in df.values])

print(f'start cross-validate on data size: {x.shape}, {y.shape}')
cv_res = cross_validate(regressor, x, y, cv=10, scoring=make_scorer(mean_squared_error), verbose=10)['test_score']
avg = sum(cv_res) / float(len(cv_res))
print(f'cv-score: {cv_res}')
print(f'cv-score average: {avg}')

# print(x.shape, y.shape)
# trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.1)
# print('start training...')
# regressor.fit(trainx, trainy)
# print('finished training...')
# predict = regressor.predict(testx)
# score = mean_squared_error(testy, predict)
# print(f'test rmse={score}')




# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from config import *
from utils import segment

import joblib
import os
import random
import numpy as np
import pandas as pd


class VocabVectorizer:
    def __init__(self, retrain=False):
        self.retrain = retrain
        self.counter = CountVectorizer(min_df=5, max_df=0.5, analyzer='word', token_pattern='\\b\\w+\\b')
        self.tfidf = TfidfTransformer()
        self._init_model()

    def _init_model(self):
        if not os.path.exists('model/tfidf.pkl') or self.retrain:
            self.train()
            print('train tfidf from scratch...')
            joblib.dump(self.tfidf, os.path.join(BASE_DIR, 'model/tfidf.pkl'))
            joblib.dump(self.counter, os.path.join(BASE_DIR, 'model/counter.pkl'))
        else:
            print('load trained tfidf...')
            self.tfidf = joblib.load(os.path.join(BASE_DIR, 'model/tfidf.pkl'))
            self.counter = joblib.load(os.path.join(BASE_DIR, 'model/counter.pkl'))

    def train(self):
        df = pd.read_csv(os.path.join(BASE_DIR, 'data/train.csv')).fillna('')
        lines = [x[3] for x in df.values]
        random.shuffle(lines)
        corpus = [' '.join(segment(line)) for line in lines]
        vect = self.counter.fit_transform(corpus)
        self.tfidf.fit(vect)

    def vectorize(self, text):
        """
        return sparse vector, should call array.toarray() for further use
        """
        segs = ' '.join(segment(text))
        vec = self.tfidf.transform(self.counter.transform(np.array([segs])))
        return vec.toarray()[0]






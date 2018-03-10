# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc

from scipy.stats import mode
# customes 
def mape_object(y,d):
    d = d.get_label()
    g = 1.0*np.sign(y-d)/d
    h = 1.0/d
    return g,h
# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    result= np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result,False

def add_constact(df):
    return np.sum(1.0/df) / np.sum(1.0/df/df)
# 中位数
def mode_function(df):
    df = df.astype(int)
    counts = mode(df)
    return counts[0][0]
# load or create your dataset


print('Start training...')
# train
lgb_train = lgb.Dataset(train, train_label)
lgb_eval = lgb.Dataset(test, test_label, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 128,
    'learning_rate': 0.002,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                # init_model=gbm,
                fobj=mape_object,
                feval=mape_ln,
                valid_sets=lgb_eval,
                early_stopping_rounds = 5)

print('Start predicting...')

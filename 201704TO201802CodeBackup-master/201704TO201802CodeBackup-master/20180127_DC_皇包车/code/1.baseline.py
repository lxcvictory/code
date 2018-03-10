#coding:utf-8

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score
from gensim.models import word2vec


train = pd.read_csv('../data/data/label_train_drop.csv')
test = pd.read_csv('../data/data/label_test_drop.csv')
print(list(train.columns))
train.drop(["['商品零售价格指数(上年同月=100)_2017年11月'", " '国内生产总值(亿元)_2015年'", " '国内生产总值(亿元)_2014年'", " '年末总人口(万人)_2015年'", " '年末总人口(万人)_2014年'", " '工业生产者出厂价格指数(上年同月=100)_2017年11月'", " '居民消费价格指数(上年同月=100)_2017年11月'", " '社会商品零售总额(亿元)_2015年'", " '社会商品零售总额(亿元)_2014年']"],inplace=True,axis=1)
test.drop(["['商品零售价格指数(上年同月=100)_2017年11月'", " '国内生产总值(亿元)_2015年'", " '国内生产总值(亿元)_2014年'", " '年末总人口(万人)_2015年'", " '年末总人口(万人)_2014年'", " '工业生产者出厂价格指数(上年同月=100)_2017年11月'", " '居民消费价格指数(上年同月=100)_2017年11月'", " '社会商品零售总额(亿元)_2015年'", " '社会商品零售总额(亿元)_2014年']"],inplace=True,axis=1)
# 用户独立
y_train = train['orderType']

sub_id = test['userid']
del train['orderType']

del train['userid']
del train['useridUse']
del test['userid']
del test['useridUse']

cv = KFold(n_splits=4,shuffle=True,random_state=42)
results = []
feature_import = pd.DataFrame()
sub_array = []
feature_import['col'] = list(train.columns)

train = train.values
test = test.values
y_train = y_train.values

import xgboost as xgb

for traincv, testcv in cv.split(train,y_train):
    # dtrain = xgb.DMatrix(train[traincv], label=y_train[traincv])
    # dval = xgb.DMatrix(train[testcv], label=y_train[testcv])
    #
    # params = {
    #     'learning_rate': 0.01,
    #     'n_estimators': 1000,
    #     'max_depth': 10,
    #     'min_child_weight': 5,
    #     'gamma': 0,
    #     'colsample_bytree': 0.8,
    #     'eta': 0.05,
    #     'silent': 1,
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    #     'scale_pos_weight': 1,
    #     # 'nthread': 16,
    # }
    #
    # watchlist = [(dtrain, 'train'), (dval, 'eval')]
    # model = xgb.train(params, dtrain, 4000, watchlist,verbose_eval=200, early_stopping_rounds=100)
    # y_t = model.predict(xgb.DMatrix(train[testcv]))
    # results.append(roc_auc_score(y_train[testcv],y_t))
    # sub_array.append(model.predict(xgb.DMatrix(test)))

    lgb_train = lgb.Dataset(train[traincv], y_train[traincv])
    lgb_eval = lgb.Dataset(train[testcv], y_train[testcv], reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 0
    }

    # params = {
    #     'learning_rate': 0.05,
    #     'metric': 'auc',
    #     'num_leaves': 60,
    #     'num_trees': 490,
    #     'min_sum_hessian_in_leaf': 0.2,
    #     'min_data_in_leaf': 70,
    #     'bagging_fraction': 0.5,
    #     'feature_fraction': 0.3,
    #     'lambda_l1': 0,
    #     'lambda_l2': 11.88,
    #     'num_threads': 4,
    #     'scale_pos_weight': 1,
    #     'application': 'binary',
    # }

    gbm = lgb.train(params, lgb_train, num_boost_round=4000, valid_sets=[lgb_train, lgb_eval],verbose_eval = 200,early_stopping_rounds= 200)

    y_t = gbm.predict(train[testcv], num_iteration=gbm.best_iteration)
    results.append(roc_auc_score(y_train[testcv],y_t))

    feature_import[roc_auc_score(y_train[testcv],y_t)] = list(gbm.feature_importance())

    sub_array.append(gbm.predict(test,num_iteration=gbm.best_iteration))

print("Results: " + str( np.array(results).mean()))

feature_import.to_csv('./feature_imp_oo.csv')

s = 0
for i in sub_array:
    s = s + i

s = s / 4

r = pd.DataFrame()
r['userid'] = list(sub_id.values)
r['orderType'] = s

r.to_csv('../result/result_20180208_1.csv' ,index=False)

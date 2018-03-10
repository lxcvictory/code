# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

train = pd.read_excel(u'../session2/训练_20180117.xlsx')
test = pd.read_excel(u'../session2/测试B_20180117.xlsx')
print train.shape
print test.shape
train = train.fillna(0)
test = test.fillna(0)

# le = LabelEncoder()
# le.fit(train['TOOL_ID'].values)
# train['TOOL_ID'] = le.transform(train['TOOL_ID'])
# test['TOOL_ID'] = le.transform(test['TOOL_ID'])
new_test1_col = test['ID']
feature_columns = [x for x in train.columns if x not in ['ID', 'Value'] and train[x].dtype != object]
X_train, y = train[feature_columns], train['Value']
X_test = test[feature_columns]
from scipy.stats import kendalltau
# kendalltau(a, b, initial_lexsort=None, nan_policy='omit')

# rf = RandomForestRegressor(n_estimators=200, n_jobs=3)
# rf.fit(X_train, y)
# # print(rf.feature_importances_)
#
# model = SelectFromModel(rf, prefit=True)
# X_train_new, X_test_new = model.transform(X_train), model.transform(X_test)
# rf.fit(X_train_new, y)
# y_predict = rf.predict(X_test_new)

# corr = {}
# for f in X_train.columns:
#     data = X_train[f]
#     corr[f] = kendalltau(data.values, y.values, initial_lexsort=None, nan_policy='omit')[0]

# feature = []
#
# for k, v in corr.items():
#     if abs(v) >= 0.15:
#         feature.append(k)
#
# print len(feature)
# print feature

X_train = train[feature_columns]
X_test = test[feature_columns]
#
model = xgb.XGBRegressor(n_estimators=125, learning_rate=0.08, gamma=0, subsample=0.75,
                         colsample_bytree=1, max_depth=7)

model.fit(X_train, y,eval_set=[(X_train,y)])
y_predict = model.predict(X_test)

with open('../result/zs_{}.csv'.format(time.strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
    for id, y in zip(test['ID'], y_predict):
        f.write('{},{}\n'.format(id, y))
#
#
# from sklearn.model_selection import KFold
#
# cv = KFold(n_splits=2,shuffle=True,random_state=42)
#
# results = []
# sub_array = []
# train = X_train.values
# y_train = y.values
# test1 = X_test.values
# from sklearn.metrics import mean_squared_error
# # model xgb _ cv
# for model in [xgbbb]:
#     for traincv, testcv in cv.split(train,y_train):
#         model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])],early_stopping_rounds=150)
#         y_tmp = model.predict(train[testcv],ntree_limit=model.best_iteration)
#
#         res = mean_squared_error(y_train[testcv],y_tmp)
#         results.append(res)
#
#         sub_array.append(model.predict(test1,ntree_limit=model.best_iteration))
#     print("Results: " + str( np.array(results).mean() ))
#
#
# s = 0
# for i in sub_array:
#     s = s + i
#
# r = pd.DataFrame()
# r['ID'] = list(new_test1_col.values)
# r['Y'] = list(s/2)
#
# print(r[['ID', 'Y']])
#
# r[['ID', 'Y']].to_csv('../result/result_20180128.csv',index=None,header=None)

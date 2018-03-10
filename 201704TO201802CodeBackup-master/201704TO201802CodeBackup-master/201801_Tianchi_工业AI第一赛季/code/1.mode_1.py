#coding:utf-8

import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse

fileDir = '../data/'
train = pd.read_excel(fileDir + u"训练.xlsx")
test1 = pd.read_excel(fileDir + u"测试A.xlsx")
# test2 = pd.read_excel(fileDir + u"测试B.xlsx")
org_col_num = train.shape[1]
new_col = []
same_col = []
for i in train.columns:
    miss_val = train[i].isnull().sum()
    if miss_val < 200:
        if train[i].dtypes != 'object':
            train_values = train[i].dropna().unique()
            if (np.std(train_values) != 0) and (len(train_values) > 1 ):
                if min(train_values) < 9000:
                    # print pearsonr(train[i].values,train['Y'].values)

                    tmp_corr = np.corrcoef(train[i].values,train['Y'].values)[0][1]
                        # print tmp_corr
                    if str(tmp_corr) != 'nan':
                        # print tmp_corr
                        if abs(float(tmp_corr)) >= 0.05:
                            new_col.append(i)
        else:
            new_col.append(i)

print "清洗数据",org_col_num - len(new_col)

new_train = train[list(new_col)]
new_train_Y = new_train.pop('Y')


new_test1 = test1[new_train.columns]

new_test1_col = new_test1.pop('ID')
new_train_col = new_train.pop('ID')


import lightgbm as lgb
from sklearn.model_selection import train_test_split

for col in new_train.columns:
    if new_train[col].dtype == object:
        del new_train[col]
        del new_test1[col]
        # new_train[col] = new_train[col].astype('category')
        # new_test1[col] = new_test1[col].astype('category')

# num_folds = 5
# X_train_folds = np.array_split(new_train, num_folds)
# y_train_folds = np.array_split(new_train_Y, num_folds)
result = pd.DataFrame()
val_score = []
# for index,i in enumerate([0.2,0.3,0.4,0.5,0.6]):
#     X_tr, X_val, y_tr, y_val = train_test_split(new_train, new_train_Y,random_state=2017,test_size=i)
#
#     lgb_train  = lgb.Dataset(X_tr, y_tr)
#     lgb_eval  = lgb.Dataset(X_val, y_val,reference=lgb_train)

# params = {
#     # 'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': {'l2'},
#     'num_leaves': 16,
#     'learning_rate': 0.1,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
import xgboost as xgb

# xgb0 = xgb.XGBRegressor(
#     # silent=0,
#     max_depth=8,
#     learning_rate=0.05,
#     n_estimators=10000,
#     objective='reg:linear',
#     gamma=0,
#     min_child_weight=1,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     scale_pos_weight=1,
#     seed=20170105,
#     )
#
# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,
#                               learning_rate=0.05, n_estimators=1000,
#                               max_bin = 20, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# results = []
# sub_array = []
train = new_train.values
y_train = new_train_Y.values

#training xgboost
dtrain = xgb.DMatrix(train,label=y_train)
dtest = xgb.DMatrix(new_test1)

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':6,
	'lambda':100,
	'subsample':0.6,
	'colsample_bytree':0.6,
	'min_child_weight':5,#5~10
	'eta': 0.01,
	'sample_type':'uniform',
	'normalize':'tree',
	'rate_drop':0.1,
	'skip_drop':0.9,
	'seed':87,
	'nthread':12,
    'slice':0
	}

watchlist  = [(dtrain,'train')]

print 'cv'
# #通过cv找最佳的nround
cv_log = xgb.cv(params,dtrain,num_boost_round=1000,nfold=5,metrics='rmse',early_stopping_rounds=50,seed=1024)
bst_rmse= cv_log['test-rmse-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-rmse-mean']
bst_nb = cv_log.nb.to_dict()[bst_rmse]

# watchlist  = [(dtrain,'train')]
model = xgb.train(params,dtrain,num_boost_round=bst_nb+50,evals=watchlist)

#predict test set
test_y = model.predict(dtest)
print test_y
#




# print('Start training...')
    # train
    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 num_boost_round=200,
    #                 valid_sets=lgb_eval,
    #                 early_stopping_rounds=5)
    # y_p_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    # val_score.append(mse(y_val,y_p_val))
    # y_pred = gbm.predict(new_test1, num_iteration=gbm.best_iteration)
    #
    # result['Y%d'%(index)] = list(y_pred)



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# print '交叉验证'
# cv = KFold(n_splits=5,shuffle=True,random_state=42)

# results = []
# sub_array = []
# train = new_train.values
# y_train = new_train_Y.values

# model xgb _ cv
# for model in [model_lgb]:
#     for traincv, testcv in cv.split(train,y_train):
#         #     lgb_train  = lgb.Dataset(X_tr, y_tr)
#         #     lgb_eval  = lgb.Dataset(X_val, y_val,reference=lgb_train)
#         m = model.fit(train[traincv], y_train[traincv],eval_metric='mse',eval_set=[(train[testcv], y_train[testcv])],early_stopping_rounds=25)
#         y_tmp = m.predict(train[testcv],num_iteration=m.best_iteration)
#         res = mean_squared_error(y_train[testcv],y_tmp)
#         results.append(res)
#         # test_x
#         sub_array.append(m.predict(new_test1.values,num_iteration=m.best_iteration))
#     print("Results: " + str( np.array(results).mean() ))
#
# print(np.array(sub_array))
# s = 0
# for i in sub_array:
#     s = s + i
#
# r = pd.DataFrame()
# r['ID'] = list(new_test1_col.values)
# r['Y'] = list(s/5)
# print(r[['ID', 'Y']])
#
# r[['ID', 'Y']].to_csv('../result/result_20180108.csv',index=None,header=None)


# print np.mean(val_score)
#
# result['ID'] = list(new_test1_col.values)
#
# result['Y'] = (result['Y0'] + result['Y1'] + result['Y2'] + result['Y3'] + result['Y4']) / 5
#
# # result[['ID','Y']].to_csv('../result/result_20171226_lgb.csv',index=False,header=False)
# print result
#
# #







# # 处理NULL
# train_null_col_count = train.isnull().sum().reset_index()
# train_null_col_count.columns = ['index','null_count']
# null_col_list = train_null_col_count[train_null_col_count['null_count']<250]['index'].unique()
# # 198 null
# null_col_list = list(null_col_list)
# train = train[null_col_list]






# print sorted(list(train.isnull().sum()))
# 一起处理数据，降低数据维度
# train_corr = train.corr()

# print sorted(list(train_corr.Y.unique()))
# 查看Y的分布
# col_name = train.columns
# object_col_name = []

# for index,col_type in enumerate(train.dtypes):
#     if col_type == 'object':
#         print col_type,col_name[index]
#         object_col_name.append(col_name[index])
#
# object_columns_train = train[object_col_name+['Y']]
# object_columns_train_feat = obj_encoder.fit_transform(train[object_col_name[1:]].values)
# print object_columns_train_feat
# object_columns_train.to_csv('../tmp/object_col_train.csv',index=False)


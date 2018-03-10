#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mape_object(y,d):

    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h

# 评价函数
def mape(y,d):
    c=d.get_label()
    result= - np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

# 评价函数ln形式
def mape_ln(y,d):
    c=d.get_label()
    result= - np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)
    return "mape",result

def AddBaseTimeFeature(df):

    #添加column time_interval_begin, 删除data和time_interval
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date', 'time_interval'], axis=1)
    
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    return df

# gy_contest_link_top.txt
# link_top = pd.read_table('./gy_contest_link_top.txt',sep=';')
# print link_top
# 4377906284594800514                                4377906284514600514

'''
# txt => csv
link_info = pd.read_table('./gy_contest_link_info.txt',sep=';')
link_info = link_info.sort_values('link_ID')
training_data = pd.read_table('./gy_contest_link_traveltime_training_data.txt',sep=';')
print(training_data.shape)
training_data = pd.merge(training_data,link_info,on='link_ID')

testing_data = pd.read_table('./sub_demo.txt',sep='#',header=None)
testing_data.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
testing_data = pd.merge(testing_data,link_info,on='link_ID')
testing_data['travel_time'] = np.NaN
print(testing_data.shape)

#将3、4、5和6月6-8点的数据融合在一起
feature_date = pd.concat([training_data,testing_data],axis=0)
#将feature_date 根据link_ID和time_interval排序
feature_date = feature_date.sort_values(['link_ID','time_interval'])
feature_date.to_csv('./pre_data/feature_data.csv',index=False)
#构造date feature
feature_data = pd.read_csv('./pre_data/feature_data.csv')
feature_data_date = AddBaseTimeFeature(feature_data)
feature_data_date.to_csv('./pre_data/feature_data.csv',index=False)

'''
'''
# test
feature_data = pd.read_csv('./pre_data/feature_data.csv')
test = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
test.to_csv('./pre_data/test.csv',index=False)
'''
from scipy.stats import mode
# 中位数
def mode_function(df):
    counts = mode(df)
    return counts[0][0]

#读取的数据，会自动生成int数据类型
# feature_data = pd.read_csv('./pre_data/feature_data.csv')
# feature_data['link_ID'] = feature_data['link_ID'].astype(str)
# week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
# feature_data.drop(['time_interval_week','link_class'],inplace=True,axis=1)
# feature_data = pd.concat([feature_data,week],axis=1)
# print(feature_data.head())



# train = pd.DataFrame([])
# # train_label = pd.DataFrame([])
# #计算4月份 每个小时的前一个小时的统计量
# for curHour in range(1,24):
#     trainTmp = feature_data.loc[(feature_data.time_interval_month == 4)&
#                          (feature_data.time_interval_begin_hour==curHour),:]
#     for i in [58,48,38,28,18,8,0]:
#         tmp = feature_data.loc[(feature_data.time_interval_month == 4)
#               &(feature_data.time_interval_begin_hour == (curHour-1))&(feature_data.time_interval_minutes >= i),:]
#         #根据link_ID、time_interval_day来索引 分组，显示travel_time列，然后分组计算 mean,median等
#         #mean: 均值， median: 中位数， mode: 众数
#         tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#                 'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                 ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#         trainTmp = pd.merge(trainTmp,tmp,on=['link_ID','time_interval_day'],how='left')
#     # trainTmp_label = np.log1p(trainTmp.pop('travel_time'))
#     trainTmp.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
#     #每个小时的数据汇总
#     train = pd.concat([train, trainTmp], axis=0)
#     # train_label = pd.concat([train_label, trainTmp_label], axis=0)
#     print("curHour",curHour , "trainTmp.shape",trainTmp.shape, "train.shape", train.shape)
# train.to_csv('./pre_data/Apir_feat.csv',index=False)
#train.shape: (2296720, 38)


# train3 = pd.DataFrame([])
# # train3_label = pd.DataFrame([])
# #计算3月份 每个小时的前一个小时的统计量
# for curHour in range(1,24):
#     trainTmp = feature_data.loc[(feature_data.time_interval_month == 3)&
#                          (feature_data.time_interval_begin_hour==curHour),:]
#     for i in [58,48,38,28,18,8,0]:
#         tmp = feature_data.loc[(feature_data.time_interval_month == 3)
#               &(feature_data.time_interval_begin_hour == (curHour-1))&(feature_data.time_interval_minutes >= i),:]
#         #根据link_ID、time_interval_day来索引 分组，显示travel_time列，然后分组计算 mean,median等
#         #mean: 均值， median: 中位数， mode: 众数
#         tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#                 'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                 ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#         trainTmp = pd.merge(trainTmp,tmp,on=['link_ID','time_interval_day'],how='left')
#     # trainTmp_label = np.log1p(trainTmp.pop('travel_time'))
#     trainTmp.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
#     #每个小时的数据汇总
#     train3 = pd.concat([train3, trainTmp], axis=0)
#     # train3_label = pd.concat([train3_label, trainTmp_label], axis=0)
#     print("curHour ",curHour , "trainTmp.shape",trainTmp.shape, "train3.shape", train3.shape)
# train3.to_csv('./pre_data/Mrith_feat.csv', index=False)
#train3.shape : (2401035, 38)

#
# train3 = pd.read_csv('./pre_data/Mrith_feat.csv')
# train4 = pd.read_csv('./pre_data/feature_data.csv')
# train = pd.concat([train3, train4], axis =0)
# print train
# train_label = pd.concat([train_label, train3_label], axis=0)
#train.shape: (4697755, 38)
#
#
# train5 = pd.DataFrame([])
# # train5_label = pd.DataFrame([])
# #计算5月份 每个小时的前一个小时的统计量
# for curHour in [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
#     trainTmp = feature_data.loc[(feature_data.time_interval_month == 5)&
#                          (feature_data.time_interval_begin_hour==curHour),:]
#     for i in [58,48,38,28,18,8,0]:
#         tmp = feature_data.loc[(feature_data.time_interval_month == 5)
#               &(feature_data.time_interval_begin_hour == (curHour-1))&(feature_data.time_interval_minutes >= i),:]
#         #根据link_ID、time_interval_day来索引 分组，显示travel_time列，然后分组计算 mean,median等
#         #mean: 均值， median: 中位数， mode: 众数
#         tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#                 'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#         trainTmp = pd.merge(trainTmp,tmp,on=['link_ID','time_interval_day'],how='left')
#
#     # trainTmp_label = np.log1p(trainTmp.pop('travel_time'))
#     trainTmp.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
#
#     #每个小时的数据汇总
#     train5 = pd.concat([train5, trainTmp], axis=0)
#     # train5_label = pd.concat([train5_label, trainTmp_label], axis=0)
#     print("curHour ",curHour , "trainTmp.shape",trainTmp.shape, "train5.shape", train5.shape)
# train5.to_csv('./pre_data/May_feat.csv', index=False)
#train5.shape :  (2348352, 38)
#
#
# train3 = pd.read_csv('./pre_data/Mrith_feat.csv')
# train4 = pd.read_csv('./pre_data/Apir_feat.csv')
# train5 = pd.read_csv('./pre_data/May_feat.csv')
# train = pd.concat([train3, train4, train5], axis =0)
# print train
# train_label = pd.concat([train_label, train5_label], axis=0)
# #train.shape: (7046107, 38)
#
#
#
#计算5月份，每天 7点i分-8点之间 所有link的 一些统计量。
# test = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour==8),: ]
# for i in [58,48,38,28,18,8,0]:
#     tmp = feature_data.loc[(feature_data.time_interval_month == 5)&(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
#     tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#             'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                 ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#     test = pd.merge(test,tmp,on=['link_ID','time_interval_day'],how='left')
#
# # test_label = np.log1p(test.pop('travel_time'))
# test.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
# test.to_csv('./pre_data/test_feat.csv', index=False)
#
#
#计算6月份，每天 7点i分-8点之间 所有link的 一些统计量。
# submit = feature_data.loc[(feature_data.time_interval_month == 6)&(feature_data.time_interval_begin_hour==8),: ]
# for i in [58,48,38,28,18,8,0]:
#     tmp = feature_data.loc[(feature_data.time_interval_month == 6)
#           &(feature_data.time_interval_begin_hour == 7)&(feature_data.time_interval_minutes >= i),:]
#     tmp = tmp.groupby(['link_ID', 'time_interval_day'])[
#             'travel_time'].agg([('mean_%d' % (i), np.mean), ('median_%d' % (i), np.median),
#                                 ('mode_%d' % (i), mode_function), ('std_%d' % (i), np.std), ('max_%d' % (i), np.max),('min_%d' % (i), np.min)]).reset_index()
#     submit = pd.merge(submit,tmp,on=['link_ID','time_interval_day'],how='left')
#
# submit.pop('travel_time')
# submit.drop(['time_interval_begin_hour','time_interval_month','time_interval_begin'],inplace=True,axis=1)
# submit.to_csv('./pre_data/submit_feat.csv', index=False)
#submit_label =
#
#
# #link_ID -> one_hot
# # train = pd.get_dummies(train)
# # test = pd.get_dummies(test)
# # submit = pd.get_dummies(submit)
#
#

# train3 = pd.read_csv('./pre_data/Mrith_feat.csv')
# train3['link_ID'] = train3['link_ID'].astype(str)
# train3 = train3.drop_duplicates(train3.columns)
# train4 = pd.read_csv('./pre_data/Apir_feat.csv')
# train4['link_ID'] = train4['link_ID'].astype(str)
# train4 = train4.drop_duplicates(train4.columns)
# train3_4 = pd.concat([train3, train4], axis =0)
# train3_4.to_csv('./pre_data/train_3_4.csv',index=False)
#
# train5 = pd.read_csv('./pre_data/May_feat.csv')
# train5['link_ID'] = train5['link_ID'].astype(str)
# train5 = train5.drop_duplicates(train5.columns)
# train3_4 = pd.read_csv('./pre_data/train_3_4.csv')
# train3_4['link_ID'] = train3_4['link_ID'].astype(str)

# train = pd.concat([train3, train4, train5], axis =0)
# print train
# train_label = pd.concat([train_label, train5_label], axis=0)
#train.shape: (7046107, 38)

train = pd.read_csv('./pre_data/train_3_4.csv')
train['link_ID'] = train['link_ID'].astype(str)
train_label = np.log1p(train.pop('travel_time'))

test = pd.read_csv('./pre_data/test_feat.csv')
test['link_ID'] = test['link_ID'].astype(str)
test_label = np.log1p(test.pop('travel_time'))




import xgboost as xgb

xlf = xgb.XGBRegressor(max_depth=11,
                       learning_rate=0.01,
                       n_estimators=1000,
                       silent=True,
                       objective=mape_object,
                       gamma=0,
                       min_child_weight=5,
                       max_delta_step=0,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       colsample_bylevel=1,
                       reg_alpha=1e0,
                       reg_lambda=0,
                       scale_pos_weight=1,
                       seed=9,
                       missing=None)


xlf.fit(train.values, train_label.values, eval_metric=mape_ln,
        verbose=True, eval_set=[(test.values, test_label.values)],
        early_stopping_rounds=5)
print(xlf.get_params())


submit = pd.read_csv('./pre_data/submit_feat.csv')
submit['link_ID'] = submit['link_ID'].astype(str)

#sub_data: 构造sub_demo.txt
submit_label = xlf.predict(submit.values,  ntree_limit= xlf.best_ntree_limit)


result = xlf.predict(submit.values)

travel_time = pd.DataFrame({'travel_time':list(result)})
sub_demo = pd.read_table('./sub_demo.txt',header=None,sep='#')
sub_demo.columns = ['link_ID','date','time_interval','travel_time']
del sub_demo['travel_time']
tt = pd.concat([sub_demo,travel_time],axis=1)
# tt = tt.fillna(0)
tt['travel_time'] = np.round(np.expm1(tt['travel_time']),6)
tt[['link_ID','date','time_interval','travel_time']].to_csv('./mapodoufu_2017-08-08.txt',sep='#',index=False,header=False)


# rate 0.01, n 1000: 0.361603
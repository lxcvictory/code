#coding:utf-8
# TO DO 2018 02 08
# 对于piupiu代码的阅读和笔记

# 导入需要的资源包
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder

# 导入数据
data_path = '../data/'

air_reserve = pd.read_csv(data_path + 'air_reserve.csv').rename(columns={'air_store_id':'store_id'})
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv').rename(columns={'hpg_store_id':'store_id'})
air_store = pd.read_csv(data_path + 'air_store_info.csv').rename(columns={'air_store_id':'store_id'})
hpg_store = pd.read_csv(data_path + 'hpg_store_info.csv').rename(columns={'hpg_store_id':'store_id'})
air_visit = pd.read_csv(data_path + 'air_visit_data.csv').rename(columns={'air_store_id':'store_id'})
store_id_map = pd.read_csv(data_path + 'store_id_relation.csv').set_index('hpg_store_id',drop=False)
date_info = pd.read_csv(data_path + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'}).drop('day_of_week',axis=1)
submission = pd.read_csv(data_path + 'sample_submission.csv')

# 对数据的格式进行操作
# 对submission数据的id进行分割，id=》store_id , visit_date 格式
submission['visit_date'] = submission['id'].apply(lambda x:str(x)[21:])
submission['store_id'] = submission['id'].apply(lambda x:str(x)[:20])

# 星期信息是因为题目本身提供了星期信息，以及EDA的分析（大概猜测）
# visit_datetime , reserve_datetime 获取年月日,以及访问时间的日期
air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]
air_reserve['dow'] = pd.to_datetime(air_reserve['visit_date']).dt.dayofweek
# 对 hpg的信息采取相同的处理方式
hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].str[:10]
hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].str[:10]
hpg_reserve['dow'] = pd.to_datetime(hpg_reserve['visit_date']).dt.dayofweek

# air_visit 构造与目标 submission 类似的id结构
air_visit['id'] = air_visit['store_id'] + '_'  + air_visit['visit_date']

# 题目要求是air_reserve的store商店，需要对hgp进行映射,没有丢掉hpg的信息
hpg_reserve['store_id'] = hpg_reserve['store_id'].map(store_id_map['air_store_id']).fillna(hpg_reserve['store_id'])
hpg_store['store_id'] = hpg_store['store_id'].map(store_id_map['air_store_id']).fillna(hpg_reserve['store_id'])
hpg_store.rename(columns={'hpg_genre_name':'air_genre_name','hpg_area_name':'air_area_name'},inplace=True)

# 对于节假日的标记 周6 7 和本身的节假日
date_info['holiday_flg2'] = pd.to_datetime(date_info['visit_date']).dt.dayofweek
date_info['holiday_flg2'] = ((date_info['holiday_flg2']>4) | (date_info['holiday_flg']==1)).astype(int)

# 组合训练数据和测试数据之后一起处理
data = pd.concat([air_visit,submission]).copy()
data['dow'] = pd.to_datetime(data['visit_date']).dt.dayofweek

# 对地区进行编码操作
# 对经营类别进行编码
air_store['air_area_name0'] = air_store['air_area_name'].apply(lambda x: x.split(' ')[0])
lbl = LabelEncoder()
air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])
air_store['air_area_name0'] = lbl.fit_transform(air_store['air_area_name0'])

# 题目的评测函数是对  log 进行评测
data['visitors'] = np.log1p(data['visitors'])

# 数据合并,包括日期数据，商店信息
data = pd.merge(data,air_store,on=['store_id'],how='left')
data = data.merge(date_info[['visit_date','holiday_flg','holiday_flg2']], on=['visit_date'],how='left')

# 时间推移
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days

def get_label(end_date,n_day):
    # 39天
    label_end_date = date_add_days(end_date, n_day)
    # 截取时间段
    print('标签数据的时间开始 >=%s <%s 共%s天'%(end_date,label_end_date,n_day))

    label = data[(data['visit_date'] < label_end_date) & (data['visit_date'] >= end_date)].copy()
    label['end_date'] = end_date
    label['diff_of_day'] = label['visit_date'].apply(lambda x: diff_of_days(x,end_date))
    label['month'] = label['visit_date'].str[5:7].astype(int)
    label['year'] = label['visit_date'].str[:4].astype(int)

    for i in [3,2,1,-1]:
        date_info_temp = date_info.copy()
        date_info_temp['visit_date'] = date_info_temp['visit_date'].apply(lambda x: date_add_days(x,i))
        # 上下x天的周末标记
        date_info_temp.rename(columns={'holiday_flg':'ahead_holiday_{}'.format(i),'holiday_flg2':'ahead_holiday2_{}'.format(i)},inplace=True)
        label = label.merge(date_info_temp, on=['visit_date'],how='left')
    label = label.reset_index(drop=True)
    return label

# 自定义的一个数据拼接操作
# list列表中的datafram数据进行拼接
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result

# 自定义的一个左连接操作
# 连接到标签数据，保留data2的标签
def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    #     这个干啥用的
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    # label merge data 获取data的columns
    columns = [f for f in data2.columns if f not in on]
    # 标签为主
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result

# 制作特征
def get_store_visitor_feat(label, key, n_day):
    start_date = date_add_days(key[0],-n_day)
    print('特征区间>%s<%s,共%s天'%(start_date , key[0], n_day))
    data_temp = data[(data.visit_date < key[0]) & (data.visit_date > start_date)].copy()
    # 统计特征
    result = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({'store_min{}'.format(n_day): 'min',
                                                                             'store_mean{}'.format(n_day): 'mean',
                                                                             'store_median{}'.format(n_day): 'median',
                                                                             'store_max{}'.format(n_day): 'max',
                                                                             'store_count{}'.format(n_day): 'count',
                                                                             'store_std{}'.format(n_day): 'std',
                                                                             'store_skew{}'.format(n_day): 'skew'})

    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


# 制作训练集
def make_feats(end_date, n_day):
    # t0 = time.time()
    key = end_date, n_day
    print('data key为：{}'.format(key))
    print('add label')
    label = get_label(end_date, n_day)
    print('make feature...')
    result = []
    result.append(get_store_visitor_feat(label, key, 1000))  # store features
    result.append(label)

    result = concat(result)
    return result

import datetime
import lightgbm as lgb

train_feat = pd.DataFrame()
start_date = '2017-03-12'
# 类似于窗口数据
# 标签数据
# 2017-03-12 - 2017-04-20 提取39天的标签 （左闭右开） 每次前移动一个星期
# 特征数据
# x天 - 2017-03-12（开区间）
#
for i in range(58):
    train_feat_sub = make_feats(date_add_days(start_date, i*(-7)),39)
    # train_feat = pd.concat([train_feat,train_feat_sub])
    # print(train_feat.columns)
    # train_feat = pd.concat([train_feat,train_feat_sub])
for i in range(1,6):
    train_feat_sub = make_feats(date_add_days(start_date,i*(7)),42-(i*7))
    # train_feat = pd.concat([train_feat,train_feat_sub])
# test_feat = make_feats(date_add_days(start_date, 42),39)
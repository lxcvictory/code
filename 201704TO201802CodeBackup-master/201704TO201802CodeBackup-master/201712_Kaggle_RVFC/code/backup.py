#coding:utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import Geohash

# ps: 周围为星期
# read data from data
data = {
    'tra':
    pd.read_csv('../data/air_visit_data.csv',parse_dates=['visit_date']),

    # 基础信息
    'as':
    pd.read_csv('../data/air_store_info.csv'),

    'hs':
    pd.read_csv('../data/hpg_store_info.csv'),

    # 记录信息，到店时间 和 预约时间
    'ar':
    pd.read_csv('../data/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime']),
    'hr':
    pd.read_csv('../data/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime']),
    # id 映射 air - hgp

    'id':
    pd.read_csv('../data/store_id_relation.csv'),
    'tes':
    pd.read_csv('../data/sample_submission.csv'),

    # 日期信息
    'hol':
    pd.read_csv('../data/date_info.csv',parse_dates=['calendar_date']).rename(columns={
        'calendar_date': 'visit_date'
    })
}

# 根据id映射hr到air
data['hr'] = data['hr'].merge(data['id'],on='hpg_store_id',how='inner')

print data['tra'].shape
print data['tes'].shape

# 月映射
def trp_(m):
    m = int(m)
    if m <= 10:
        return 1
    elif m >= 20:
        return 2
    else:
        return 3

# train 时间数据
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['quarter'] = data['tra']['visit_date'].dt.quarter
data['tra']['weekofyear'] = data['tra']['visit_date'].dt.weekofyear
data['tra']['day_trp'] = data['tra']['visit_date'].dt.day
data['tra']['day_trp'] = data['tra']['day_trp'].map(trp_)
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

# test 时间数据 / 格式整理
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['quarter'] = data['tes']['visit_date'].dt.quarter
data['tes']['weekofyear'] = data['tes']['visit_date'].dt.weekofyear
data['tes']['day_trp'] = data['tes']['visit_date'].dt.day
data['tes']['day_trp'] = data['tes']['day_trp'].map(trp_)
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

from scipy.stats import mode
def mode_np(df):
    return mode(df)[0][0]

# 非时间窗口统计数据 - 过滤200一下的数据
data['tra'] = data['tra'][data['tra']['visitors'] <= 200]

# 历史数据统计
static_data_air = data['tra'].groupby(['air_store_id'],as_index=False).visitors.sum()
static_data_air.rename(columns={'visitors':'histoy_visitors'},inplace=True)

# 星期统计数据
static_data = data['tra'].groupby(['air_store_id','dow'],as_index=False).visitors.agg(
    {'min_visitors':np.min,'mean_visitors':np.mean,'median_visitors':np.median,
     'max_visitors':np.max,'count_observations':np.sum,'mode_visitors':mode_np,
     'std_visitors':np.nanstd}
)

static_data['seven_visitors'] = static_data['count_observations'] / 7.0
# static_data['half_visitors'] = static_data['count_observations'] / 3.0
# static_data['ratio_visitors'] = static_data['min_visitors'] / static_data['max_visitors'] + static_data['median_visitors']

static_data_month = data['tra'].groupby(['air_store_id','day_trp'],as_index=False).visitors.agg(
    {'min_t_visitors':np.min,'mean_t_visitors':np.mean,
     'median_t_visitors':np.median,'max_t_visitors':np.max,
     'count_t_observations':np.sum,'mode_t_visitors':mode_np,
     'std_t_visitors':np.nanstd}
)

# 基本信息数据
stores = pd.merge(static_data, data['as'], how='left', on=['air_store_id'])
stores = pd.merge(stores,static_data_air,how='left',on=['air_store_id'])

# 查看类别统计
static_data_kind = stores.groupby(['air_genre_name'],as_index=False)['count_observations'].sum()
static_data_kind.rename(columns={'count_observations':'kind_visitors_sum'},inplace=True)
stores = pd.merge(stores,static_data_kind,how='left',on=['air_genre_name'])

static_data_air_area_name = stores.groupby(['air_area_name'],as_index=False)['count_observations'].sum()
static_data_air_area_name.rename(columns={'count_observations':'air_area_name_visitors_sum'},inplace=True)
stores = pd.merge(stores,static_data_air_area_name,how='left',on=['air_area_name'])

# 字段处理 -- air_genre_name air_area_name
# 经营种类
stores['air_genre_name_kinds'] = stores['air_genre_name'].map(lambda x: len(str(x).split('/')))

lbe = preprocessing.LabelEncoder()
# 通过分析 发现长度为 2
for i in range(2):
    stores['air_genre_name_' + str(i)] = stores['air_genre_name'].map(lambda x: str(str(x).split('/')[i]) if len(str(x).split('/'))>i else '')
    stores['air_genre_name_' + str(i)] = lbe.fit_transform(stores['air_genre_name_' + str(i)])

for i in range(3):
    stores['air_area_name_' + str(i)] = stores['air_area_name'].map(
        lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else '')
    stores['air_area_name_' + str(i)] = lbe.fit_transform(stores['air_area_name_' + str(i)])

stores['air_genre_name'] = lbe.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbe.fit_transform(stores['air_area_name'])

del stores['air_area_name_0']
#  geohash 替代聚类处理
import mzgeohash

train_loc = zip(stores['longitude'],stores['latitude'])
# test_loc = zip(test['longitude'],test['latitude'])
list_store_code_1 = []
list_store_code_2 = []
# list_test_code = []

for i in train_loc:
    list_store_code_1.append(mzgeohash.encode(i,length=6))
    list_store_code_2.append(mzgeohash.encode(i,length=8))

lbe = preprocessing.LabelEncoder()
stores['geohash'] = list_store_code_1
stores['geohash'] = lbe.fit_transform(stores['geohash'])

stores['geohash_sm'] = list_store_code_2
stores['geohash_sm'] = lbe.fit_transform(stores['geohash_sm'])

stores_count = stores.groupby(['geohash'],as_index=False).count_observations.sum()
stores_count.rename(columns={'count_observations':'geohash_visitors_sum'},inplace=True)
stores = pd.merge(stores,stores_count,how='left',on=['geohash'])

stores_count_sm = stores.groupby(['geohash_sm'],as_index=False).count_observations.sum()
stores_count_sm.rename(columns={'count_observations':'geohash_visitors_sum_sm'},inplace=True)
stores = pd.merge(stores,stores_count_sm,how='left',on=['geohash_sm'])

# ratio features 比例特征
stores['count_observations/histoy_visitors'] = stores['count_observations'] / stores['histoy_visitors']

stores['count_observations/kind_visitors_sum'] = stores['count_observations'] / stores['kind_visitors_sum']
stores['histoy_visitors/kind_visitors_sum'] = stores['histoy_visitors'] / stores['kind_visitors_sum']

stores['histoy_visitors/air_area_name_visitors_sum'] = stores['histoy_visitors'] / stores['air_area_name_visitors_sum']
stores['count_observations/air_area_name_visitors_sum'] = stores['count_observations'] / stores['air_area_name_visitors_sum']
stores['kind_visitors_sum/air_area_name_visitors_sum'] = stores['kind_visitors_sum'] / stores['air_area_name_visitors_sum']

stores['kind_visitors_sum/geohash_visitors_sum'] = stores['kind_visitors_sum'] / stores['geohash_visitors_sum']
stores['count_observations/geohash_visitors_sum'] = stores['count_observations'] / stores['geohash_visitors_sum']
stores['histoy_visitors/geohash_visitors_sum'] = stores['histoy_visitors'] / stores['geohash_visitors_sum']
stores['air_area_name_visitors_sum/geohash_visitors_sum'] = stores['air_area_name_visitors_sum'] / stores['geohash_visitors_sum']

stores['kind_visitors_sum/geohash_visitors_sum_sm'] = stores['kind_visitors_sum'] / stores['geohash_visitors_sum_sm']
stores['count_observations/geohash_visitors_sum_sm'] = stores['count_observations'] / stores['geohash_visitors_sum_sm']
stores['histoy_visitors/geohash_visitors_sum_sm'] = stores['histoy_visitors'] / stores['geohash_visitors_sum_sm']
stores['air_area_name_visitors_sum/geohash_visitors_sum_sm'] = stores['air_area_name_visitors_sum'] / stores['geohash_visitors_sum_sm']
stores['geohash_visitors_sum/geohash_visitors_sum_sm'] = stores['geohash_visitors_sum'] / stores['geohash_visitors_sum_sm']

# 归一化
stores['kind_visitors_sum'] = stores['kind_visitors_sum'] / (stores['kind_visitors_sum'].max() - stores['kind_visitors_sum'].min())
stores['count_observations'] = stores['count_observations'] / (stores['count_observations'].max() - stores['count_observations'].min())
stores['histoy_visitors'] = stores['histoy_visitors'] / (stores['histoy_visitors'].max() - stores['histoy_visitors'].min())
stores['geohash_visitors_sum'] = stores['geohash_visitors_sum'] / (stores['geohash_visitors_sum'].max() - stores['geohash_visitors_sum'].min())
stores['geohash_visitors_sum_sm'] = stores['geohash_visitors_sum_sm'] / (stores['geohash_visitors_sum_sm'].max() - stores['geohash_visitors_sum_sm'].min())

print stores

# stores.drop(['kind_visitors_sum','histoy_visitors','air_area_name_visitors_sum'],axis=1,inplace=True)

# 合并数据
train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

# 节假日标记1 周末标记2
wkend = data['hol'].apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday')), axis=1)
data['hol'].loc[wkend, 'holiday_flg'] = 1
wkend_holidays = data['hol'].apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
data['hol'].loc[wkend_holidays, 'holiday_flg'] = 2

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

hol_info = data['hol'][['visit_date','holiday_flg']]

train = pd.merge(train, hol_info, how='left', on=['visit_date'])
test = pd.merge(test, hol_info, how='left', on=['visit_date'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(
        ['air_store_id', 'visit_datetime'], as_index=False)[[
            'reserve_datetime_diff', 'reserve_visitors'
        ]].sum().rename(columns={
            'visit_datetime': 'visit_date'
        })

for df in ['ar', 'hr']:
    train = pd.merge(
        train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(
        test, data[df], how='left', on=['air_store_id', 'visit_date'])

#
train['reserve_visitors_x'] = train['reserve_visitors_x'].fillna(0)
train['reserve_visitors_y'] = train['reserve_visitors_y'].fillna(0)

test['reserve_visitors_x'] = test['reserve_visitors_x'].fillna(0)
test['reserve_visitors_y'] = test['reserve_visitors_y'].fillna(0)

train['reserve_datetime_diff_x'] = train['reserve_datetime_diff_x'].fillna(0)
train['reserve_datetime_diff_y'] = train['reserve_datetime_diff_y'].fillna(0)

test['reserve_datetime_diff_x'] = test['reserve_datetime_diff_x'].fillna(0)
test['reserve_datetime_diff_y'] = test['reserve_datetime_diff_y'].fillna(0)


# 两组店铺的信息 //
train['total_reserv_sum'] = train['reserve_visitors_x'] + train['reserve_visitors_y']
train['total_reserv_mean'] = (train['reserve_visitors_x'] + train['reserve_visitors_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['reserve_datetime_diff_x'] + train['reserve_datetime_diff_y']) / 2
train['total_reserv_dt_diff_totle'] = train['reserve_datetime_diff_x'] + train['reserve_datetime_diff_y']

test['total_reserv_sum'] = test['reserve_visitors_x'] + test['reserve_visitors_y']
test['total_reserv_mean'] = (test['reserve_visitors_x'] + test['reserve_visitors_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['reserve_datetime_diff_x'] + test['reserve_datetime_diff_y']) / 2
test['total_reserv_dt_diff_totle'] = test['reserve_datetime_diff_x'] + test['reserve_datetime_diff_y']

train['var_max_lat'] = train['latitude'].max() - train['latitude']
# train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
# test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

train = train.fillna(-1)
test = test.fillna(-1)

print('Binding to float32')
# 可以提高成绩，集体原因不知道为啥
for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)

train = pd.merge(train,static_data_month,on=['air_store_id','day_trp'],how='left')
test = pd.merge(test,static_data_month,on=['air_store_id','day_trp'],how='left')

# 类别特征组合
train['month_day_trp'] = train['month'] * 100 + train['day_trp']
test['month_day_trp'] = test['month'] * 100 + test['day_trp']

train['month_geohash'] = train['month'] * 1000 + train['geohash']
test['month_geohash'] = test['month'] * 1000 + test['geohash']

train['month_geohash_sm'] = train['month'] * 1000 + train['geohash_sm']
test['month_geohash_sm'] = test['month'] * 1000 + test['geohash_sm']

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

train_x = train.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(train['visitors'].values)
print(train_x.shape, train_y.shape)

test_x = test.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)
print(test_x.shape)


#################################################################################
# print train_x.head()
# print test_x.head()

xgb0 = xgb.XGBRegressor(
    # silent=0,
    max_depth=8,
    learning_rate=0.05,
    n_estimators=10000,
    objective='reg:linear',
    gamma=0.01,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=1,
    seed=20170105,
    )


import lightgbm as lgb
gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=10000)

# [447]	validation_0-rmse:0.436691	validation_1-rmse:0.487573
# [439]	validation_0-rmse:0.437108	validation_1-rmse:0.486929
# [426]	validation_0-rmse:0.437595	validation_1-rmse:0.486803


###########################无cv##############################################
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
# # 尺度归一
# ss_x = preprocessing.StandardScaler()
# train_x_disorder = ss_x.fit_transform(X_train)
# test_x_disorder = ss_x.transform(X_test)
#
# ss_y = preprocessing.StandardScaler()
# train_y_disorder = ss_y.fit_transform(y_train.reshape(-1, 1))
# test_y_disorder = ss_y.transform(y_test.reshape(-1, 1))
#
#
# xgb0.fit(X_train, y_train,eval_metric='rmse',eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=25)
#
# gbm0.fit(X_train, y_train,eval_metric='rmse',eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=25)
#
# fe = pd.DataFrame()
# fe['name'] = list(train_x.columns)
# fe['fear'] = list(xgb0.feature_importances_)
# fe = fe.sort_values(['fear'])
#
# fe_gbm = pd.DataFrame()
# fe_gbm['name'] = list(train_x.columns)
# fe_gbm['fear'] = list(gbm0.feature_importances_)
# fe_gbm = fe_gbm.sort_values(['fear'])
#
#
# print fe
#
# print fe_gbm
#############################################################################


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print '交叉验证'
cv = KFold(n_splits=9,shuffle=True,random_state=42)

results = []
sub_array = []
train = train_x.values
y_train = train_y

# model xgb _ cv
# for model in [gbm0]: xgb0
# CV 模型融合 xgb+lgb
for traincv, testcv in cv.split(train,y_train):
    # m_0 = xgb0.fit(train[traincv], y_train[traincv],eval_metric='rmse',eval_set=[(train[testcv], y_train[testcv])],early_stopping_rounds=25)
    m_1 = gbm0.fit(train[traincv], y_train[traincv],eval_metric='rmse',eval_set=[(train[testcv], y_train[testcv])],early_stopping_rounds=25)
    # y_tmp_0 = m_0.predict(train[testcv],ntree_limit=m_0.best_ntree_limit)
    y_tmp_1 = m_1.predict(train[testcv],num_iteration=m_1.best_iteration)
    res = mean_squared_error(y_train[testcv],y_tmp_1) ** 0.5
    results.append(res)
    # test_x
    # sub_array.append((m_0.predict(test_x.values,ntree_limit=m_0.best_ntree_limit)))
    sub_array.append(m_1.predict(test_x.values,num_iteration=m_1.best_iteration))
print("Results: " + str( np.array(results).mean() ))

print(np.array(sub_array))
s = 0
for i in sub_array:
    s = s + i

r = pd.DataFrame()
test['visitors'] = np.expm1(list(s/9))
print(test[['id', 'visitors']])

test[['id', 'visitors']].to_csv('../result/20180114_1.csv', float_format='%.3f', index=None)

# 490 - 495
# Results: 0.488719904789 - 493
# Results: 0.486502398547 - 492
# Results: 0.484244150815 - 490
# 0.006

# lgb --
# Results: 0.481844787973 - 489
# Results: 0.48219531924 - 489
# Results: 0.478067897905 0.485
# Results: 0.477719275419 0.485








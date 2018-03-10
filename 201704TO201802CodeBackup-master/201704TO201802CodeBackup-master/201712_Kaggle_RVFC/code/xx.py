#coding:utf-8
import pandas as pd
from datetime import timedelta

dir = '../data/'

print('read date')

train = pd.read_csv(
    dir + 'air_visit_data.csv',
    parse_dates=['visit_date']
                    )
# train['visitors'] = pd.np.log1p(train['visitors'])

test = pd.read_csv(dir + 'sample_submission.csv')
# test['visitors'] = pd.np.log1p(test['visitors'])

test['air_store_id'] = test['id'].apply(lambda x:'_'.join(str(x).split('_')[:2]))
test['visit_date'] = test['id'].apply(lambda x:str(x).split('_')[2])

test = test[train.columns]

a = list(train['air_store_id'].unique())
b = list(test['air_store_id'].unique())

ret_list = list(set(a)^set(b))
print(ret_list)

train = train[~train['air_store_id'].isin(ret_list)]


from sklearn.preprocessing import LabelEncoder
dow = LabelEncoder()
hol = pd.read_csv(
    dir + 'date_info.csv'
)

hol['dow'] =  pd.to_datetime(hol['calendar_date']).dt.weekday + 1
hol['holiday_w'] = 0
wkend = hol.apply((lambda x: (x.dow == 6 or x.dow == 7 or x.dow == 5)), axis=1)
hol.loc[wkend, 'holiday_w'] = 1

del hol['day_of_week']

air_reserve = pd.read_csv(
    dir + 'air_reserve.csv',
    parse_dates=['reserve_datetime','visit_datetime']
)

#
air_reserve['visit_date'] = air_reserve['reserve_datetime'].dt.date
air_reserve['visit_reserve_diff'] = (air_reserve['visit_datetime'] - air_reserve['reserve_datetime']).dt.days
air_reserve['visit_date'] = pd.to_datetime(air_reserve['visit_date'])

# train = pd.merge(train,air_reserve,on=['visit_date','air_store_id'],how='left')
# train['reserve_visitors'] = pd.np.log1p(train['reserve_visitors'])
# train['today_r_v_diff'] = train['visitors'] - train['reserve_visitors']



sub_to_submit = pd.merge(test,hol,left_on='visit_date',right_on='calendar_date')
sub_to_submit = sub_to_submit[['air_store_id','visit_date','holiday_flg','dow','visitors','holiday_w']]

sub_to_submit['visit_date'] = pd.to_datetime(sub_to_submit['visit_date'])
sub_to_submit_air = list(sub_to_submit['air_store_id'].unique())

# train_air = list(train['air_store_id'].unique())


from sklearn.preprocessing import LabelEncoder
# air_store_id_label = LabelEncoder()
# air_store_id_label.fit(train_air + sub_to_submit_air)


# 构造提交数据的特征
def week_feat(train,sub_to_submit):
    for i in [35,21,14,7]:
        # 获取开始日期
        sub_begin_date = pd.to_datetime(sub_to_submit['visit_date'].min())
        print('特征日期',sub_begin_date - timedelta(days=i))

        tmp = train[train['visit_date'] >= (sub_begin_date - timedelta(days=i))]
        hol['calendar_date'] = pd.to_datetime(hol['calendar_date'])

        tmp = pd.merge(tmp,hol,left_on=['visit_date'],right_on=['calendar_date'],how='left')


        f_tmp_hol = tmp.groupby(['air_store_id','dow','holiday_w'],as_index=False).visitors.agg({
            'hol_std_dow_%d' % (i): pd.np.nanstd,
            'hol_median_dow_%d' % (i): pd.np.nanmedian,
            })

        f_tmp = tmp.groupby(['air_store_id', 'dow'], as_index=False).visitors.agg({
            'std_dow_%d' % (i): pd.np.nanstd,
            'median_dow_%d' % (i): pd.np.nanmedian,
            'max_dow_%d' % (i): pd.np.nanmax,
            'min_dow_%d' % (i): pd.np.nanmin,
        })

        f_day_tmp = tmp.groupby(['air_store_id'], as_index=False).visitors.agg({
            'std_day_%d' % (i): pd.np.nanstd,
            'median_day_%d' % (i): pd.np.nanmedian,
        })

        # f_reserve_visitors_tmp = tmp.groupby(['air_store_id'], as_index=False).reserve_visitors.agg({
        #     'mean_reserve_visitors_%d' % (i): pd.np.nanmean,
        #     'std_reserve_visitors_%d' % (i): pd.np.nanstd,
        #     'median_reserve_visitors_%d' % (i): pd.np.nanmedian,
        #     'max_reserve_visitors_%d' % (i): pd.np.nanmax,
        #     'min_reserve_visitors_%d' % (i): pd.np.nanmin,
        # })

        # f_time = tmp.groupby(['air_store_id'],as_index=False).visit_reserve_diff.agg({'mean_time_%d' % (i): pd.np.nanmean})
        # f_today_r_v_diff = tmp.groupby(['air_store_id'],as_index=False).today_r_v_diff.agg({'mean_today_r_v_diff_%d' % (i): pd.np.nanmean})

        sub_to_submit = pd.merge(sub_to_submit,f_tmp,on=['air_store_id','dow'],how='left')
        sub_to_submit = pd.merge(sub_to_submit,f_day_tmp,on=['air_store_id'],how='left')
        sub_to_submit = pd.merge(sub_to_submit,f_tmp_hol,on=['air_store_id','dow','holiday_w'],how='left')
        # sub_to_submit = pd.merge(sub_to_submit,f_reserve_visitors_tmp,on=['air_store_id'],how='left')
        # sub_to_submit = pd.merge(sub_to_submit,f_today_r_v_diff,on=['air_store_id'],how='left')

    return sub_to_submit

train['visit_date'] = pd.to_datetime(train['visit_date'])
sub_to_submit = week_feat(train,sub_to_submit)


print(pd.to_datetime(sub_to_submit['visit_date'].min()) - pd.to_datetime(sub_to_submit['visit_date'].max()))
print(pd.to_datetime(sub_to_submit['visit_date'].min()) , pd.to_datetime(sub_to_submit['visit_date'].max()))

val_begin_date = pd.to_datetime(test['visit_date'].min()) - timedelta(days=39)
val_test = train[train['visit_date'] >= val_begin_date]
val_test = pd.merge(val_test,hol,left_on='visit_date',right_on='calendar_date')
val_test = val_test[['air_store_id','visit_date','holiday_flg','dow','visitors','holiday_w']]

print(val_test['visit_date'].min() - val_test['visit_date'].max())
print(val_test['visit_date'].min() , val_test['visit_date'].max())


val_to_test = week_feat(train,val_test)
print(val_to_test['visit_date'].min() - val_to_test['visit_date'].max())
print(val_to_test['visit_date'].min() , val_to_test['visit_date'].max())


train_begin_date = pd.to_datetime(val_test['visit_date'].min()) - timedelta(days=39 * 5)

val_train = train[(train['visit_date'] < val_begin_date)&(train['visit_date'] >= train_begin_date)]
val_train = pd.merge(val_train,hol,left_on='visit_date',right_on='calendar_date')
val_train = val_train[['air_store_id','visit_date','holiday_flg','dow','visitors','holiday_w']]

val_to_train = week_feat(train,val_train)
print(val_to_train['visit_date'].min() - val_to_train['visit_date'].max())
print(val_to_train['visit_date'].min() , val_to_train['visit_date'].max())

#
# val_to_train['air_store_id'] = air_store_id_label.transform(val_to_train['air_store_id'])
# val_to_test['air_store_id'] = air_store_id_label.transform(val_to_test['air_store_id'])
# sub_to_submit['air_store_id'] = air_store_id_label.transform(sub_to_submit['air_store_id'])

print(val_to_train.head())
print(val_to_test.head())
print(sub_to_submit)


del val_to_train['visit_date']
del val_to_test['visit_date']
sub_data = sub_to_submit.pop('visit_date')


y_train = pd.np.log1p(val_to_train.pop('visitors'))
x_train = val_to_train

y_test = pd.np.log1p(val_to_test.pop('visitors'))
x_test = val_to_test

y_sub = pd.np.log1p(sub_to_submit.pop('visitors'))
x_sub = sub_to_submit
#
#

del x_train['air_store_id']
del x_test['air_store_id']
air_store_id = x_sub.pop('air_store_id')
# print(air_store_id)

#
import lightgbm as lgb
gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.1,
    n_estimators=10000)

print(x_train.columns)

m_1 = gbm0.fit(x_train, y_train, eval_metric='rmse', eval_set=[(x_test, y_test)],early_stopping_rounds=25)
y_t = gbm0.predict(x_sub)
# print(y_t)

x_sub['data'] = list(sub_data)
x_sub['visitors'] = pd.np.expm1(y_t)
x_sub['air_store_id'] = list(air_store_id)
# print(x_sub[['air_store_id','data','visitors']])
# # x_sub['air_store_id'] = air_store_id_label.inverse_transform(x_sub['air_store_id'])

x_sub['data'] = x_sub['data'].astype(str)
x_sub['id'] = x_sub['air_store_id'] + '_' + x_sub['data']
x_sub = x_sub.sort_values('id')
x_sub[['id', 'visitors']].to_csv('../result/20180121_1.csv', float_format='%.3f', index=None)



#coding:utf-8

import pandas as pd

# 首先根据目标发现 air_visit_data.csv 是最终要的一张表，包含了目标数据

file_dir = '../data/'
air_visit_data = pd.read_csv(file_dir + 'air_visit_data.csv')

# sample_submission = air_visit_data[air_visit_data['visit_date']>='2017-03-11']
# sample_submission.rename(columns = {'visit_date':'calendar_date'},inplace=True)

# air_visit_data = air_visit_data[air_visit_data['visit_date']<'2017-03-11']
# 提交数据需要特殊处理
sample_submission = pd.read_csv(file_dir + 'sample_submission.csv')
sample_submission['air_store_id'] = sample_submission['id'].map(lambda x:'_'.join(str(x).split('_')[:2]))
sample_submission['calendar_date'] = sample_submission['id'].map(lambda x:str(x).split('_')[-1])
# sample_submission.drop(['id'],axis=1,inplace=True)

# data_info.csv from kaggle 根据时间增加权重 倒叙权重
date_info = pd.read_csv(file_dir + 'date_info.csv')
wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0

# date_info = date_info[date_info['calendar_date']<'2017-04-11']

# 这两个效果一样
# date_info['weight'] = ((date_info.index + 1) * 1.0 / (len(date_info))) ** 7
date_info['weight'] = 1.0 / (len(date_info) - date_info.index)**2
#0.631359846869

# 去除样本大于200的数据，保证样本的稳定性
air_visit_data = air_visit_data[air_visit_data['visitors'] <= 180]

air_visit_data = air_visit_data.merge(date_info,left_on='visit_date',right_on='calendar_date',how='left')
air_visit_data.drop('calendar_date',axis=1,inplace=True)
# log 平滑处理
air_visit_data['visitors'] = air_visit_data.visitors.map(pd.np.log1p)
# 根据星期去计算权值

wmean =  lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
air_visit_data = air_visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
air_visit_data.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.

# 结果和test集合合并
# 基本思路是 周期 + 节假日 + 历史日期的权值因子
sample_submission.drop('visitors', axis=1, inplace=True)
# sample_submission.rename(columns={'visitors':'t_visitors'},inplace=True)

sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(air_visit_data, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')


sample_submission.loc[sample_submission.visitors.isnull(),'visitors'] = sample_submission[sample_submission.visitors.isnull()].merge(
    air_visit_data[air_visit_data.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values

sample_submission.loc[sample_submission.visitors.isnull(),'visitors'] = sample_submission[sample_submission.visitors.isnull()].merge(
    air_visit_data[['air_store_id','visitors']].groupby('air_store_id').median().reset_index(),on=['air_store_id'],how='left'
)['visitors_y'].values

import numpy as np

def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypothesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


sample_submission['visitors'] = sample_submission['visitors'].map(pd.np.expm1)
# print sample_submission
sample_submission[['id', 'visitors']].to_csv('../result/2_180_median.csv', float_format='%.4f', index=None)
# print sample_submission[['t_visitors', 'visitors','day_of_week']]
# print rmsle(sample_submission['t_visitors'] ,sample_submission['visitors'] )



# import matplotlib.pyplot as plt

# list_air_id = list(air_visit_data['air_store_id'].unique())
# for i in list_air_id:
#     tmp = air_visit_data[air_visit_data['air_store_id']==i]
#     plt.plot(range(len(tmp['visitors'])),tmp['visitors'])


# viewdata
# plt.plot(range(len(air_visit_data['visitors'])),air_visit_data['visitors'])
# plt.show()

#coding:utf-8
import pandas as pd
from sklearn.metrics import mean_squared_error
res = '../result/'
# 0.496
ruler = pd.read_csv(res + '20180124_megr.csv')
# 0.495
xgb = pd.read_csv(res + 'kaggle_480.csv')
# 0.486

r = pd.merge(ruler,xgb,on=['id'])

# r['visitors'] = 2.0 / (1.0/r['visitors_x'] + 1.0/r['visitors_y'])
r['visitors'] =  (r['visitors_x'] + r['visitors_y']) / 2
r[['id', 'visitors']].to_csv(
    '../result/20180124_megr_kaggle.csv', index=False, float_format='%.3f')


xx_480 = pd.read_csv(res + 'submission.csv')
xx_482 = pd.read_csv(res + 'submission (1).csv')

xx_483 = pd.read_csv(res + '0.483.csv')
xx_482_1 = pd.read_csv(res + 'LGB_sub_0101_add_hol_cnt_prob0.482.csv')

print mean_squared_error(pd.np.log1p(xx_480['visitors']),pd.np.log1p(r['visitors']))

print mean_squared_error(pd.np.log1p(xx_480['visitors']),pd.np.log1p(xx_483['visitors']))

print mean_squared_error(pd.np.log1p(xx_480['visitors']),pd.np.log1p(r['visitors_y']))


print mean_squared_error(pd.np.log1p(xx_482_1['visitors']),pd.np.log1p(r['visitors_y']))

print mean_squared_error(pd.np.log1p(xx_480['visitors']),pd.np.log1p(xx_482['visitors']))

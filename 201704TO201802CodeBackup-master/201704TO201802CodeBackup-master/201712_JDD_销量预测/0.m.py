#coding:utf-8
# 月份模型
import pandas as pd

dir = '../data/'
t_order = pd.read_csv(dir + 't_order.csv')
t_order['year_month'] = t_order['ord_dt'].map(lambda x:str(x)[:7])
# 统计历史的sum根据年月统计
t_order_sum = t_order[t_order.columns.drop(['ord_dt','pid'])].groupby(['shop_id','year_month'],as_index=False).sum()
# 销售减去退货==这个月真正的销售额
t_order_sum['true_sale_amt'] = t_order_sum['sale_amt'] - t_order_sum['rtn_amt']
# 订单 - 退货
t_order_sum['true_ord_cnt'] = t_order_sum['ord_cnt'] - t_order_sum['rtn_cnt']
# 退货比
t_order_sum['rtn_ord_ration'] = t_order_sum['rtn_cnt'] / t_order_sum['ord_cnt']
# 人均退货比
t_order_sum['rtn_ord_ration'] = t_order_sum['rtn_cnt'] / t_order_sum['user_cnt']
# 人均退货金额
t_order_sum['rtn_user_ration'] = t_order_sum['rtn_amt'] / t_order_sum['user_cnt']
# 人均订单数
t_order_sum['ord_user_ration'] = t_order_sum['ord_cnt'] / t_order_sum['user_cnt']

# print t_order_sum

t_comment = pd.read_csv(dir + 't_comment.csv')
t_comment['year_month'] = t_comment['create_dt'].map(lambda x:str(x)[:7])
t_comment_sum = t_comment[t_comment.columns.drop(['create_dt'])].groupby(['shop_id','year_month'],as_index=False).sum()
# 以月为单位
t_comment_sum['bad_cmmt_ration'] = t_comment_sum['bad_num'] / t_comment_sum['cmmt_num']
t_comment_sum['mid_cmmt_ration'] = t_comment_sum['mid_num'] / t_comment_sum['cmmt_num']
t_comment_sum['good_cmmt_ration'] = t_comment_sum['good_num'] / t_comment_sum['cmmt_num']
t_comment_sum['dis_cmmt_ration'] = t_comment_sum['dis_num'] / t_comment_sum['cmmt_num']
# print t_comment_sum

t_ads = pd.read_csv(dir + 't_ads.csv')
t_ads['year_month'] = t_ads['create_dt'].map(lambda x:str(x)[:7])
t_ads_sum = t_ads[t_ads.columns.drop(['create_dt'])].groupby(['shop_id','year_month'],as_index=False).sum()

# print t_ads_sum

t_product = pd.read_csv(dir + 't_product.csv')
t_product['year_month'] = t_product['on_dt'].map(lambda x:str(x)[:7])
t_product_cate = t_product[t_product.columns.drop(['on_dt'])].groupby([
                                'shop_id','cate','year_month'],as_index=False).count()
t_product_cate = t_product_cate.groupby(['shop_id','year_month'],as_index=False).cate.count()
t_product_cate.rename(columns={'cate':'cate_num'},inplace=True)

t_product_brand = t_product[t_product.columns.drop(['on_dt'])].groupby([
                                'shop_id','brand','year_month'],as_index=False).count()
t_product_brand = t_product_brand.groupby(['shop_id','year_month'],as_index=False).brand.count()
t_product_brand.rename(columns={'brand':'brand_num'},inplace=True)

t_sales_sum = pd.read_csv(dir + 't_sales_sum.csv').drop_duplicates()
t_sales_sum['year_month'] = t_sales_sum['dt'].map(lambda x:str(x)[:7])


x_comment_08 = t_comment_sum[t_comment_sum['year_month']== '2016-08'].groupby(['shop_id'],as_index=False).sum()
x_comment_09 = t_comment_sum[t_comment_sum['year_month']== '2016-09'].groupby(['shop_id'],as_index=False).sum()
x_comment_10 = t_comment_sum[t_comment_sum['year_month']== '2016-10'].groupby(['shop_id'],as_index=False).sum()
x_comment_11 = t_comment_sum[t_comment_sum['year_month']== '2016-11'].groupby(['shop_id'],as_index=False).sum()
x_comment_12 = t_comment_sum[t_comment_sum['year_month']== '2016-12'].groupby(['shop_id'],as_index=False).sum()
x_comment_01 = t_comment_sum[t_comment_sum['year_month']== '2017-01'].groupby(['shop_id'],as_index=False).sum()
x_comment_02 = t_comment_sum[t_comment_sum['year_month']== '2017-02'].groupby(['shop_id'],as_index=False).sum()
x_comment_03 = t_comment_sum[t_comment_sum['year_month']== '2017-03'].groupby(['shop_id'],as_index=False).sum()
x_comment_04 = t_comment_sum[t_comment_sum['year_month']== '2017-04'].groupby(['shop_id'],as_index=False).sum()
#

x_ads_08 = t_ads_sum[t_ads_sum['year_month']== '2016-08'].groupby(['shop_id'],as_index=False).sum()
x_ads_09 = t_ads_sum[t_ads_sum['year_month']== '2016-09'].groupby(['shop_id'],as_index=False).sum()
x_ads_10 = t_ads_sum[t_ads_sum['year_month']== '2016-10'].groupby(['shop_id'],as_index=False).sum()
x_ads_11 = t_ads_sum[t_ads_sum['year_month']== '2016-11'].groupby(['shop_id'],as_index=False).sum()
x_ads_12 = t_ads_sum[t_ads_sum['year_month']== '2016-12'].groupby(['shop_id'],as_index=False).sum()
x_ads_01 = t_ads_sum[t_ads_sum['year_month']== '2017-01'].groupby(['shop_id'],as_index=False).sum()
x_ads_02 = t_ads_sum[t_ads_sum['year_month']== '2017-02'].groupby(['shop_id'],as_index=False).sum()
x_ads_03 = t_ads_sum[t_ads_sum['year_month']== '2017-03'].groupby(['shop_id'],as_index=False).sum()
x_ads_04 = t_ads_sum[t_ads_sum['year_month']== '2017-04'].groupby(['shop_id'],as_index=False).sum()


import gc
#
x_train_08 = t_order_sum[(t_order_sum['year_month'] == '2016-08')]
x_train_08 = pd.merge(x_train_08,x_comment_08,on=['shop_id'],how='left')
x_train_08 = pd.merge(x_train_08,x_ads_08,on=['shop_id'],how='left')
del x_train_08['year_month']
x_train_09 = t_order_sum[(t_order_sum['year_month'] == '2016-09')]
x_train_09 = pd.merge(x_train_09,x_comment_09,on=['shop_id'],how='left')
x_train_09 = pd.merge(x_train_09,x_ads_09,on=['shop_id'],how='left')

del x_train_09['year_month']
x_train_10 = t_order_sum[(t_order_sum['year_month'] == '2016-10')]
x_train_10 = pd.merge(x_train_10,x_comment_10,on=['shop_id'],how='left')
x_train_10 = pd.merge(x_train_10,x_ads_10,on=['shop_id'],how='left')

del x_train_10['year_month']
x_train_11 = t_order_sum[(t_order_sum['year_month'] == '2016-11')]
x_train_11 = pd.merge(x_train_11,x_comment_11,on=['shop_id'],how='left')
x_train_11 = pd.merge(x_train_11,x_ads_11,on=['shop_id'],how='left')

del x_train_11['year_month']
x_train_12 = t_order_sum[(t_order_sum['year_month'] == '2016-12')]
x_train_12 = pd.merge(x_train_12,x_comment_12,on=['shop_id'],how='left')
x_train_12 = pd.merge(x_train_12,x_ads_12,on=['shop_id'],how='left')

del x_train_12['year_month']
x_train_01 = t_order_sum[(t_order_sum['year_month'] == '2017-01')]
x_train_01 = pd.merge(x_train_01,x_comment_01,on=['shop_id'],how='left')
x_train_01 = pd.merge(x_train_01,x_ads_01,on=['shop_id'],how='left')

del x_train_01['year_month']
x_train_02 = t_order_sum[(t_order_sum['year_month'] == '2017-02')]
x_train_02 = pd.merge(x_train_02,x_comment_02,on=['shop_id'],how='left')
x_train_02 = pd.merge(x_train_02,x_ads_02,on=['shop_id'],how='left')

del x_train_02['year_month']
x_train_03 = t_order_sum[(t_order_sum['year_month'] == '2017-03')]
x_train_03 = pd.merge(x_train_03,x_comment_03,on=['shop_id'],how='left')
x_train_03 = pd.merge(x_train_03,x_ads_03,on=['shop_id'],how='left')

del x_train_03['year_month']
x_train_04 = t_order_sum[(t_order_sum['year_month'] == '2017-04')]
x_train_04 = pd.merge(x_train_04,x_comment_04,on=['shop_id'],how='left')
x_train_04 = pd.merge(x_train_04,x_ads_04,on=['shop_id'],how='left')

del x_train_04['year_month']
gc.collect()
#
# 前推移3个月 11 12 01
x_train_1 = pd.merge(x_train_11,x_train_12,on=['shop_id'],how='outer')
x_train_1 = pd.merge(x_train_1,x_train_01,on=['shop_id'],how='outer').fillna(0)

y_train_1 = t_sales_sum[(t_sales_sum['year_month'] == '2017-01')].sort_values('shop_id')[['sale_amt_3m','shop_id']]
X_train_1 = pd.merge(x_train_1,y_train_1,on=['shop_id'],how='outer')
# print x_train_1
# print y_train_1
# 10 11 12
x_train_2 = pd.merge(x_train_10,x_train_11,on=['shop_id'],how='outer')
x_train_2 = pd.merge(x_train_2,x_train_12,on=['shop_id'],how='outer').fillna(0)
y_train_2 = t_sales_sum[(t_sales_sum['year_month'] == '2016-12')].sort_values('shop_id')[['sale_amt_3m','shop_id']]
X_train_2 = pd.merge(x_train_2,y_train_2,on=['shop_id'],how='outer')

# print x_train_2
# print y_train_2

# 09 10 11
x_train_3 = pd.merge(x_train_09,x_train_10,on=['shop_id'],how='outer')
x_train_3 = pd.merge(x_train_3,x_train_11,on=['shop_id'],how='outer').fillna(0)
y_train_3 = t_sales_sum[(t_sales_sum['year_month'] == '2016-11')].sort_values('shop_id')[['sale_amt_3m','shop_id']]
X_train_3 = pd.merge(x_train_3,y_train_3,on=['shop_id'],how='outer')

# print x_train_3
# print y_train_3

# 08 09 10
x_train_4 = pd.merge(x_train_08,x_train_09,on=['shop_id'],how='outer')
x_train_4 = pd.merge(x_train_4,x_train_10,on=['shop_id'],how='outer').fillna(0)
y_train_4 = t_sales_sum[(t_sales_sum['year_month'] == '2016-10')].sort_values('shop_id')[['sale_amt_3m','shop_id']]
X_train_4 = pd.merge(x_train_4,y_train_4,on=['shop_id'],how='outer')


# print x_train_4
# print y_train_4

# 02 03 04
x_train_5 = pd.merge(x_train_02,x_train_03,on=['shop_id'],how='outer')
x_train_5 = pd.merge(x_train_5,x_train_04,on=['shop_id'],how='outer').fillna(0)
x_train_5 = x_train_5.sort_values('shop_id')
# y_train_5 = t_sales_sum[(t_sales_sum['year_month'] == '2017-10')].sort_values('shop_id')
# print x_train_5
# print y_train_5

# last_10 = pd.merge(x_train_08,t_sales_sum[t_order_sum['year_month'] == '2016-10'])[['shop_id','sale_amt_3m']]
# last_09 = pd.merge(x_train_08,t_sales_sum[t_order_sum['year_month'] == '2016-09'])
# last_08 = pd.merge(x_train_08,t_sales_sum[t_order_sum['year_month'] == '2016-08'])
# last_01 = pd.merge(x_train_08,t_sales_sum[t_order_sum['year_month'] == '2017-01'])


datas = []
y_train_1 = X_train_1.pop('sale_amt_3m')
y_train_2 = X_train_2.pop('sale_amt_3m')
y_train_3 = X_train_3.pop('sale_amt_3m')
y_train_4 = X_train_4.pop('sale_amt_3m')

datas.append([X_train_1,y_train_1])
datas.append([X_train_2,y_train_2])
datas.append([X_train_3,y_train_3])
datas.append([X_train_4,y_train_4])


import xgboost as xgb

print('bulid _3_ model')
bsts = []
bst1 = xgb.XGBRegressor(n_estimators=500,max_depth=5,seed=199,learning_rate=0.09)
bsts.append(bst1)
bst2 = xgb.XGBRegressor(n_estimators=500,max_depth=5,seed=199,learning_rate=0.09)
bsts.append(bst2)
bst3 = xgb.XGBRegressor(n_estimators=500,max_depth=5,seed=199,learning_rate=0.09)
bsts.append(bst3)
bst4 = xgb.XGBRegressor(n_estimators=500,max_depth=5,seed=199,learning_rate=0.09)
bsts.append(bst4)

import numpy as np
def score_function(ytrue, ypre):
    ypre = np.asmatrix(ypre)
    ytrue = np.asmatrix(ytrue)
    a = ypre - ytrue
    b = abs(a).sum()
    c = ytrue.sum()
    return b / c

for i in range(len(bsts)):
    bsts[i].fit(datas[i][0].as_matrix(),datas[i][1].as_matrix())

for i,bst in enumerate(bsts):
    for j,data in enumerate(datas):
        if i!=j:
            res = bst.predict(data[0].as_matrix())
            print('data:%s,bst:%s'%(j,i))
            print(score_function(data[1],res))
            print('------------')
gc.collect()

res = []
for i in bsts:
    res.append(i.predict(x_train_5.as_matrix()))
print('generate result')
#%%
res = pd.DataFrame(res).T
res.columns=['pre0','pre1','pre2','pre3']
# 0.7 0.4 0.5 0.6
res = pd.DataFrame(((res.pre0*0.9+res.pre1*1.3+res.pre2*0.9+res.pre3*1)/4)).reset_index()
# res[0]=res[0].astype('int')
res['index'] = res['index']+1
ResFileName = './1203_xgb_4_flod.csv'
res.to_csv(ResFileName,header = False,encoding='utf-8',index=False)
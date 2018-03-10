#coding:utf-8

# 主要思路 寻找当月销量和未来90天的一个系数 一个简单的规则 A榜 0.415的线上分数
# 改善思路，数据去掉一些异常值
# 主观相反，推测未来销量不低于当月销量的一个倍数

import pandas as pd
import math
# 读取原始数据 订单数据
dir = '../data/'
t_order = pd.read_csv(dir + 't_order.csv')
t_order['year_month'] = t_order['ord_dt'].map(lambda x:str(x)[:7])
# 统计历史的sum根据年月统计
t_order_sum = t_order[t_order.columns.drop(['ord_dt','pid'])].groupby(['shop_id','year_month'],as_index=False).sum()

# 此处可以手动查看 11 12 1 月份每个shop当月销量的的sum 
# 2016-11 2016-12 2017-01
month_4_sale = t_order_sum[t_order_sum['year_month']=='2016-11'][['sale_amt','shop_id','offer_amt','rtn_amt']]
month_4_sale['sale_amt'] = month_4_sale['sale_amt']
# 读取未来90天的数据
true_12_sale = pd.read_csv(dir + 't_sales_sum.csv')
true_12_sale['year_month'] = true_12_sale['dt'].map(lambda x:str(x)[:7])
# 2016-11 2016-12 2017-01
true_12_sale = true_12_sale[true_12_sale['year_month']=='2016-11']
# 计算比值
all_ = pd.merge(month_4_sale,true_12_sale,on='shop_id').drop_duplicates('shop_id')

print sum(all_['sale_amt_3m'] / all_['sale_amt']) 


# 以下是预测过程
# month_4_sale = month_4_sale.groupby(['shop_id'],as_index=False).sale_amt.sum()
# month_4_sale['sale'] = month_4_sale['sale_amt'] * (8025.17717344 / 3000)
# month_4_sale[['shop_id','sale']].to_csv('./1203_e_sale_ruler.csv',index=False,header=False)
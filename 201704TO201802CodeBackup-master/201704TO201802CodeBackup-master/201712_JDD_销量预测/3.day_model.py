#coding:utf-8
import pandas as pd
# 自己统计的后90天的数据 用90天预测90天，180天 ， 270天-180 = 90天 向前推移80次
# 80 * 3000 = 240000数据

dir = '../data/'
t_order = pd.read_csv(dir + 't_order.csv',parse_dates=['ord_dt'])
t_order['day'] = t_order['ord_dt'].dt.day
t_order['month'] = t_order['ord_dt'].dt.month
t_order['year'] = t_order['ord_dt'].dt.year
t_order['week'] = t_order['ord_dt'].dt.weekday + 1

t_order['max_seq'] = max(t_order['ord_dt'])
t_order['time_seq'] = t_order['max_seq'].sub(t_order['ord_dt'],axis=0).dt.days
t_order.drop(['max_seq','ord_dt'],axis=1,inplace=True)

print t_order.sort_values(['time_seq'])

# shop_id_list = list(t_order.shop_id.unique())

# tmp = t_order[t_order['shop_id']== 1630]
# print tmp.groupby(['ord_dt'],as_index=False).sale_amt.sum().sort_values(['ord_dt'])

# sale = pd.read_csv(dir + 't_sales_sum.csv')
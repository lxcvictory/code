#encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import gc
import time
from datetime import datetime
from datetime import timedelta
def get_cate8_sku_brand_data():
	actions2 = pd.read_csv('./data/JData_Action_201602.csv')
	actions23 = pd.concat([actions2,pd.read_csv('./data/JData_Action_201603.csv')])
	del actions2
	print "1"
	actions = pd.concat([actions23,pd.read_csv('./data/JData_Action_201604.csv')])
	del actions23
	print "2"
	# actions = pd.read_csv('./data/JData_Action_201604.csv')
	
	del actions['model_id']
	gc.collect()
	actions = actions.sort_values('user_id')
	actions['lookcate8num'] = 1
	user_sku_count = actions[actions['cate'] == 8].drop_duplicates(['user_id','sku_id']).groupby('user_id')['lookcate8num'].sum().reset_index()
	actions['lookcate8brandnum'] = 1
	user_brand_count = actions[actions['cate'] == 8].drop_duplicates(['user_id','brand']).groupby('user_id')['lookcate8brandnum'].sum().reset_index()
	user_sku_brand = pd.merge(user_sku_count,user_brand_count,on='user_id',how='outer')
	user_sku_brand = user_sku_brand.fillna(0)
	return user_sku_brand

def get_sku_time_seq():
	# 读取数据
	actions2 = pd.read_csv('./JData_Action_201602.csv')[['user_id','time','cate','type']]
	# del actions2['model_id']
	# del actions2['brand']
	# del actions2['sku_id']
	actions23 = pd.concat([actions2,pd.read_csv('./JData_Action_201603.csv')[['user_id','time','cate','type']]])
	del actions2
	print "1"
	actions = pd.concat([actions23,pd.read_csv('./JData_Action_201604.csv')[['user_id','time','cate','type']]])
	del actions23
	print "2"
	# actions = pd.read_csv('./data/JData_Action_201604.csv')
	gc.collect()
	print "read_finsih"
	# 根据用户id排序
	actions = actions.sort_values('user_id')
	# 获取所有与cate8有关的用户
	actions_cate_8 = actions[actions['cate'] == 8]
	# 获取购买cate8的用户 =》 获取购买时间 buy_time
	actions_cate_8_buy = actions_cate_8[actions_cate_8['type'] == 4]
	actions_cate_8_buy = actions_cate_8_buy.drop_duplicates(['user_id','time'])
	# 获取用户的开始接触时间和结束时间
	user_time_seq_all_cate = actions_cate_8.groupby('user_id')['time'].agg([('beigintime',np.min),('endtime_tmp',np.max)]).reset_index()
	actions_cate_8_time_seq = pd.merge(user_time_seq_all_cate,actions_cate_8_buy,on='user_id',how='left')
	del user_time_seq_all_cate
	del actions_cate_8_buy
	del actions_cate_8_time_seq['cate']
	del actions_cate_8_time_seq['type']
	gc.collect()
	actions_cate_8_time_seq = actions_cate_8_time_seq.fillna(0)
	# 获取到需要的时间序列
	# print actions_cate_8_time_seq.head()
	actions_cate_8_time_seq.to_csv('./tmp_time_seq.csv')

def get_time_point():
		# 读取数据
	actions2 = pd.read_csv('./JData_Action_201602.csv')
	# del actions2['model_id']
	# del actions2['brand']
	# del actions2['sku_id']
	actions23 = pd.concat([actions2,pd.read_csv('./JData_Action_201603.csv')])
	del actions2
	print "1"
	actions = pd.concat([actions23,pd.read_csv('./JData_Action_201604.csv')])
	del actions23
	print "2"
	# actions = pd.read_csv('./data/JData_Action_201604.csv')
	gc.collect()
	print "read_finsih"

	tmp_time_seq = pd.read_csv('./tmp_time_seq.csv')
	# print tmp_time_seq
	# data.iterrows()
	user_id_sets = []
	beigintime = []
	endtime = []
	# time_seq = []
	time_point = []
	cate_8_time_point = []
	lookcate8num = []
	lookcate8brandnum = []
	# actions = actions[actions['user_id'] <= 200036]
	# tmp_time_seq = tmp_time_seq[tmp_time_seq['user_id'] == 200036]
	for i,sun_action in tmp_time_seq.iterrows():
		flag = 0
	# for i,sun_action in 
		user_id_sets.append(sun_action['user_id'])
		beigintime.append(sun_action['beigintime'])
		if sun_action['time'] == '0':
			endtime.append('2016-04-16')
			# times = pd.to_datetime(sun_action['endtime_tmp']) - pd.to_datetime(sun_action['beigintime'])
			# print times
			# time_seq.append(times)
		else:
			flag = 1
			endtime.append(sun_action['time'])
			# times = pd.to_datetime(sun_action['time']) - pd.to_datetime(sun_action['beigintime'])
			# print times
			# time_seq.append(times)	
		# print "=================="
		# print user_id_sets[-1]
		# print beigintime[-1]
		# print endtime[-1]
		# 最后一个数字
		tmp_user_action = actions[actions['user_id'] == user_id_sets[-1]]


		tmp_user_action = tmp_user_action[(tmp_user_action['time']>=beigintime[-1])&(tmp_user_action['time']<=endtime[-1])]
		# print tmp_user_action
		ttmp_user_action = tmp_user_action
		tmp_user_action_cate_8 = tmp_user_action
		# print ttmp_user_action
		tmp_user_action['time'] = pd.to_datetime(tmp_user_action['time']).map(lambda x : x.strftime('%Y-%m-%d')) 
		tmp_user_action = tmp_user_action.drop_duplicates(['time'])
		# print tmp_user_action
		# print len(tmp_user_action['time'].unique()) - flag 
		time_point.append(len(tmp_user_action['time'].unique()) - flag )
		# 获取这段时间内 cate8的操作记录
		tmp_user_action_cate_8 = tmp_user_action_cate_8[tmp_user_action_cate_8['cate'] == 8 ]
		tmp_user_action_cate_8['time'] = pd.to_datetime(tmp_user_action_cate_8['time']).map(lambda x : x.strftime('%Y-%m-%d'))  
		tmp_user_action_cate_8 = tmp_user_action_cate_8.drop_duplicates(['user_id','time'])
		# print len(tmp_user_action_cate_8['time'].unique()) - flag
		cate_8_time_point.append(len(tmp_user_action_cate_8['time'].unique()) - flag )

		user_sku_count = ttmp_user_action[ttmp_user_action['cate'] == 8].drop_duplicates(['sku_id'])
		# print len(user_sku_count['sku_id'].unique()) - flag
		lookcate8num.append(len(user_sku_count['sku_id'].unique()) - flag)
		user_brand_count = ttmp_user_action[ttmp_user_action['cate'] == 8].drop_duplicates(['brand'])
		# print len(user_brand_count['brand'].unique()) - flag
		lookcate8brandnum.append(len(user_brand_count['brand'].unique()) - flag)
	
	res = pd.DataFrame({'user_id':user_id_sets,
				'beigintime':beigintime,
				'endtime':endtime,
				# 'time_seq':time_seq,
				'time_point':time_point,
				'cate_8_time_point':cate_8_time_point,
				'lookcate8num':lookcate8num,
				'lookcate8brandnum':lookcate8brandnum})
	res.to_csv('./tmp_time_seq_num.csv')

	# print res
def count_cc():
	tmp_time_seq_num = pd.read_csv('./tmp_time_seq_num.csv')
	tmp_time_seq_num['time_seq'] = pd.to_datetime(tmp_time_seq_num['endtime']) - pd.to_datetime(tmp_time_seq_num['beigintime'])
	tmp_time_seq_num['time_seq'] = tmp_time_seq_num['time_seq'].dt.days
	# tmp_time_seq_num['time_seq_1'] = pd.to_datetime('2016-04-15') - pd.to_datetime(tmp_time_seq_num['beigintime'])
	# tmp_time_seq_num['time_seq_1'] = pd.to_datetime(tmp_time_seq_num['time_seq_1']).dt.day
	# tmp_time_seq_num['dd'] = pd.to_datetime('2016-04-15') - pd.to_datetime(tmp_time_seq_num['beigintime'])

	tmp_time_seq_num.to_csv('./count_user_action_time_fecture.csv')

if __name__ == '__main__':
	get_sku_time_seq()
	get_time_point()
	count_cc()
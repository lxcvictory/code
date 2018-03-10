#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import gc

# action_1_path = "./data/JData_Action_201602.csv"
# action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"
# 之前用excle以及其他方式统计的用户行为在时间段内的具体行为 2 3 4
Count_PATH = "./data/count_user_action_time_fecture.csv"

# 评论的日期 根据 comment 获取的时间
def get_comment_list():
	comment = pd.read_csv(comment_path)
	print  comment['dt'].unique()
# 获取到的评论日期的list
comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", 
				"2016-03-07", "2016-03-14","2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

# 用户对cate8的点击和浏览的统计
def get_action_cate8(start_date, end_date,i):
    actions = get_actions(start_date, end_date) 
    actions = actions[['user_id', 'type','time','cate']]
    # print actions.head(10)
    actions = actions[actions['cate']==8]
    actions['date']=pd.to_datetime(actions['time']).map(lambda x : x.strftime('%Y-%m-%d'))  
    # print actions.head(10)
    actions=actions.groupby(['user_id','date'])['type'].value_counts().unstack().reset_index().groupby(['user_id'])[1,6].agg([('%s_num_mean_cate8'% i,np.mean),('%s_num_max_cate8'% i,np.max),('%s_num_min_cate8'% i,np.min),('%s_num_std_cate8'% i,np.std),('%s_num_sum_cate8'% i,np.sum)]).fillna(0)
    # print actions
    actions.columns = ['_'.join((str(col[0]), str(col[1]))) for col in actions.columns]
    actions=actions.reset_index()
    actions['1_%s_num_sum_ratio_cate8'% i]=actions['1_%s_num_sum_cate8'% i]/(actions['6_%s_num_sum_cate8'% i]+actions['1_%s_num_sum_cate8'% i])
    actions['6_%s_num_sum_ratio_cate8'% i]=actions['6_%s_num_sum_cate8'% i]/(actions['6_%s_num_sum_cate8'% i]+actions['1_%s_num_sum_cate8'% i])
    return actions

def get_action_multi_cate8(train_start_date, train_end_date):
    actions = None
    # 参考0.07 构造特征
    for i in (1, 5, 7, 10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_cate8(start_days, train_end_date,i)
        else:
            actions = pd.merge(actions, get_action_cate8(start_days, train_end_date,i), how='outer',
                               on=['user_id']).fillna(0) 
    return actions

# 获取交互天数
def get_action_interaction_day(start_date, end_date,i):
    # print i
    actions = get_actions(start_date, end_date) 
    actions = actions[['user_id', 'type','time']]
    actions['time']=pd.to_datetime(actions['time'])
    actions['days']=actions['time'].dt.day
    actions=actions[['user_id','type','days']].drop_duplicates()
    actions=actions.groupby(['user_id'],)['type'].value_counts().unstack().fillna(0).reset_index()
    actions.rename(columns={1: '%s_type1_days'% i,
                            2: '%s_type2_days'% i,
                            3: '%s_type3_days'% i,
                            4: '%s_type4_days'% i,
                            5: '%s_type5_days'% i,
                            6: '%s_type6_days'% i,}, inplace=True)
    return actions

def get_action_interaction_day_multi(train_start_date, train_end_date):     
    actions = None
    for i in (1,5,7,10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i) 
        start_days = start_days.strftime('%Y-%m-%d') 
        if actions is None:
            actions = get_action_interaction_day(start_days, train_end_date,i) 
        else:
            actions = pd.merge(actions, get_action_interaction_day(start_days, train_end_date,i), how='outer',
                    on=['user_id']).fillna(0) 
    return actions

# 年龄字符数据 => 数值特征 
def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1
# 用户年龄映射为数值特征，=> 年龄 性别 等级 
def get_basic_user_feat():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        count_res = pd.read_csv(Count_PATH,encoding='gbk')
        count_res = count_res.fillna(0)
        # time_seq 用户第一次接触商品到购买（或没有购买按照最后一天）的差 一个时间段
        # time_point 实际交互的时间点 各个时间点
        # cate_8_time_point 实际交互的时间点 各个时间点 cate8
        count_res = count_res[['user_id','time_point','cate_8_time_point','time_seq','lookcate8num','lookcate8brandnum']]
        # 
        count_res['cate8_point_ration'] = count_res['cate_8_time_point'] / count_res['time_point']
        count_res['tim_seq_ration'] = count_res['time_seq'] / count_res['time_point']
        count_res['brand_cate8_ratio'] = count_res['lookcate8num'] / count_res['lookcate8brandnum']
        # 对类别特征的one hot 处理
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="lv")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        user = pd.merge(user,count_res,on='user_id',how='left')
        pickle.dump(user, open(dump_path, 'w'))
    return user

# 产品的特征 => 属性 品牌
def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="attr1")
        attr2_df = pd.get_dummies(product["a2"], prefix="attr2")
        attr3_df = pd.get_dummies(product["a3"], prefix="attr3")
        product = pd.concat([product[['sku_id', 'cate']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'w'))
    return product

# 获取用户行为数据 根据图标分析 10-15天的购买斜率最大 仅采用4月份数据训练
def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3

# 左闭右开 获取所有的行为数据
def get_actions(start_date, end_date):
  
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_3 = get_actions_3()
        actions = pd.concat([ action_3 ])
        del action_3
        gc.collect()
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

# 获取行为特征 行为数据的时间数据作为其实数据和最终数据
def get_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
    	# 首先获取基础的用户行为
        actions = get_actions(start_date, end_date)
        # 选取字段
        actions = actions[['user_id', 'sku_id', 'type']]
        # 操作
        type = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        # 品牌
        print 'cacle' 
        actions = pd.concat([actions, type], axis=1)  
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        print 'cacle finish'
        del actions['type']
        gc.collect()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

# 计算产品特征 评论数量  修改为当前最差的产品平评价
def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        # for date in reversed(comment_date):
        #     if date < comment_date_end:
        #         comment_date_begin = date
        #         break
        # 最坏的情况下都卖了那么好的情况也会买因此选择最坏评论的最坏
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        comments = pd.DataFrame(comments).sort_values('bad_comment_rate',ascending=False)
        comments = comments.drop_duplicates(['sku_id'])
        comment_num = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, comment_num], axis=1) 
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'w'))
    return comments

# 计算用户的特征 添加购买力
def get_accumulate_user_feat(start_date, end_date):
    print start_date
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio', 'user_product_power','cate_ratio','cate_ratio_all','model_id_ration','model_id_-1.0','model_id_0.0']
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions['model_id'] = actions['model_id'].fillna(-1)
        df_model = pd.get_dummies(actions['model_id'],prefix='model_id')
        df = pd.get_dummies(actions['type'], prefix='action')
        df_cate = pd.get_dummies(actions['cate'],prefix='cate')
        actions = pd.concat([actions['user_id'], df, df_cate, df_model], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions = actions.fillna(0)
        # 根据群友 北风行的共享的图 在 model_id 为 0 和 空 的情况下发生购买行为比较多 因此认为 选择作为特征 空填充为 -1
        actions['model_id_ration'] = actions['model_id_-1.0'] / actions['model_id_0.0']
        # 转化率
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        # 购买力 用户一段时间购买的总数
        actions['user_product_power'] = actions['action_4'] 
        # 用户对不同cate的操作比例
        actions['cate_ratio'] = actions['cate_8'] / (actions['cate_4'] + actions['cate_5'] + actions['cate_6'] + actions['cate_7'] + actions['cate_9'])
        actions['cate_ratio_all'] = actions['cate_8'] / (actions['cate_4'] + actions['cate_5'] + actions['cate_6'] + actions['cate_7'] + actions['cate_8'] + actions['cate_9'])
        actions = actions[feature]
        # print actions
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

# 计算产品的特征
def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio','product_sale']
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        # 产品销量
        actions['product_sale'] = actions['action_4']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

# 获取标签特征 
def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-04-06"
        multi_cate8 = get_action_multi_cate8(train_start_date, train_end_date)
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        interaction_day = get_action_interaction_day_multi(train_start_date, train_end_date)
        # 获取整个时间的行为数据，作为第一层action 避免数据丢失
        actions = get_action_feat(start_days, train_end_date)
        for i in (1, 5, 7):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            # if actions is None:
            #     actions = get_action_feat(start_days, train_end_date)
            # else:
            # 每次增加不同时间段的
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='outer',on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, multi_cate8, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        # actions = pd.merge(actions, model_feat_multi, how='left', on='user_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, interaction_day, how='left', on='user_id')
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        actions = actions.replace(np.inf,0)
        # 只保留类型为 cate8的数据
        actions = actions[actions['cate'] == 8]

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    gc.collect()
    return users, actions

# 训练样本
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    # 读取pkl的数据
    dump_path = './cache/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:

        start_days = "2016-03-31"
        # 获取用户特征 性别 年龄 等级
        multi_cate8 = get_action_multi_cate8(train_start_date, train_end_date)
        user = get_basic_user_feat()
        # 获取产品特征主要是 属性
        product = get_basic_product_feat()
        # 计算用户特征
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        # 计算产品特征
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        #计算评论产品特征
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        # model_feat_multi = get_model_feat_multi(train_start_date, train_end_date)
        # 获取标签
        labels = get_labels(test_start_date, test_end_date)
        interaction_day = get_action_interaction_day_multi(train_start_date, train_end_date)
        # 首先获取全部的数据 从开始日期到结束
        actions = get_action_feat(start_days, train_end_date)
        for i in (1, 5, 7):

            print u'距离结束%d天'%(i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            # if actions is None:
            #     actions = get_action_feat(start_days, train_end_date)
            # else:
            # actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',on=['user_id', 'sku_id'])
            # 采用外链接 避免丢失数据 样本数据
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='outer',on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, multi_cate8, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, interaction_day, how='left', on='user_id')
        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])

        actions = actions.fillna(0)
        actions = actions.replace(np.inf,0)
    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']
    gc.collect()
    return users, actions, labels

# if __name__ == '__main__':
#     get_action_cate8('2016-03-31','2016-04-10',1)
#     # train_start_date = '2016-04-01'
#     # train_end_date = '2016-04-10'
#     # test_start_date = '2016-04-10'
#     # test_end_date = '2016-04-15'
#     # user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
#     # print user.head(10)
#     # print action.head(10)





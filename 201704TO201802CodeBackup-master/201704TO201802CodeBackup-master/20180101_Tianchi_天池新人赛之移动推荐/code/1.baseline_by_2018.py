#coding:utf-8

# 训练数据包含了抽样出来的一定量用户在一个月时间（11.18~12.18）之内的移动端行为数据（D），
# 评分数据是这些用户在这个一个月之后的一天（12.19）对商品子集（P）的购买数据。
# 参赛者要使用训练数据建立推荐模型，并输出用户在接下来一天对商品子集购买行为的预测结果。


import pandas as pd
import numpy as np
import datetime

path = '../data/'

train_item = pd.read_csv(path + 'tianchi_fresh_comp_train_item.csv')
train_user = pd.read_csv(path + 'tianchi_fresh_comp_train_user.csv')
print(train_user.shape)
print('获取在train_item中的train_user行为')
# train_user = train_user[train_user.item_id.isin(list(train_item.item_id.unique()))]
print(train_user.shape)
print('时间格式修正')
train_user['hour'] = train_user['time'].apply(lambda x:str(x).split(' ')[1])
train_user['time'] = train_user['time'].apply(lambda x:str(x).split(' ')[0])
print(train_user.columns)
# print('剔除12月12日和11月11日数据')
# train_user = train_user[train_user['time']!='2014-12-12']
# train_user = train_user[train_user['time']!='2014-11-11']
# print(train_user.shape)

########################## 针对三元组问题的解决方案 ##########################

# 获取user item label 三元组
def get_user_item_set(train_user,LableDay,is_Train):
    print('标签时间',LableDay)
    print('user-item对的时间',pd.to_datetime(LableDay) - datetime.timedelta(days=1))
    # 获取LabelDay的标签数据
    train_user_label = train_user[train_user['time'] == LableDay]
    # 题目为估计时间点之后的后一天是否会购买商品，对于LabelDay的前一天提取user-item对
    train_user = train_user[pd.to_datetime(train_user['time']) == (pd.to_datetime(LableDay) - datetime.timedelta(days=1))]
    if is_Train == True:
        # 获取目标天数 type = 4 的标签
        train_user_label = pd.DataFrame(train_user_label)[train_user_label['behavior_type'] == 4]
        train_user_label = train_user_label.drop_duplicates(['user_id', 'item_id'])
        # 获取标签数据
        if LableDay =='2014-12-18':
            print(LableDay,'""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""')
            train_user_label[['user_id','item_id']].to_csv('../result/ansewr.csv',index=False)
        # 采用开源的方案获取标签的关系
        label_flag = train_user_label['user_id'] / train_user_label['item_id']
        # 获取前一天有交互的user item 对
        train_user = train_user.drop_duplicates(['user_id','item_id'])
        train_user_flag = train_user['user_id'] / train_user['item_id']
        train_user_flag = train_user_flag.isin(label_flag)
        # 构造 user item label 三元组
        dict = {True: 1, False: 0}
        train_user_flag = train_user_flag.map(dict)
        train_user['label'] = train_user_flag
        train_user = train_user[['user_id','item_id','time','label','item_category']]
        print('购买/未购买比',
              len(train_user[train_user['label']==1]) / len(train_user),
              len(train_user[train_user['label']==1]),
              len(train_user))
    else:
        # Sublimt数据构成 user item
        train_user = pd.DataFrame(train_user).drop_duplicates(['user_id','item_id','time','item_category'])
    return train_user

# 构造特征

# 用户商品交互特征
def user_item_click_feat(data,train_user,is_Train):
    print('样本数据的最后一天', pd.to_datetime(data.time.min()) )
    last_date = pd.to_datetime(data.time.min())
    if is_Train:
        data = data[['user_id','item_id','item_category','label']]
    else:
        data = data[['user_id', 'item_id','item_category']]
    # print('样本数据前一天数据', last_date)
    train_user['time'] = pd.to_datetime((train_user['time']))
    train_user = train_user[pd.to_datetime(train_user['time']) <= last_date]
    # 以天为单位统计行为 1 2 3 4 的特征计数以及比例
    for day in [1]:
        print('特征时间数据> <=',pd.to_datetime(last_date)- datetime.timedelta(days=day),last_date)

        user_item_click_ = train_user[(train_user['time'] > pd.to_datetime(last_date) - datetime.timedelta(days=day))]
        user_item_click_one_hot = pd.get_dummies(user_item_click_['behavior_type'],prefix='user_click_c')
        user_item_click_one_hot['user_click_count'] = user_item_click_one_hot.sum(axis=1)
        user_item_click_ = pd.concat([user_item_click_[['user_id','item_id']],user_item_click_one_hot],axis=1)
        user_item_click_ = user_item_click_.groupby(['user_id','item_id'],as_index=False).sum().add_suffix('_before_%d'%(day))

        for i in user_item_click_.columns:
            if (i == 'user_id_before_%d'%(day)) | (i == 'user_click_count_before_%d'%(day)):
                pass
            else:
                user_item_click_[i + '_ratdio'] = user_item_click_[i] / (user_item_click_['user_click_count_before_%d'%(day)] + 0.00001)

        print('比例特征',day)
        user_item_click_['user_click_c_1_before_%d'%(day)+'/'+'user_click_c_4_before_%d'%(day)] = user_item_click_['user_click_c_1_before_%d'%(day)] / (user_item_click_['user_click_c_4_before_%d'%(day)] + 0.0001)
        user_item_click_['user_click_c_2_before_%d'%(day)+'/'+'user_click_c_4_before_%d'%(day)] = user_item_click_['user_click_c_2_before_%d'%(day)] / (user_item_click_['user_click_c_4_before_%d'%(day)] + 0.0001)
        user_item_click_['user_click_c_3_before_%d'%(day)+'/'+'user_click_c_4_before_%d'%(day)] = user_item_click_['user_click_c_3_before_%d'%(day)] / (user_item_click_['user_click_c_4_before_%d'%(day)] + 0.0001)
        user_item_click_['user_click_c_1_before_%d'%(day)+'/'+'user_click_c_3_before_%d'%(day)] = user_item_click_['user_click_c_1_before_%d'%(day)] / (user_item_click_['user_click_c_3_before_%d'%(day)] + 0.0001)
        user_item_click_['user_click_c_1_before_%d'%(day)+'/'+'user_click_c_2_before_%d'%(day)] = user_item_click_['user_click_c_1_before_%d'%(day)] / (user_item_click_['user_click_c_2_before_%d'%(day)] + 0.0001)
        user_item_click_['user_click_c_2_before_%d'%(day)+'/'+'user_click_c_3_before_%d'%(day)] = user_item_click_['user_click_c_2_before_%d'%(day)] / (user_item_click_['user_click_c_2_before_%d'%(day)] + 0.0001)

        user_item_click_.rename(columns={'user_id_before_%d'%(day):'user_id','item_id_before_%d'%(day):'item_id'},inplace=True)
        data = pd.merge(data,user_item_click_,on=['user_id','item_id'],how='left').fillna(0)
    # print(data.columns)
    return data

# 获取窗口内的行为数据针对用户
def get_user_windows_feat(data,train_user,is_Train):
    print('时间窗口的最后一天',pd.to_datetime(data.time.min()))
    last_date = pd.to_datetime(data.time.min()) - datetime.timedelta(days=7)
    windows_data = train_user[(pd.to_datetime(train_user['time']) > last_date)&(pd.to_datetime(train_user['time']) <= pd.to_datetime(data.time.min()))]

    # 用户第一次浏览
    user_action_type_1 = windows_data[windows_data['behavior_type'] == 1]
    user_action_type_1_first = user_action_type_1.groupby(['user_id'], as_index=False).first()
    user_action_type_1_first.rename(columns={'time': 'time_action_1'}, inplace=True)

    # 用户第一次收藏
    user_action_type_2 = windows_data[windows_data['behavior_type'] == 2]
    user_action_type_2_first = user_action_type_2.groupby(['user_id'], as_index=False).first()
    user_action_type_2_first.rename(columns={'time': 'time_action_2'}, inplace=True)

    # 用户第一次加购
    user_action_type_3 = windows_data[windows_data['behavior_type'] == 3]
    user_action_type_3_first = user_action_type_3.groupby(['user_id'], as_index=False).first()
    user_action_type_3_first.rename(columns={'time': 'time_action_3'}, inplace=True)

    # 用户在指定时间内最后一次购买的时间
    user_action_type_4 = windows_data[windows_data['behavior_type']==4]
    user_action_type_4_last = user_action_type_4.groupby(['user_id'],as_index=False).last()
    user_action_type_4_last.rename(columns={'time':'time_buy'},inplace=True)
    # 用户第一次购买的时间
    user_action_type_4_first = user_action_type_4.groupby(['user_id'], as_index=False).first()
    user_action_type_4_first.rename(columns={'time': 'time_action_4'}, inplace=True)

    # 用户在这段时间内的第一次接触的时间
    user_action_type_first = windows_data.groupby(['user_id'],as_index=False).first()
    #
    user_buy_time = pd.merge(user_action_type_first,user_action_type_4_last[['user_id','time_buy']],on=['user_id'],how='left')
    user_buy_time = pd.merge(user_buy_time,user_action_type_4_first[['user_id','time_action_4']],on=['user_id'],how='left')
    user_buy_time = pd.merge(user_buy_time,user_action_type_1_first[['user_id','time_action_1']],on=['user_id'],how='left')
    user_buy_time = pd.merge(user_buy_time,user_action_type_2_first[['user_id','time_action_2']],on=['user_id'],how='left')
    user_buy_time = pd.merge(user_buy_time,user_action_type_3_first[['user_id','time_action_3']],on=['user_id'],how='left')

    user_buy_time['time_buy'] = user_buy_time['time_buy'].fillna(data.time.unique()[0])
    user_buy_time['time_action_4'] = user_buy_time['time_action_4'].fillna(data.time.unique()[0])
    user_buy_time['time_action_1'] = user_buy_time['time_action_1'].fillna(data.time.unique()[0])
    user_buy_time['time_action_2'] = user_buy_time['time_action_2'].fillna(data.time.unique()[0])
    user_buy_time['time_action_3'] = user_buy_time['time_action_3'].fillna(data.time.unique()[0])
    # 第一次和最后一次购买的时间差
    # 时间windwos内最后一次购买距离第一次购买的时间差
    user_buy_time['buy_4_time_diff'] = pd.to_datetime(user_buy_time['time_buy']) - pd.to_datetime(user_buy_time['time'])
    user_buy_time['buy_4_time_diff_between_2_last'] = pd.to_datetime(user_buy_time['time_buy']) - pd.to_datetime(user_buy_time['time_action_4'])
    user_buy_time['buy_1_4_diff_last'] = pd.to_datetime(user_buy_time['time_buy']) - pd.to_datetime(user_buy_time['time_action_1'])
    user_buy_time['buy_2_4_diff_last'] = pd.to_datetime(user_buy_time['time_buy']) - pd.to_datetime(user_buy_time['time_action_2'])
    user_buy_time['buy_3_4_diff_last'] = pd.to_datetime(user_buy_time['time_buy']) - pd.to_datetime(user_buy_time['time_action_3'])
    #
    user_buy_time['buy_4_time_diff'] = user_buy_time['buy_4_time_diff'].apply(lambda x:x.days)
    user_buy_time['buy_4_time_diff_between_2_last'] = user_buy_time['buy_4_time_diff_between_2_last'].apply(lambda x:x.days)
    user_buy_time['buy_1_4_diff_last'] = user_buy_time['buy_1_4_diff_last'].apply(lambda x:x.days)
    user_buy_time['buy_2_4_diff_last'] = user_buy_time['buy_2_4_diff_last'].apply(lambda x:x.days)
    user_buy_time['buy_3_4_diff_last'] = user_buy_time['buy_3_4_diff_last'].apply(lambda x:x.days)

    # 合并相同用户
    user_buy_time = user_buy_time[['user_id','buy_4_time_diff','buy_4_time_diff_between_2_last',
                                   'buy_1_4_diff_last','buy_2_4_diff_last','buy_3_4_diff_last']]

    return user_buy_time

# 获取用户 品牌的交互记录
def get_user_item_feat(data,train_user,is_Train):
    last_data = data.time.min()
    print('user_item_last_time == ',last_data)
    train_user = train_user[pd.to_datetime(train_user['time']) <= pd.to_datetime(last_data)]
    for day in [1,3,7]:
        print('user_cate > <=',pd.to_datetime(last_data) - datetime.timedelta(days=day),last_data)
        tmp = train_user[pd.to_datetime(train_user['time']) > pd.to_datetime(last_data) - datetime.timedelta(days=day)]
        tmp = tmp[['user_id','item_category','behavior_type']]
        user_item_one_hot = pd.get_dummies(tmp['behavior_type'],prefix='user_item_c')
        user_item_one_hot['user_item_s'] = user_item_one_hot.sum(axis=1)
        user_item = pd.concat([tmp[['user_id','item_category']],user_item_one_hot],axis=1)
        user_item = user_item.groupby(['user_id','item_category'],as_index=False).sum().add_suffix('_before_%d'%(day))
        user_item.rename(columns={'user_id_before_%d'%(day):'user_id','item_category_before_%d'%(day):'item_category'},inplace=True)

        # for i in user_item.columns:
        #     if (i == 'user_id') | (i == 'item_category') | (i == 'user_item_s_before_%d'%(day)):
        #         pass
        #     else:
        #         user_item['%s_ratdio_'%(i)] = user_item[i] / (user_item['user_item_s_before_%d'%(day)] + 0.00001)

        if day == 1:
            result = user_item
        else:
            result = pd.merge(result,user_item,on=['user_id','item_category'],how='outer')
    return result

# 用户行为
def get_user_action(data,train_user,is_Train):
    last_data = data.time.min()
    print('user_last_time == ', last_data)
    train_user = train_user[pd.to_datetime(train_user['time']) <= pd.to_datetime(last_data)]
    for day in [1]:
        print('user > <=', pd.to_datetime(last_data) - datetime.timedelta(days=day), last_data)
        tmp = train_user[pd.to_datetime(train_user['time']) > pd.to_datetime(last_data) - datetime.timedelta(days=day)]
        tmp = tmp[['user_id', 'behavior_type']]
        user_item_one_hot = pd.get_dummies(tmp['behavior_type'], prefix='user_c')
        user_item_one_hot['user_s'] = user_item_one_hot.sum(axis=1)
        user_item = pd.concat([tmp[['user_id']], user_item_one_hot], axis=1)
        user_item = user_item.groupby(['user_id'], as_index=False).sum().add_suffix('_before_%d' % (day))
        user_item.rename(
            columns={'user_id_before_%d' % (day): 'user_id',},
            inplace=True)

        for i in user_item.columns:
            if (i == 'user_id') | (i == 'user_s_before_%d'%(day)):
                pass
            else:
                user_item['%s_ratdio_'%(i)] = user_item[i] / (user_item['user_s_before_%d'%(day)] + 0.00001)

        if day == 1:
            result = user_item
        else:
            result = pd.merge(result, user_item, on=['user_id'], how='outer')
    return result

# 获取数据
print('train_set')
train_user_to_train_1 = get_user_item_set(train_user,'2014-12-17',True)
train_user_to_train_2 = get_user_item_set(train_user,'2014-12-16',True)
train_user_to_train_3 = get_user_item_set(train_user,'2014-12-15',True)
# train_user_to_train_4 = get_user_item_set(train_user,'2014-12-14',True)
# train_user_to_train = pd.concat([train_user_to_train_1,train_user_to_train_2],axis=0)
# 验证数据
print('val_set')
train_user_to_val = get_user_item_set(train_user,'2014-12-18',True)
#
print('sub_set')
train_user_to_sub = get_user_item_set(train_user,'2014-12-19',False)

# windows特征
train_windows_1 = get_user_windows_feat(train_user_to_train_1,train_user,True)
train_windows_2 = get_user_windows_feat(train_user_to_train_2,train_user,True)
train_windows_3 = get_user_windows_feat(train_user_to_train_3,train_user,True)
# train_windows_4 = get_user_windows_feat(train_user_to_train_4,train_user,True)
test_windows = get_user_windows_feat(train_user_to_val,train_user,True)
sub_windows = get_user_windows_feat(train_user_to_sub,train_user,False)

# user cate
train_user_item_category_1 = get_user_item_feat(train_user_to_train_1,train_user,False)
train_user_item_category_2 = get_user_item_feat(train_user_to_train_2,train_user,False)
train_user_item_category_3 = get_user_item_feat(train_user_to_train_3,train_user,False)

val_user_item_category = get_user_item_feat(train_user_to_val,train_user,False)
sub_user_item_category = get_user_item_feat(train_user_to_sub,train_user,False)

# user item
train_1 = user_item_click_feat(train_user_to_train_1,train_user,True)
train_2 = user_item_click_feat(train_user_to_train_2,train_user,True)
train_3 = user_item_click_feat(train_user_to_train_3,train_user,True)
# train_4 = user_item_click_feat(train_user_to_train_4,train_user,True)
val = user_item_click_feat(train_user_to_val,train_user,True)
sub = user_item_click_feat(train_user_to_sub,train_user,False)

# user
train_user_action_1 = get_user_action(train_user_to_train_1,train_user,True)
train_user_action_2 = get_user_action(train_user_to_train_2,train_user,True)
train_user_action_3 = get_user_action(train_user_to_train_3,train_user,True)
# train_4 = user_item_click_feat(train_user_to_train_4,train_user,True)
val_user_action = get_user_action(train_user_to_val,train_user,True)
sub_user_action = get_user_action(train_user_to_sub,train_user,False)

##################### merge ######## merge ###################
# user_item
train_1 = pd.merge(train_1,train_user_item_category_1,on=['user_id','item_category'],how='left')
train_2 = pd.merge(train_2,train_user_item_category_2,on=['user_id','item_category'],how='left')
train_3 = pd.merge(train_3,train_user_item_category_3,on=['user_id','item_category'],how='left')
# train_4 = pd.merge(train_4,train_windows_4,on=['user_id'],how='left')
val = pd.merge(val,val_user_item_category,on=['user_id','item_category'],how='left')
sub = pd.merge(sub,sub_user_item_category,on=['user_id','item_category'],how='left')
#############################################################
# windows
train_1 = pd.merge(train_1,train_windows_1,on=['user_id'],how='left')
train_2 = pd.merge(train_2,train_windows_2,on=['user_id'],how='left')
train_3 = pd.merge(train_3,train_windows_3,on=['user_id'],how='left')
# train_4 = pd.merge(train_4,train_windows_4,on=['user_id'],how='left')
val = pd.merge(val,test_windows,on=['user_id'],how='left')
sub = pd.merge(sub,sub_windows,on=['user_id'],how='left')

#############################################################
# user_action
train_1 = pd.merge(train_1,train_user_action_1,on=['user_id'],how='left')
train_2 = pd.merge(train_2,train_user_action_2,on=['user_id'],how='left')
train_3 = pd.merge(train_3,train_user_action_3,on=['user_id'],how='left')
# train_4 = pd.merge(train_4,train_windows_4,on=['user_id'],how='left')
val = pd.merge(val,val_user_action,on=['user_id'],how='left')
sub = pd.merge(sub,sub_user_action,on=['user_id'],how='left')

#############################################################

train = pd.concat([train_1,train_2],axis=0)
train = pd.concat([train,train_3],axis=0)

# train = pd.concat([train,train_4],axis=0)
val = val
sub = sub

sub_user_id = sub.pop('user_id')
sub_item_id = sub.pop('item_id')
sub_cate_id = sub.pop('item_category')

val_user_id = val.pop('user_id')
val_item_id = val.pop('item_id')
val_cate_id = val.pop('item_category')
val_label = val.pop('label')

train_item_id = train.pop('item_id')
train_user_id = train.pop('user_id')
train_cate_id = train.pop('item_category')
train_label = train.pop('label')

print('++++++++++++++++++++++++++++++++++++')
print(train.columns)
#

import lightgbm as lgb

lgb_train = lgb.Dataset(train.values, train_label.values)
lgb_eval = lgb.Dataset(val.values, val_label.values, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=4000,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=25)


y_t = gbm.predict(val.values, num_iteration=gbm.best_iteration)
y_sub = gbm.predict(sub.values, num_iteration=gbm.best_iteration)

res = pd.DataFrame()
res['true'] = list(val_label)
res['pre'] = list(y_t)
# val_item_id = base_val['item_id']
res['item_id'] = list(val_item_id)
res['user_id'] = list(val_user_id)
res.to_csv('./res.csv',index=False)


res_2 = pd.DataFrame()
res_2['user_id'] = list(sub_user_id)
res_2['sub_pre'] = list(y_sub)
res_2['item_id'] = list(sub_item_id)
# res['pre'] = res['pre'].astype(int)

res_2.to_csv('./res_2.csv',index=False)


# Early stopping, best iteration is:
# [776]	training's binary_logloss: 0.0159666	valid_1's binary_logloss: 0.0181321

# [57]	training's auc: 0.847233	valid_1's auc: 0.83386
# [36]	training's auc: 0.847339	valid_1's auc: 0.835578

# [63]	training's auc: 0.886859	valid_1's auc: 0.853861

# [13]	training's auc: 0.824146	valid_1's auc: 0.813499
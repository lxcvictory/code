#encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
# 文件所在路径
Action_2 = './data/JData_Action_201602.csv'
Action_3 = './data/JData_Action_201603.csv'
Action_4 = './data/JData_Action_201604.csv'

user_id_2 = './numpy_file/0.npy'
user_id_3 = './numpy_file/1.npy'
user_id_4 = './numpy_file/2.npy'
# 从数据中读取所有和cate8有关的用户 保存为 numpy格式
def get_user_id():
    for i,action in enumerate([Action_2,Action_3,Action_4]):
        print(u"读取数据%s"%(action))
        org_action = pd.read_csv(action)
        # 获取原始数据中与cate8有关的用户
        print(u"获取原始数据中与cate8有关的用户")
        new_action = org_action[org_action['cate'] == 8]
        print(u"获取在与cate8有关的用户中购买cate8的用户")
        # 获取在与cate8有关的用户中购买cate8的用户
        buy_action = new_action[new_action['type'] == 4]
        print(u"将用户id存储在numpy文件中")
        user_id = buy_action['user_id'].unique()
        print(u"购买了cate的用户数",len(user_id))
        print(u"保存到./numpy_file/%d"%(i))
        np.save(u"./numpy_file/%d"%(i),user_id)
# 根绝读取到的用户在文件中获取到所有购买了cate8的用户的行为数据 
def get_user_action():
    user_id2 = np.load(user_id_2)
    user_id3 = np.load(user_id_3)
    user_id4 = np.load(user_id_4)

    for i,user_action in enumerate([Action_2,Action_3,Action_4]):
        print(u"获取行为数据",user_action)
        user_actions = pd.read_csv(user_action)
        buy_cate_8_user_action = pd.concat([
            user_actions[user_actions['user_id'].isin(user_id2)],
            user_actions[user_actions['user_id'].isin(user_id3)],
            user_actions[user_actions['user_id'].isin(user_id4)],
        ]
        )
        print(len(buy_cate_8_user_action['user_id'].unique()))
        buy_cate_8_user_action.to_csv("./create_data/%d.csv"%(i))
# 获取行为数据合并为一个数据查看其中的数据量
def merage_user_action():
    print(u"读取2月份购买cate_8的用户数据")
    base_0 = pd.read_csv("./create_data/0.csv",index_col=0)
    print(u"读取3月份购买cate_8的用户数据")
    base_1 = pd.read_csv("./create_data/1.csv",index_col=0)
    print(u"读取4月份购买cate_8的用户数据")
    base_2 = pd.read_csv("./create_data/2.csv",index_col=0)
    print(u"合并数据")
    user_action_buy_cate_8 = pd.concat([base_0,base_1,base_2],axis=0)
    print(u"保存数据")
    user_action_buy_cate_8.to_csv("./create_data/user_action_buy_cate_8.csv")
    print(u"75天购买的用户总数")
    print(len(user_action_buy_cate_8['user_id'].unique()))
# 分组计算日期
def calc_date():
    # 基准时间作为计算购买第一天到最后一天的时间间隔
    print(u"基准时间作为计算购买第一天到最后一天的时间间隔")
    base_time = pd.to_datetime('2016-01-30')
    # 读取购买过cate8的用户数据
    print(u"读取购买过cate8的用户数据")
    org_action = pd.read_csv("./create_data/user_action_buy_cate_8.csv")
    new_action = org_action.copy()
    print(u"购买行为的用户")
    new_action = new_action[new_action['type'] == 4]
    print(u"创建时间序列1-75")
    time_seqs = np.linspace(1,76,16)

    # 修改时间格式 yyyy-mm-dd
    print(u"修改时间格式 yyyy-mm-dd")
    new_action['newtime'] = pd.to_datetime(new_action['time']).map(lambda x : x.strftime("%Y-%m-%d"))
    print(u"创建cate的one hot")
    cate = pd.get_dummies(new_action['cate'],prefix='cate')
    print(u"合并cate和原始数据")
    new_action = pd.concat([new_action,cate],axis=1)
    # 由于1月31日属于无任何行为数据 所以数据从2月开始
    new_action['group'] = pd.to_datetime(new_action['newtime']).map(lambda x: int(((x - base_time).days) - 2))
    for time_seq in time_seqs:
    # for time_seq in [1]:
        time_seq = int(time_seq)
        if time_seq !=1:
            time_seq = time_seq - 1
        print("创建时间分组：%d天一组"%(time_seq))
        new_action['merage_group'] = new_action['group'].map(lambda x : int( x / time_seq ))
        for cate in [4,5,6,7,9,10,]:
            print("cate8 and cate%d"%(cate))
            # 备份数据
            tmp = new_action.copy()
            # 截取字段
            tmp = tmp[['user_id','merage_group','cate_4','cate_5','cate_6','cate_7','cate_8','cate_9','cate_10','cate_11']]
            tmp['user_id'] = tmp['user_id'].astype(int)
            out = tmp.groupby(['user_id','merage_group'],as_index=False).sum()
            c_out = out[(out['cate_%d'%(cate)]!=0)&(out['cate_8']!=0)]
            print(u"购买cate：%d"%(len(c_out['user_id'].unique())))


def get_all_user_id():
    user_ids = pd.read_csv('./create_data/user_action_buy_cate_8.csv')
    user_id_unique = user_ids['user_id'].unique()
    np.save('./unique_8/user_id_unique',user_id_unique)
def acc_comment():
    print "读取评论数据"
    comment = pd.read_csv('./data/JData_Comment.csv')
    print len(comment['sku_id'].unique())
    print "读取行为数据"
    action = pd.read_csv('./create_data/user_action_buy_cate_8.csv')
    action = action[action['cate'] == 8]
    sku_ids = action['sku_id'].unique()
    print len(sku_ids)
    comment = comment[comment['sku_id'].isin(sku_ids)]
    print np.mean(comment)
    action = pd.read_csv('./data/JData_Product.csv')
    action = action[action['sku_id'].isin(sku_ids)]
    print action
    print np.mean(action)
    print action['a1'].value_counts()
    print action['a2'].value_counts()
    print action['a3'].value_counts()


if __name__ == '__main__':
	get_user_id()
	get_user_action()
	merage_user_action()
	get_all_user_id()
 

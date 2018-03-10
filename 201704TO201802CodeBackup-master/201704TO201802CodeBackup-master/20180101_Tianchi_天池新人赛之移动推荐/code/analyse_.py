import pandas as pd
from sklearn.metrics import f1_score



def evaluate(p, v):
    pset = set(map(tuple, p.values.tolist()))
    vset = set(map(tuple, v.values.tolist()))
    hits_set = pset & vset

    hits = float(len(hits_set))
    pnums = float(len(pset))
    vnums = float(len(vset))

    print(
    'F1：',  format(2 * (hits / pnums) *  (hits / vnums )/ ((hits / pnums) + (hits / vnums )),'.2%')),
    print(
    '\t准确率：', format((hits / pnums), '.2%')),
    print(
    '\t召回率：', format(hits / vnums, '.2%')),
    print(
    '\thits：%d' % hits)


train_item = pd.read_csv('../data/tianchi_fresh_comp_train_item.csv')
train_xxitem = pd.read_csv('../result/result_x.csv')

# user_id,item_id
res = pd.read_csv('./res.csv')
res = res[res.item_id.isin(list(train_item.item_id.unique()))].drop_duplicates(['user_id','item_id'])

item_set = res[res['true']==1][['user_id','item_id']]

answer = pd.read_csv('../result/ansewr.csv')
answer = answer[answer.item_id.isin(list(train_item.item_id.unique()))].drop_duplicates(['user_id','item_id'])

evaluate(item_set[['user_id','item_id']],answer[['user_id','item_id']])

evaluate(train_xxitem[['user_id','item_id']],answer[['user_id','item_id']])
# print(item_set)

for i in [400,500,550,600,605,610,620,630,640,650,700]:
    print(i)
    pre_res = res[['user_id','item_id','pre']].sort_values(['pre'],ascending=False).drop_duplicates(['user_id','item_id'])[:i]
    evaluate(pre_res[['user_id','item_id']],answer[['user_id','item_id']])





res_2 = pd.read_csv('./res_2.csv')

res_2 = res_2[res_2.item_id.isin(list(train_item.item_id.unique()))]
res_2 = res_2.sort_values(['sub_pre'],ascending=False)[:700]
# print(res_2)


res_2[['user_id','item_id']].to_csv('../result/zs_20180207_1.csv',index=False)

xxx = pd.read_csv('../result/tianchi_mobile_recommendation_predict.csv')
resx = pd.read_csv('../result/result.csv')

evaluate(res_2[['user_id','item_id']],resx[['user_id','item_id']])
evaluate(xxx[['user_id','item_id']],resx[['user_id','item_id']])
evaluate(xxx[['user_id','item_id']],res_2[['user_id','item_id']])
# xx = pd.read_csv('../result/zs_20180202_1.csv')
# evaluate(xx[['user_id','item_id']],resx)

# 810
# F1： 0.13468013468013468
# 	准确率： 12.35%
# 	召回率： 14.81%
# 	hits：100


# 810
# F1： 0.12390572390572391
# 	准确率： 11.36%
# 	召回率： 13.63%
# 	hits：92


# 630
# F1： 1.04%
# 	准确率： 5.56%
# 	召回率： 0.57%
# 	hits：35


# 630
# F1： 5.15%
# 	准确率： 5.56%
# 	召回率： 4.79%
# 	hits：35


# 700
# F1： 5.31%
# 	准确率： 5.43%
# 	召回率： 5.21%
# 	hits：38

# 700
# F1： 5.87%
# 	准确率： 6.00%
# 	召回率： 5.75%
# 	hits：42

# 700
# F1： 5.87%
# 	准确率： 6.00%
# 	召回率： 5.75%
# 	hits：42


# 620
# F1： 6.07%
# 	准确率： 6.61%
# 	召回率： 5.62%
# 	hits：41


# 700
# F1： 6.29%
#         准确率： 6.43%
#         召回率： 6.16%
#         hits：45
#coding:utf-8
import pandas as pd
import numpy as np
from datetime import  timedelta




print('线上提交')
train = tra
tes = pd.read_csv('sample_submission.csv')
tes['visit_date'] = tes['id'].map(lambda x: str(x).split('_')[2])
tes['air_store_id'] = tes['id'].map(lambda x: '_'.join(x.split('_')[:2]))
tes = tes[['air_store_id', 'visit_date', 'visitors']]
test = tes[tes['visit_date'] <= '2017-04-30']
print("第一个周期构造特征")
train,test=make_feature(train,test)
print('返回test')
train,test=lgbCV(train, test)

################################开始循环################################
for i in range(1,6):
    print(i)
    train = pd.concat([train, test], axis=0)
    #train['visit_date'] = train['visit_date'].apply(datetime_toString)
    test = tes[tes['visit_date'] <= (pd.to_datetime('2017-04-23')+timedelta(days=7*(i+1))).strftime('%Y-%m-%d')]
    test = test[test['visit_date'] > (pd.to_datetime('2017-04-23')+timedelta(days=7*i)).strftime('%Y-%m-%d')]
    print((pd.to_datetime('2017-04-23') + timedelta(days=7 * (i + 1))).strftime('%Y-%m-%d'))
    print((pd.to_datetime('2017-04-23') + timedelta(days=7 * i)).strftime('%Y-%m-%d'))
    print("第i个周期构造特征")
    train, test = make_feature(train, test)
    print('返回test')
    train, test = lgbCV(train, test)

train = pd.concat([train, test], axis=0)

#################################线下总的rmse#############################
all_pred = train[train['visit_date'] >= '2017-04-23']
print(all_pred)
del tes['visitors']
for df in [tes, all_pred]:
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['visit_date'] = df['visit_date'].dt.date
tes = pd.merge(tes, all_pred, on=['air_store_id', 'visit_date'], how='left')

print(tes)
index_test = pd.read_csv('sample_submission.csv')
index_test = index_test['id']
result = pd.DataFrame({"id": index_test, "visitors": tes['visitors']})
result.to_csv('OSA_0116.csv', index=False)
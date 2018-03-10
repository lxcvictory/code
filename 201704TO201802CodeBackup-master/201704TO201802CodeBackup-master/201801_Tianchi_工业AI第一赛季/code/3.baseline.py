#coding:utf-8

import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse

fileDir = '../data/'
train = pd.read_excel(fileDir + u"训练.xlsx")
test1 = pd.read_excel(fileDir + u"测试A.xlsx")
# test2 = pd.read_excel(fileDir + u"测试B.xlsx")
org_col_num = train.shape[1]
new_col = []
same_col = []
for i in train.columns:
    miss_val = train[i].isnull().sum()
    if miss_val < 200:
        if train[i].dtypes != 'object':
            train_values = train[i].dropna().unique()
            if (np.std(train_values) != 0) and (len(train_values) > 1 ):
                if min(train_values) < 9000:
                    # print pearsonr(train[i].values,train['Y'].values)

                    tmp_corr = np.corrcoef(train[i].values,train['Y'].values)[0][1]
                        # print tmp_corr
                    if str(tmp_corr) != 'nan':
                        # print tmp_corr
                        if abs(float(tmp_corr)) >= 0.05:
                            new_col.append(i)
        else:
            new_col.append(i)

print "清洗数据",org_col_num - len(new_col)


train = train[list(new_col)]
train_Y = train.pop('Y')
# train = pd.get_dummies(train)
# test1 = pd.get_dummies(test1)

test1 = test1[train.columns]

print train.shape,test1.shape

from sklearn.preprocessing import StandardScaler
train = train.fillna(-1)
test1 = test1.fillna(-1)
new_test1_col = test1['ID']

for col in train.columns:
    if train[col].dtype == object:
        del train[col]
        del test1[col]

train = train.convert_objects(convert_numeric=True)
test1 = test1.convert_objects(convert_numeric=True)

# s = StandardScaler()
# train = s.fit_transform(train.values)
# test1 = s.fit_transform(test1.values)

from sklearn.model_selection import train_test_split


# x_train, x_test, y_train, y_test = train_test_split(train, train_Y,random_state=2017,test_size=0.2)


# print train
# print train_Y

# def try_different_method(clf):
#     clf.fit(x_train,y_train)
#     # score = mse(y_test,x_test)
#     # score = clf.score(x_test, y_test,score='mse')
#     result = clf.predict(x_test)
#     score = mse(y_test,result)
#     plt.figure()
#     plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
#     plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
#     plt.title('score: %f'%score)
#     plt.legend()
#     plt.show()

from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble

# rf =ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
# svr = svm.SVR()
knn = neighbors.KNeighborsRegressor(n_neighbors=4)
from sklearn.model_selection import KFold

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results = []
sub_array = []
train = train.values
y_train = train_Y.values
test1 = test1.values

# model xgb _ cv
for model in [knn]:
    for traincv, testcv in cv.split(train,y_train):
        knn.fit(train[traincv], y_train[traincv])
        y_tmp = knn.predict(train[testcv])
        res = mse(y_train[testcv],y_tmp)
        results.append(res)

        sub_array.append(knn.predict(test1))
    print("Results: " + str( np.array(results).mean() ))


s = 0
for i in sub_array:
    s = s + i

r = pd.DataFrame()
r['ID'] = list(new_test1_col.values)
r['Y'] = list(s/5)

print(r[['ID', 'Y']])
#
r[['ID', 'Y']].to_csv('../result/result_20180109.csv',index=None,header=None)

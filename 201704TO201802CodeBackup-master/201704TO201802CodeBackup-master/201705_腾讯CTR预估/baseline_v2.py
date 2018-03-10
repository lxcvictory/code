# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
import gc
if __name__=='__main__':
    # warnings.filterwarnings("ignore")
    # load data
    data_root = "./org_data"
    dfTrain = pd.read_csv("%s/train.csv"%data_root)
    dfTest = pd.read_csv("%s/test.csv"%data_root)
    dfAd = pd.read_csv("%s/ad.csv"%data_root)
    dfApp_Cate = pd.read_csv("%s/app_categories.csv"%data_root)
    dfAd = pd.merge(dfAd, dfApp_Cate, on="appID", how="left")
    del dfApp_Cate
    gc.collect()
    dfPosition = pd.read_csv("%s/position.csv"%data_root)
    dfUser = pd.read_csv("%s/user.csv"%data_root)
    dfUser['age'] = dfUser['age'].map(lambda x : x / 15)
    # process data
    dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
    dfTrain = pd.merge(dfTrain, dfPosition, on="positionID")
    dfTrain = pd.merge(dfTrain, dfUser, on="userID")
    dfTrain['week'] = dfTrain['clickTime'].map(lambda x: (x/1000)%7)
    dfTest = pd.merge(dfTest, dfAd, on="creativeID")
    dfTest = pd.merge(dfTest, dfPosition, on="positionID")
    dfTest = pd.merge(dfTest, dfUser, on="userID")
    dfTest['week'] = dfTest['clickTime'].map(lambda x: (x/1000)%7)
    del dfUser
    del dfPosition
    gc.collect()
    y_train = dfTrain["label"].values

    # feature engineering/encoding
    enc = OneHotEncoder()
    feats = ["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform", "appCategory","positionID",
    		"sitesetID","positionType","connectionType","telecomsOperator","gender","education","age","marriageStatus"]


    for i,feat in enumerate(feats):
        x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
        x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
            # print x_test
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

    X_train,X_test = sparse.hstack((X_train,dfTrain['haveBaby'].values.reshape(-1,1))),sparse.hstack((X_test,dfTest['haveBaby'].values.reshape(-1,1)))
    print "++++++++++++++++++++"
    # print X_test
    # model training
    # pipeline = Pipeline([
    # 	('clr',LogisticRegression())
    # 	])

    # parameters = {
    # 	'clr__penalty': ('l1', 'l2'),
    # 	'clr__C': (0.01, 0.1, 1, 10),
    # }

    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    lr = LogisticRegression()


    lr.fit(X_train, y_train)
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print('\t%s: %r' % (param_name, best_parameters[param_name]))
    proba_test = lr.predict_proba(X_test)[:,1]


    # submission
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    with zipfile.ZipFile("submission.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
